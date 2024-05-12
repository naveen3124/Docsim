#include <docsim/docsim.h>
#include <docsim/version.h>
#include <zlib.h>
#include <random>
#include <cxxopts.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <hiredis.h> // Include hiredis header file
#include <nlohmann/json.hpp>
#include "Base64.h"  // Include the header file that defines Base64 class
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <zlib.h>
#include <iconv.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
using json = nlohmann::json;

// Function to decompress zlib-compressed data
std::vector<char> decompressZlib(const std::string& compressed_data) {
    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;
    stream.avail_in = static_cast<uInt>(compressed_data.size());
    stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compressed_data.data()));

    int ret = inflateInit(&stream);
    if (ret != Z_OK) {
        std::cerr << "Error initializing zlib: " << stream.msg << std::endl;
        return {}; // Return an empty vector on error
    }

    std::vector<char> decompressed_data; // Initially empty
    const int chunk_size = 1024; // Chunk size for decompression
    char chunk[chunk_size];

    do {
        stream.avail_out = chunk_size;
        stream.next_out = reinterpret_cast<Bytef*>(chunk);

        ret = inflate(&stream, Z_NO_FLUSH);
        if (ret == Z_STREAM_ERROR) {
            std::cerr << "Error during decompression: " << stream.msg << std::endl;
            inflateEnd(&stream);
            return {}; // Return an empty vector on error
        }

        decompressed_data.insert(decompressed_data.end(), chunk, chunk + (chunk_size - stream.avail_out));
    } while (ret != Z_STREAM_END);

    inflateEnd(&stream);

    return decompressed_data;
}

std::string decodeToUTF8(const std::vector<char>& data, const std::string& source_encoding) {
    std::string utf8_data;
    iconv_t conv = iconv_open("UTF-8", source_encoding.c_str());
    if (conv == (iconv_t)(-1)) {
        std::cerr << "Error opening iconv: " << errno << std::endl;
        return ""; // Return an empty string on error
    }

    const char* src_ptr = data.data();
    size_t src_size = data.size();
    size_t dst_size = src_size * 2; // Assume max double the size for UTF-8
    char* dst_ptr = new char[dst_size];
    char* dst_start = dst_ptr;

    if (iconv(conv, const_cast<char**>(&src_ptr), &src_size, &dst_ptr, &dst_size) == (size_t)(-1)) {
        std::cerr << "Error converting data: " << errno << std::endl;
        iconv_close(conv);
        delete[] dst_start;
        return ""; // Return an empty string on error
    }

    utf8_data.assign(dst_start, dst_ptr - dst_start);
    iconv_close(conv);
    delete[] dst_start;

    return utf8_data;
}
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int checkfaiss() {
    double t0 = elapsed();

    // dimension of the vectors to index
    int d = 128;

    // size of the database we plan to index
    size_t nb = 200 * 1000;

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 100 * 1000;

    // make the index object and train it
    faiss::IndexFlatL2 coarse_quantizer(d);

    // a reasonable number of centroids to index nb vectors
    int ncentroids = int(4 * sqrt(nb));

    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)
    faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, 4, 8);

    std::mt19937 rng;

    { // training
        printf("[%.3f s] Generating %ld vectors in %dD for training\n",
               elapsed() - t0,
               nt,
               d);

        std::vector<float> trainvecs(nt * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }

        printf("[%.3f s] Training the index\n", elapsed() - t0);
        index.verbose = true;

        index.train(nt, trainvecs.data());
    }

    { // I/O demo
        const char* outfilename = "/tmp/index_trained.faissindex";
        printf("[%.3f s] storing the pre-trained index to %s\n",
               elapsed() - t0,
               outfilename);

        write_index(&index, outfilename);
    }

    size_t nq;
    std::vector<float> queries;

    { // populating the database
        printf("[%.3f s] Building a dataset of %ld vectors to index\n",
               elapsed() - t0,
               nb);

        std::vector<float> database(nb * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);

        index.add(nb, database.data());

        printf("[%.3f s] imbalance factor: %g\n",
               elapsed() - t0,
               index.invlists->imbalance_factor());

        // remember a few elements from the database as queries
        int i0 = 1234;
        int i1 = 1243;

        nq = i1 - i0;
        queries.resize(nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries[(i - i0) * d + j] = database[i * d + j];
            }
        }
    }

    { // searching the database
        int k = 5;
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);

        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        index.search(nq, queries.data(), k, dis.data(), nns.data());

        printf("[%.3f s] Query results (vector ids, then distances):\n",
               elapsed() - t0);

        for (int i = 0; i < nq; i++) {
            printf("query %2d: ", i);
            for (int j = 0; j < k; j++) {
                printf("%7ld ", nns[j + i * k]);
            }
            printf("\n     dis: ");
            for (int j = 0; j < k; j++) {
                printf("%7g ", dis[j + i * k]);
            }
            printf("\n");
        }

        printf("note that the nearest neighbor is not at "
               "distance 0 due to quantization errors\n");
    }

    return 0;
}
auto main(int argc, char** argv) -> int {
	checkfaiss();
	const std::unordered_map<std::string, docsim::LanguageCode> languages{
		{"en", docsim::LanguageCode::EN},
			{"de", docsim::LanguageCode::DE},
			{"es", docsim::LanguageCode::ES},
			{"fr", docsim::LanguageCode::FR},
	};

	cxxopts::Options options(*argv, "A program to build the similiarity index of legal documents");

	std::string language;
	std::string name;

	// clang-format off
	options.add_options()
		("h,help", "Show help")
		("v,version", "Print the current version number")
		("n,name", "Name to greet", cxxopts::value(name)->default_value("World"))
		("l,lang", "Language code to use", cxxopts::value(language)->default_value("en"))
		;
	// clang-format on

	auto result = options.parse(argc, argv);

	if (result["help"].as<bool>()) {
		std::cout << options.help() << std::endl;
		return 0;
	}

	if (result["version"].as<bool>()) {
		std::cout << "DocSim, version " << DOCSIM_VERSION << std::endl;
		return 0;
	}

	docsim::DocSim docsim(name);

	redisContext *ctx = redisConnect("127.0.0.1", 6380);
	if (ctx == nullptr || ctx->err) {
		if (ctx) {
			std::cerr << "Connection error: " << ctx->errstr << std::endl;
			redisFree(ctx);
		} else {
			std::cerr << "Failed to allocate Redis ctx" << std::endl;
		}
		return 1;
	} else {
		std::cout << "success grand success" << std::endl;
	}

    const char *hash_name = "id_to_doc_map";
    long long cursor = 0;
    redisReply *reply;
    do {
        reply = (redisReply *)redisCommand(ctx, "HSCAN %s %lld COUNT 10", hash_name, cursor);
        if (reply == NULL) {
            std::cerr << "HSCAN error" << std::endl;
            redisFree(ctx);
            return EXIT_FAILURE;
        }

        if (reply->type == REDIS_REPLY_ARRAY && reply->elements >= 2) {
            // Process the keys and values returned in the reply
            for (size_t i = 0; i < reply->element[1]->elements; i += 2) {
				std::cout << "Key: " << reply->element[1]->element[i]->str << std::endl;
				std::string value = reply->element[1]->element[i + 1]->str;
                json parsedValue = json::parse(value);
				if (parsedValue["_values"].contains("case_judgement")) {
					// Extract the value of the "case_judgement" key
					std::string caseJudgement = parsedValue["_values"]["case_judgement"];
					std::string decodedData;
					macaron::Base64::Decode(caseJudgement, decodedData);
					std::vector<char> decompressed_data = decompressZlib(decodedData);
					if (decompressed_data.empty()) {
						std::cerr << "Decompression failed!" << std::endl;
						return 1;
					}

					// Decode the decompressed data to UTF-8 (assuming source encoding is ASCII)
					std::string utf8_data = decodeToUTF8(decompressed_data, "ASCII");

					if (!utf8_data.empty()) {
						// Output the decoded UTF-8 data
						std::cout << "Decoded data (UTF-8):\n" << utf8_data << std::endl;
					} else {
						std::cerr << "Decoding failed!" << std::endl;
					}
				}
			}
            // Update the cursor for the next iteration
            cursor = strtoll(reply->element[0]->str, NULL, 10);
        }

        freeReplyObject(reply);
    } while (cursor != 10);

    redisFree(ctx);
    std::cout << "Indexing Successful" << std::endl;
    
    return EXIT_SUCCESS;
    
}



