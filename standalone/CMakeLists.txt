cmake_minimum_required(VERSION 3.14...3.22)

project(DocSimStandalone LANGUAGES CXX)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage(
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 3.0.0
  OPTIONS "CXXOPTS_BUILD_EXAMPLES NO" "CXXOPTS_BUILD_TESTS NO" "CXXOPTS_ENABLE_INSTALL YES"
)

CPMAddPackage(NAME DocSim SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Find zlib package
find_package(ZLIB REQUIRED)

find_package(Faiss REQUIRED)

find_package(OpenMP REQUIRED)


if(ZLIB_FOUND)
    message(STATUS "Found zlib: ${ZLIB_INCLUDE_DIRS}, ${ZLIB_LIBRARIES}")
else()
    message(FATAL_ERROR "Zlib not found!")
endif()

# Include zlib headers
include_directories(${ZLIB_INCLUDE_DIRS})

include(ExternalProject)

# Add hiredis as an external project
ExternalProject_Add(
    hiredis
    PREFIX ${CMAKE_BINARY_DIR}/external/hiredis
    GIT_REPOSITORY https://github.com/redis/hiredis.git
    GIT_TAG v1.2.0  # Specify the version/tag you want to use
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -C ${CMAKE_BINARY_DIR}/external/hiredis/src/hiredis
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

include_directories(${CMAKE_BINARY_DIR}/external/hiredis/src/hiredis)
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# ---- Create standalone executable ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

add_executable(${PROJECT_NAME} ${sources})

add_dependencies(${PROJECT_NAME} hiredis)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 OUTPUT_NAME "DocSim")

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        DocSim::DocSim
        cxxopts
        faiss
        ${ZLIB_LIBRARIES}
        OpenMP::OpenMP_CXX
    PRIVATE
        nlohmann_json::nlohmann_json
        ${CMAKE_BINARY_DIR}/external/hiredis/src/hiredis/libhiredis.a
)
