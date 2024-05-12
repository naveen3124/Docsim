#include <doctest/doctest.h>
#include <docsim/docsim.h>
#include <docsim/version.h>

#include <string>

TEST_CASE("DocSim") {
  using namespace docsim;

  DocSim docsim("Tests");

  CHECK(docsim.greet(LanguageCode::EN) == "Hello, Tests!");
  CHECK(docsim.greet(LanguageCode::DE) == "Hallo Tests!");
  CHECK(docsim.greet(LanguageCode::ES) == "¡Hola Tests!");
  CHECK(docsim.greet(LanguageCode::FR) == "Bonjour Tests!");
}

TEST_CASE("DocSim version") {
  static_assert(std::string_view(DOCSIM_VERSION) == std::string_view("1.0"));
  CHECK(std::string(DOCSIM_VERSION) == std::string("1.0"));
}
