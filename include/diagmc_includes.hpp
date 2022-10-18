#pragma once

/* Standard library includes */
#include <algorithm>
#include <cassert>
#include <cmath>  // std::pow
#include <cstddef>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>  // std::random_device
#include <sstream>
#include <stdexcept>
#include <type_traits>

/* Standard data structures */
#include <array>
#include <bitset>
#include <complex>
#include <map>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

// NOTE: We have used the stdlib random_device for simplicity.
//       This will not generate true random seeds on all systems,
//       so should be changed to the following in the future!
// #include <boost/random/random_device.hpp>

/* HDF5 library include */
#include "H5Cpp.h"

/* JSON library include */
#define JSON_USE_IMPLICIT_CONVERSIONS 0  // Turn off implicit conversions from JSON values
#include "json.hpp"

// Injects simple std::optional conversion functions into the nlohmann JSON library
// (see: https://github.com/nlohmann/json/issues/1749#issuecomment-772996219).
//
// NOTE: This feature will be added to the library more robustly in a future release!
//       (see: https://github.com/nlohmann/json/pull/2117,
//             https://github.com/nlohmann/json/pull/2229)
namespace nlohmann {

template <class T>
void to_json(nlohmann::json &j, const std::optional<T> &v) {
  if (v.has_value())
    j = *v;
  else
    j = nullptr;
}

template <class T>
void from_json(const nlohmann::json &j, std::optional<T> &v) {
  if (j.is_null())
    v = std::nullopt;
  else
    v = j.get<T>();
}

}  // namespace nlohmann