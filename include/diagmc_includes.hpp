#pragma once

/* Standard library includes */
#include <algorithm>
#include <cassert>
#include <cmath>  // std::pow
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>  // std::random_device
#include <stdexcept>
#include <type_traits>

/* Standard data structures */
#include <array>
#include <bitset>
#include <complex>
#include <map>
#include <string>
#include <tuple>
#include <vector>

/* Boost includes */
#include <boost/version.hpp>
#if BOOST_VERSION >= 106500
#include <boost/math/distributions/arcsine.hpp>
#endif
#include <boost/math/distributions/binomial.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

// NOTE: We have used the stdlib random_device for simplicity.
//       This will not generate true random seeds on all systems,
//       so should be changed to the following in the future!
// #include <boost/random/random_device.hpp>

/* HDF5 header */
#include "H5Cpp.h"