#pragma once

/* Standard library includes */
#include <algorithm>
#include <cassert>
#include <cmath>  // std::pow
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>  // std::random_device
#include <stdexcept>

/* Data structures */
#include <array>
#include <bitset>
#include <cmath>
#include <complex>
#include <map>
#include <string>
#include <tuple>
#include <vector>

/* HDF5 includes */
#include "H5Cpp.h"

/* Boost includes */
#include <boost/math/distributions/arcsine.hpp>
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
