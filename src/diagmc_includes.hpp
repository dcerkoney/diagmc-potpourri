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
// NOTE: we use the stdlib random_device instead,
//       as the boost random library must be linked
//       explicitly for this functionality to work,
//       and it is not yet clear to me how to achieve
//       this using the cpp2py/cmake wrapping framework.
// TODO:
//       This will not generate true random seeds on all
//       systems, so should be changed in the future!
// #include <boost/random/random_device.hpp>
