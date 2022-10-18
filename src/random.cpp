#include "random.hpp"

// Some common distributions to be used in the Metropolis step and configuration updates.
// Unif_gen std_uniform(0.0, 1.0);
// DUnif_gen coin_flip(0, 1);
// template <int n>
// DUnif_gen roll_1dn(0, n - 1);
// DUnif_gen roll_1d3(0, 3);

#if BOOST_VERSION >= 106500
Arcsin_dist arcsin_dist(0.0, 1.0);
// Arcsine distribution for imaginary time generation
double arcsin_gen(Rand_engine rand_gen, Unif_gen std_uniform) {
  return boost::math::quantile(arcsin_dist, std_uniform(rand_gen));
}
#endif