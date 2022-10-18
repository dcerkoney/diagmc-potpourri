#pragma once

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

// Distribution/generator typedefs
typedef boost::random::mt19937 Rand_engine;
typedef boost::math::binomial Binom_dist;
typedef boost::random::binomial_distribution<int> Binom_gen;
typedef boost::random::uniform_int_distribution<int> DUnif_gen;
typedef boost::random::uniform_real_distribution<double> Unif_gen;
typedef boost::random::discrete_distribution<int, double> Discr_gen;

#if BOOST_VERSION >= 106500
typedef boost::math::arcsine_distribution<double> Arcsin_dist;
// Arcsine distribution for imaginary time generation
double arcsin_gen(Rand_engine rand_gen, Unif_gen std_uniform);
#endif
