#pragma once
#include "diagmc_includes.hpp"

/* cmath constant definitions */
#define _USE_MATH_DEFINES

// Distribution/generator typedefs
typedef boost::random::mt19937 Rand_engine;
typedef boost::math::binomial Binom_dist;
typedef boost::random::binomial_distribution<int> Binom_gen;
typedef boost::random::uniform_int_distribution<int> DUnif_gen;
typedef boost::random::uniform_real_distribution<double> Unif_gen;
typedef boost::random::discrete_distribution<int, double> Discr_gen;

// Some common distributions to be used in the Metropolis step and configuration updates.
// Unif_gen std_uniform(0.0, 1.0);
// DUnif_gen coin_flip(0, 1);
// template <int n>
// DUnif_gen roll_1dn(0, n - 1);

// Repeat a string s n times
std::string repeat(int n, std::string s) {
  std::ostringstream os;
  for (int i = 0; i < n; i++) {
    os << s;
  }
  return os.str();
}

// Unicode divider lines
std::string BULLET_PT = "\u2022";
std::string BOLD_DIVIDER_SMALL = repeat(20, "\u2550");
std::string DIVIDER_MEDIUM = repeat(51, "\u2500");

class not_implemeted_error : public virtual std::logic_error {
 public:
  using std::logic_error::logic_error;
};

#if __cplusplus >= 201703L
// Computes (a modulo b) for integral or floating types
// following the standard mathematical (Pythonic) convention
// for negative numbers, i.e., pymod(a,b) = (b + (a % b)) % b.
template <typename T>
constexpr T pymod(T a, T b) {
  static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value);
  if constexpr (std::is_integral<T>::value) {
    return (b + (a % b)) % b;
  } else {
    return std::fmod(b + std::fmod(a, b), b);
  }
}
#else
// Computes (a modulo b) for integers following
// the standard mathematical (Pythonic) convention for
// negative numbers, i.e., pymod(a,b) = (b + (a % b)) % b.
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
constexpr T pymod(T a, T b) {
  static_assert(std::is_integral<T>::value);
  return (b + (a % b)) % b;
}
// Computes (a modulo b) for floating-point doubles
// following the standard mathematical (Pythonic) convention
// for negative numbers, i.e., pymod(a,b) = (b + (a % b)) % b.
template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
constexpr T pymod(T a, T b) {
  static_assert(std::is_floating_point<T>::value);
  return std::fmod(b + std::fmod(a, b), b);
}
#endif

// Type-safe signum function template; see:
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
// NOTE: returns a double for valid multiplication with complex numbers (for
// which int will fail)
template <typename T>
constexpr double sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

// Computes nCk without unnecessary factorial computations; see:
// https://stackoverflow.com/questions/9330915/number-of-combinations-n-choose-r-in-c
template <typename T>
constexpr T n_choose_k(int n, int k) {
  // Handle edge cases and take advantage of the symmetry about n / 2
  if (k > n) {
    return 0;
  }
  if (2 * k > n) {
    k = n - k;
  }
  if (k == 0) {
    return 1;
  }
  // Compute the result without direct use of factorials
  T result = n;
  for (int i = 2; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}

// Returns the radius of a d-dimensional ball as a function of its volume and dimension
constexpr double rad_d_ball(double vol, int dim) {
  return std::pow(vol * std::tgamma(1.0 + (dim / 2.0)), 1.0 / static_cast<double>(dim)) /
         std::sqrt(M_PI);
}

// Emulates np.allclose from python, applied to singleton arrays (two floating point numbers);
// see: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
constexpr bool are_close(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
  return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ...
// i_first_bos = {2, if poln; 0, o/w}
// (...).test((i_v - i_first_bos) / 2);
//
// Returns true if, in the current set of temporal constraints,
// the time tau[i_vert] is to be pinned to tau[i_vert - 1].
constexpr bool is_pinned(int i_constr, int i_vert, int n_legs, int i_first_bos, int diag_type) {
  std::bitset<32> constraint_bitset(i_constr);
  // The two external legs are always unpinned,
  // except for static self energy diagrams
  if ((diag_type != 2) && (i_vert < n_legs)) {
    return false;
  }
  // For dynamic self energy diagrams, the two external
  // boson lines break the alternating convention, and
  // hence are hard-coded as special cases here
  else if ((diag_type == 3) && (i_vert == 2)) {
    return constraint_bitset.test(0);
  } else if ((diag_type == 3) && (i_vert == 3)) {
    return constraint_bitset.test(1);
  }
  // Barring the above special cases, even vertices
  // are always unpinned (by convention)
  else if (i_vert % 2 == 0) {
    return false;
  }
  // For all other vertices, check if they are pinned
  // in the current set of (temporal) constraints,
  // defined wlog by the integer i_constr
  else {
    return constraint_bitset.test((i_vert - i_first_bos) / 2);
  }
}

bool string_contains(std::string string, std::string substring) {
  return (string.find(substring) != std::string::npos);
}

// Parameter configuration for MCMC on the square lattice Hubbard model
struct hub_2dsqlat_mcmc_config {
  /* Diagram parameters */
  struct diag_config {
    std::string diag_type;
    std::vector<int> subspaces;
    double norm_space_weight = 1.0;
    int order;
    int n_legs;
    int n_intn;
    int n_times;
    int n_posns;
  } diag;
  /* MCMC parameters */
  struct mcmc_config {
    bool debug = false;
    bool verbose = true;
    bool normalize = true;
    bool save_serial = false;
    bool use_batch_U = false;
    int n_warm = 100000;
    int n_meas = 5000000;
    int n_skip = 1;
    int n_threads = 1;
    int n_nu_meas = 1;
    int n_k_meas = 1;
    int max_posn_shift = 1;
    // job_id and save_dir fields are set at runtime, and hence
    // may be undefined during initial conversion from JSON
    std::optional<std::time_t> job_id;
    std::optional<std::string> save_dir;
    std::string save_name;
  } mcmc;
  /* Physical parameters */
  struct phys_config {
    int dim;
    int n_site;
    int n_site_pd;
    int n_site_irred;
    int num_elec;
    std::optional<int> n_band;
    double lat_const;
    double lat_length;
    double vol_lat;
    // Either target_mu or target_n0 may be defined, but not both
    std::optional<double> target_mu;
    std::optional<double> target_n0;
    double mu_tilde;
    double mu;
    double n0;
    double rs;
    double ef;
    double beta;
    double t_hop;
    double s_ferm;
    double U_loc;
    std::vector<double> U_batch;
  } phys;
  /* Propagator parameters */
  struct propr_config {
    double delta_tau;
    int n_nu;
    int n_tau;
    std::time_t job_id;
    std::string save_dir;
  } propr;
};

// // Multiband MCMC config refinement
// struct mb_hub_2dsqlat_mcmc_config : hub_2dsqlat_mcmc_config {
//   struct phys_config : hub_2dsqlat_mcmc_config::phys_config {
//     int n_band;
//   } phys;
// };

// Macros to define JSON (de)serialization methods to_json/from_json
// for each mcmc config group (all children/parent structs) concisely
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::diag_config, diag_type, subspaces,
                                   norm_space_weight, order, n_legs, n_intn, n_times, n_posns)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::mcmc_config, debug, verbose, normalize,
                                   save_serial, use_batch_U, n_warm, n_meas, n_skip, n_threads,
                                   n_nu_meas, n_k_meas, max_posn_shift, job_id, save_dir, save_name)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::phys_config, dim, n_site, n_site_pd,
                                   n_site_irred, num_elec, n_band, lat_const, lat_length, vol_lat,
                                   target_mu, target_n0, mu_tilde, mu, n0, rs, ef, beta, t_hop,
                                   s_ferm, U_loc, U_batch)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::propr_config, delta_tau, n_nu, n_tau,
                                   job_id, save_dir)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config, diag, mcmc, phys, propr)

namespace hdf5 {

// Build an HDF5 compound datatype for complex numbers out of complex structs
typedef struct complex_t {
  double re;
  double im;
  complex_t(double re_, double im_) : re(re_), im(im_) {}
} complex_t;

class ComplexType {
 public:
  const static H5::CompType &COMP_COMPLEX;
  ComplexType() {
    // Initialize HDF5 compound datatype for complex numbers
    H5::CompType COMP_COMPLEX(sizeof(complex_t));
    COMP_COMPLEX.insertMember("re", HOFFSET(complex_t, re), H5::PredType::NATIVE_DOUBLE);
    COMP_COMPLEX.insertMember("im", HOFFSET(complex_t, im), H5::PredType::NATIVE_DOUBLE);
  }
};

// Enum for implemented attribute kinds for reads/writes (we enforce strict size checks on each)
enum class attr_kind {
  boolean,
  string,
  integral,
  real,
  complex,
};

// Map to native sizes of each attribute kind
std::unordered_map<attr_kind, std::size_t> attr_sizes = {
    {attr_kind::boolean, sizeof(bool)},      {attr_kind::string, sizeof(const char *)},
    {attr_kind::integral, sizeof(long)},     {attr_kind::real, sizeof(double)},
    {attr_kind::complex, sizeof(complex_t)},
};

// Map to unique pointers of H5::DataType derivatives for each attribute kind
std::unordered_map<attr_kind, std::unique_ptr<H5::DataType>> attr_type_ptrs = {
    {attr_kind::boolean, std::make_unique<H5::PredType>(H5::PredType::NATIVE_HBOOL)},
    {attr_kind::string, std::make_unique<H5::StrType>(0, H5T_VARIABLE)},
    {attr_kind::integral, std::make_unique<H5::PredType>(H5::PredType::NATIVE_LONG)},
    {attr_kind::real, std::make_unique<H5::PredType>(H5::PredType::NATIVE_DOUBLE)},
    {attr_kind::complex, std::make_unique<H5::CompType>(ComplexType::COMP_COMPLEX)},
};

// Perform some simple checks on H5 data/pred types (to avoid unexpected casting)
template <typename Tattr>
void verify_attribute_type(std::string attr_name, H5::DataType data_type) {
  // Use type traits to deduce the correct H5::PredType and native size
  attr_kind kind;
  if constexpr (std::is_same<Tattr, bool>::value) {
    kind = attr_kind::boolean;
  } else if (std::is_same<Tattr, std::string>::value) {
    kind = attr_kind::string;
  } else if (std::is_integral<Tattr>::value) {
    kind = attr_kind::integral;
  } else if (std::is_floating_point<Tattr>::value) {
    kind = attr_kind::real;
  } else if (std::is_same<Tattr, complex_t>::value) {
    kind = attr_kind::complex;
  } else {
    throw not_implemeted_error("Attribute storage for H5 type of attr_buffer '" + attr_name +
                               "' not yet implemented.");
  }
  std::size_t expected_size = attr_sizes.at(kind);
  H5T_class_t expected_class = attr_type_ptrs.at(kind)->getClass();
  // Test that the H5 data/pred type and size information agree
  if (data_type.getClass() != expected_class) {
    throw std::runtime_error("Unable to read " + attr_name + ", incorrect data-type");
  }
  if (data_type.getSize() != expected_size) {
    throw std::runtime_error("Unable to read " + attr_name + ", incorrect precision");
  }
  return;
}

// Read an attribute from an HDF5 location and return data of corresponding predtype
// NOTE: may throw an exception, which should be caught!
template <typename Tattr, typename Tloc = H5::Group>
Tattr read_attribute(const std::string &attr_name, const Tloc &h5loc) {
  if constexpr (!std::is_base_of<H5::Attribute, Tattr>::value) {
    throw std::runtime_error("Tattr is not an H5 attribute.");
  }
  if constexpr (!std::is_base_of<H5::Group, Tloc>::value &&
                !std::is_same<H5::DataSet, Tloc>::value) {
    throw std::runtime_error("Tloc is not a valid H5 file, group, or dataset.");
  }
  H5::Attribute attr = h5loc.openAttribute(attr_name);
  H5::DataType attr_type = attr.getDataType();
  // Verify that this attribute is loadable (correct H5::DataType and type size)
  verify_attribute_type<Tattr>(attr_name, attr_type);
  // Read the H5 attribute into the buffer
  Tattr attr_buffer;
  attr.read(attr_type, &attr_buffer);
  return attr_buffer;
}

// Write a parameter to an HDF5 location as an attribute
// NOTE: may throw an exception, which should be caught!
template <typename Tattr, typename Tloc = H5::Group>
void write_attribute(std::string param_name, Tattr param, Tloc h5loc) {
  if constexpr (!std::is_base_of<H5::Attribute, Tattr>::value) {
    throw std::runtime_error("Tattr is not an H5 attribute.");
  }
  if constexpr (!std::is_base_of<H5::Group, Tloc>::value &&
                !std::is_same<H5::DataSet, Tloc>::value) {
    throw std::runtime_error("Tloc is not a valid H5 file, group, or dataset.");
  }
  // Use type traits to deduce the correct H5::PredType and native size
  attr_kind kind;
  if constexpr (std::is_same<Tattr, bool>::value) {
    kind = attr_kind::boolean;
  } else if (std::is_same<Tattr, std::string>::value) {
    kind = attr_kind::string;
  } else if (std::is_integral<Tattr>::value) {
    kind = attr_kind::integral;
  } else if (std::is_floating_point<Tattr>::value) {
    kind = attr_kind::real;
  } else if (std::is_same<Tattr, complex_t>::value) {
    kind = attr_kind::complex;
  } else {
    throw not_implemeted_error("Attribute storage for H5 type of param '" + param_name +
                               "' not yet implemented.");
  }
  H5std_string attr_name(param_name);
  H5::DataSpace attr_space(H5S_SCALAR);
  H5::DataType attr_type = *attr_type_ptrs.at(kind);
  H5::Attribute attr = h5loc.createAttribute(attr_name, attr_type, attr_space);
  attr.write(attr_type, &param);
  return;
}

// Check if a specified attribute in an H5File is equal to an expected value
template <typename T>
bool attribute_equals(const T &expected_value, const std::string &attr_name,
                      const std::string &filename, const H5::H5File &h5file) {
  const T &param_found = read_attribute<T, H5::H5File>(attr_name, h5file);
  return (param_found == expected_value);
}

}  // end namespace hdf5

// Represent a 2D lattice of arbitrary objects using
// a variable-size contiguous (1D) std::vector
template <typename T>
class vector_2d {
 public:
  // int N_i;
  int N_j;
  std::vector<T> data;
  vector_2d() = default;
  vector_2d(int N_j_, const std::vector<T> &data_) : N_j(N_j_), data(data_) {}
  // Index the 2D lattice
  constexpr const T &operator()(int i, int j) const { return data[j + N_j * i]; }
  constexpr const T &operator()(const std::vector<int> &n) const { return data[n[1] + N_j * n[0]]; }
};

// Represent a 3D lattice of arbitrary objects using
// a variable-size contiguous (1D) std::vector
template <typename T>
class vector_3d {
 public:
  // int N_i;
  int N_j;
  int N_k;
  std::vector<T> data;
  vector_3d() = default;
  vector_3d(int N_j_, int N_k_, const std::vector<T> &data_) : N_j(N_j_), N_k(N_k_), data(data_) {}
  // Index the 3D lattice
  constexpr const T &operator()(int i, int j, int k) const { return data[k + N_k * (j + N_j * i)]; }
  constexpr const T &operator()(const std::vector<int> &n) const {
    return data[n[2] + N_k * (n[1] + N_j * n[0])];
  }
};

template <typename T, int N>
class vector_nd {
 public:
  std::array<int, N> n_sites;
  std::vector<T> data;
  vector_nd() = default;
  vector_nd(const std::array<int, N> &n_sites_, const std::vector<T> &data_)
      : n_sites(n_sites_), data(data_) {}
  // Index the n-dimensional lattice (assumes len(nd_idx) == len(n_sites) = n)
  constexpr const T &operator()(const std::array<int, N> &nd_idx) const {
    int index = nd_idx[0];
    for (std::size_t i = 1; i < N; ++i) {
      index = nd_idx[i] + index * n_sites[i];
    }
    return data[index];
  }
};

// 1D interpolant class (used, e.g. to define continuous-time objects from tau grid data)
class interp_1d {
 public:
  std::vector<double> f_data;
  std::vector<double> x_grid;
  interp_1d() = default;
  interp_1d(const std::vector<double> &f_data_, const std::vector<double> &x_grid_)
      : f_data(f_data_), x_grid(x_grid_) {}
  // Use bilinear interpolation to evaluate the interpoland at any point
  double eval(double point) const { return linear_interp(point); }
  // Linear interpolation function; approximates f(x) from mesh data.
  // NOTE: This algorithm applies regardless of mesh uniformity!
  double linear_interp(double x) const {
    // If the values are outside the range of the x and y grids,
    // return 0 (i.e., use extrapolation with a fill value of zero)
    if ((x < x_grid.front()) || (x > x_grid.back())) {
      return 0;
    }
    // Identify the nearest mesh neighbors of the evaluation point (x, y)
    std::vector<double>::const_iterator iter_x2;
    iter_x2 = std::lower_bound(x_grid.begin(), x_grid.end(), x);
    // Special case: if the evaluation point contains x_grid[-1],
    // shift the bounding points left by 1 manually (std::next not
    // applicable if lower bound gives last point in grid)
    if (iter_x2 == x_grid.begin()) {
      iter_x2 = std::next(iter_x2);
    }
    // Now, we can reliably get the last two corners
    // of the bounding box, including edge cases
    auto iter_x1 = std::prev(iter_x2);
    // Get the actual indices associated with each iterator
    int idx_x1 = iter_x1 - x_grid.begin();
    int idx_x2 = iter_x2 - x_grid.begin();
    // Precompute the grid points x_1 and x_2
    double x_1 = x_grid[idx_x1];
    double x_2 = x_grid[idx_x2];
    // Precompute the function values at the grid points f_1 and f_2
    double f_1 = f_data[idx_x1];
    double f_2 = f_data[idx_x2];
    // Use bilinear interpolation to extrapolate
    // the function value at the evaluation point
    return (f_1 * (x_2 - x) + f_2 * (x - x_1)) / (x_2 - x_1);
  }
};

// 2D interpolant class (used, e.g., to define spatially continuous objects from r-grid data)
class interp_2d {
 public:
  std::vector<std::vector<double>> f_data;
  const std::vector<double> x_grid;
  const std::vector<double> y_grid;
  interp_2d() = default;
  interp_2d(const std::vector<std::vector<double>> &f_data_, const std::vector<double> &x_grid_,
            const std::vector<double> &y_grid_)
      : f_data(f_data_), x_grid(x_grid_), y_grid(y_grid_) {}
  // Use bilinear interpolation to evaluate the interpoland at any point
  double eval(const std::vector<double> &point) const { return bilinear_interp(point); }
  // Evaluates the antiperiodic extension of the interp_2d object in
  // a single periodic variable using information on the principle interval.
  // The periodic variable in the evaluation point is specified by ap_idx.
  double ap_eval(const std::vector<double> &point, double period, int ap_idx = 1) const {
    int sign = 1;
    std::vector<double> point_shifted = point;
    while (point_shifted[ap_idx] < 0) {
      sign *= -1;
      point_shifted[ap_idx] += period;
    }
    while (point_shifted[ap_idx] >= period) {
      sign *= -1;
      point_shifted[ap_idx] -= period;
    }
    return sign * bilinear_interp(point_shifted);
  }

 private:
  // Bilinear interpolation function
  double bilinear_interp(const std::vector<double> &point) const {
    // Define the x and y values at which to evaluate the function
    double x = point[0], y = point[1];
    // If the values are outside the range of the x and y grids,
    // return 0 (i.e., use extrapolation with a fill value of zero)
    if ((x < x_grid.front()) || (x > x_grid.back())) {
      return 0;
    }
    if ((y < y_grid.front()) || (y > y_grid.back())) {
      return 0;
    }
    // Identify the nearest mesh neighbors of the evaluation point (x, y)
    std::vector<double>::const_iterator iter_x2;
    std::vector<double>::const_iterator iter_y2;
    iter_x2 = std::lower_bound(x_grid.begin(), x_grid.end(), x);
    iter_y2 = std::lower_bound(y_grid.begin(), y_grid.end(), y);
    // Special case: if the evaluation point contains x_grid[-1]
    //  or y_grid[-1], shift the bounding square left by 1 manually
    // (std::next not applicable if lower bound gives last point in grid)
    if (iter_x2 == x_grid.begin()) {
      iter_x2 = std::next(iter_x2);
    }
    if (iter_y2 == y_grid.begin()) {
      iter_y2 = std::next(iter_y2);
    }
    // Now, we can reliably get the last two corners
    // of the bounding box, including edge cases
    // std::vector<double>::const_iterator iter_x2 = std::next(iter_x1) -
    // x_grid.begin(); std::vector<doublidx_x2e>::const_iterator iter_y2
    // = std::next(iter_y1) - y_grid.begin();
    auto iter_x1 = std::prev(iter_x2);
    auto iter_y1 = std::prev(iter_y2);
    // Get the actual indices associated with each iterator
    int idx_x1 = iter_x1 - x_grid.begin();
    int idx_x2 = iter_x2 - x_grid.begin();
    int idx_y1 = iter_y1 - y_grid.begin();
    int idx_y2 = iter_y2 - y_grid.begin();
    // Precompute the grid points x_1, x_2, y_1, y_2
    double x_1 = x_grid[idx_x1];
    double x_2 = x_grid[idx_x2];
    double y_1 = y_grid[idx_y1];
    double y_2 = y_grid[idx_y2];
    // Precompute the function values at the grid points f_11, f_12, f_21, f_22
    double f_11 = f_data[idx_x1][idx_y1];
    double f_12 = f_data[idx_x1][idx_y2];
    double f_21 = f_data[idx_x2][idx_y1];
    double f_22 = f_data[idx_x2][idx_y2];
    // Use bilinear interpolation to extrapolate
    // the function value at the evaluation point
    return (f_11 * (x_2 - x) * (y_2 - y) + f_12 * (x_2 - x) * (y - y_1) +
            f_21 * (x - x_1) * (y_2 - y) + f_22 * (x - x_1) * (y - y_1)) /
           ((x_2 - x_1) * (y_2 - y_1));
  }
};

// For a bosonic Green's function interpolant (periodic), the period is beta
class b_interp_1d : public interp_1d {
 public:
  double beta;
  b_interp_1d() = default;
  b_interp_1d(const std::vector<double> &f_data_, const std::vector<double> &x_grid_, double beta_)
      : interp_1d(f_data_, x_grid_), beta(beta_) {}
  // Evaluates the periodic extension of the bosonic Green's function
  // object using information on the principle interval [0, beta).
  double p_eval(double point) const { return eval(pymod<double>(point, beta)); }
};

// For a fermionic Green's function interpolant (antiperiodic), the period is beta
class f_interp_1d : public interp_1d {
 public:
  double beta;
  f_interp_1d() = default;
  f_interp_1d(const std::vector<double> &f_data_, const std::vector<double> &x_grid_, double beta_)
      : interp_1d(f_data_, x_grid_), beta(beta_) {}
  // Evaluates the antiperiodic extension of the fermionic Green's function
  // object using information on the principle interval [0, beta).
  double ap_eval(double point) const {
    int sign = 1;
    double point_shifted = point;
    while (point_shifted < 0) {
      sign *= -1;
      point_shifted += beta;
    }
    while (point_shifted >= beta) {
      sign *= -1;
      point_shifted -= beta;
    }
    return sign * eval(point_shifted);
  }
};

// Continuous-time bosonic/fermionic lattice Green's function types
typedef vector_2d<b_interp_1d> lattice_2d_b_interp;
typedef vector_2d<f_interp_1d> lattice_2d_f_interp;
typedef vector_3d<b_interp_1d> lattice_3d_b_interp;
typedef vector_3d<f_interp_1d> lattice_3d_f_interp;

// Multiband, continuous-time bosonic/fermionic lattice Green's
// function types (spatial dimensions plus two band indices)
typedef vector_nd<b_interp_1d, 4> mb_lattice_2d_b_interp;
// typedef vector_nd<f_interp_1d, 4> mb_lattice_2d_f_interp;
using mb_lattice_2d_f_interp = vector_nd<f_interp_1d, 4>;
typedef vector_nd<b_interp_1d, 5> mb_lattice_3d_b_interp;
typedef vector_nd<f_interp_1d, 5> mb_lattice_3d_f_interp;

// Product basis, continuous-time bosonic lattice Green's function
// types (spatial dimensions plus two product indices)
typedef vector_nd<b_interp_1d, 4> pb_lattice_2d_b_interp;
typedef vector_nd<b_interp_1d, 5> pb_lattice_3d_b_interp;

// template <int dim>
// using lattice_b_interp = std::conditional<dim == 2, lattice_2d_b_interp, lattice_3d_b_interp>;
// template <int dim>
// using lattice_f_interp = std::conditional<dim == 2, lattice_2d_f_interp, lattice_3d_f_interp>;

// Defines the (split bosonic/fermionic) edge list representation of a set of graphs; since
// (wlog) the bosonic edges are assumed equal for all graphs, we only define the list once
typedef std::array<int, 2> edge_t;
typedef std::vector<edge_t> edge_list;
typedef std::vector<edge_list> edge_lists;

// A pool of graphs in the edge list representation
// (assumed to share a common bosonic edge basis)
struct graphs_el {
  edge_list b_edge_list;
  edge_lists f_edge_lists;
  graphs_el() = default;
  graphs_el(const edge_list &b_edge_list_, const edge_lists &f_edge_lists_)
      : b_edge_list(b_edge_list_), f_edge_lists(f_edge_lists_) {}
};

// A pool of 3-point vertices each consisting of one boson
// and two fermion edges (directed in/out of the base vertex)
struct vertices_3pt_el {
  edge_list b_edge_list;
  edge_lists f_edge_in_lists;
  edge_lists f_edge_out_lists;
  vertices_3pt_el() = default;
  // vertices_3pt_el() : b_edge_list({}), f_edge_in_lists({}), f_edge_out_lists({}) {}
  // In the (spinless) Hubbard case, we can ignore the bosonic edges,
  // which just give an overall multiplicative factor for each diagram
  vertices_3pt_el(edge_lists f_edge_lists_)
      : f_edge_in_lists(sort_incoming(f_edge_lists_)),
        f_edge_out_lists(sort_outgoing(f_edge_lists_)) {}
  // For non-Hubbard-like cases, the bosonic edges should be explicitly supplied
  vertices_3pt_el(const edge_list &b_edge_list_, edge_lists f_edge_lists_)
      : b_edge_list(b_edge_list_),
        f_edge_in_lists(sort_incoming(f_edge_lists_)),
        f_edge_out_lists(sort_outgoing(f_edge_lists_)) {}
  // Sort in-place according to incoming edge vertices ({v_out, >v_in})
  edge_lists sort_incoming(edge_lists f_edge_lists) const {
    for (edge_list &f_edges : f_edge_lists) {
      std::sort(f_edges.begin(), f_edges.end(),
                [](const edge_t &a, const edge_t &b) { return a[1] < b[1]; });
    }
    return f_edge_lists;
  }
  // Sort in-place according to outgoing edge vertices ({>v_out, v_in})
  edge_lists sort_outgoing(edge_lists f_edge_lists) const {
    for (edge_list &f_edges : f_edge_lists) {
      std::sort(f_edges.begin(), f_edges.end(),
                [](const edge_t &a, const edge_t &b) { return a[0] < b[0]; });
    }
    return f_edge_lists;
  }
};

// Diagram pool class (set of graphs at fixed order, associated
// diagram info and current/proposal spacetime coordinates)
// using (split)  el graph representation
struct diagram_pool_el {
  double s_ferm;
  int order;
  int n_legs;
  int n_intn;
  int n_times;
  int n_posns;
  int n_verts;
  int n_diags;
  int n_spins_max = 0;
  std::vector<int> symm_factors;
  std::vector<int> n_loops = {0};
  std::vector<std::vector<std::vector<int>>> loops;
  graphs_el graphs;
  vertices_3pt_el nn_vertices;
  // An explicit default constructor (a 0-dimensional diagram pool)
  diagram_pool_el()
      : s_ferm(0.5),
        order(0),
        n_legs(0),
        n_intn(0),
        n_times(0),
        n_posns(0),
        n_verts(0),
        n_diags(1),
        symm_factors({1}),
        loops({}),
        graphs(),
        nn_vertices() {}
  // Diagram pool constructor for spinless (Hubbard, G0W0) integrators
  diagram_pool_el(const hub_2dsqlat_mcmc_config &config_, const graphs_el &graphs_,
                  const std::vector<int> &symm_factors_,
                  const vertices_3pt_el &nn_vertices_ = vertices_3pt_el())
      : s_ferm(config_.phys.s_ferm),
        order(config_.diag.order),
        n_legs(config_.diag.n_legs),
        n_intn(config_.diag.n_intn),
        n_times(config_.diag.n_times),
        n_posns(config_.diag.n_posns),
        n_verts(2 * config_.diag.order),
        n_diags(graphs_.f_edge_lists.size()),
        symm_factors(symm_factors_),
        graphs(graphs_),
        nn_vertices(nn_vertices_) {}
  // Diagram pool constructor for spinful or multiband integrators (require loops)
  diagram_pool_el(const hub_2dsqlat_mcmc_config &config_, const graphs_el &graphs_,
                  const std::vector<int> &symm_factors_, const std::vector<int> &n_loops_,
                  const std::vector<std::vector<std::vector<int>>> loops_,
                  const vertices_3pt_el &nn_vertices_ = vertices_3pt_el())
      : s_ferm(config_.phys.s_ferm),
        order(config_.diag.order),
        n_legs(config_.diag.n_legs),
        n_intn(config_.diag.n_intn),
        n_times(config_.diag.n_times),
        n_posns(config_.diag.n_posns),
        n_verts(2 * config_.diag.order),
        n_diags(graphs_.f_edge_lists.size()),
        n_spins_max(*max_element(std::begin(n_loops_), std::end(n_loops_))),
        symm_factors(symm_factors_),
        n_loops(n_loops_),
        loops(loops_),
        graphs(graphs_),
        nn_vertices(nn_vertices_) {}
};
typedef std::vector<diagram_pool_el> diagram_pools_el;

// Class representing a continuous (Euclidean) spacetime coordinate
class cont_st_coord {
 public:
  int id;
  double itime;
  std::vector<double> posn;
  cont_st_coord(int id_, double itime_, std::vector<double> posn_)
      : id(id_), itime(itime_), posn(posn_) {}
  // Overload - operator for calculation of spacetime 'distance' (positional
  // distance, temporal difference)
  std::vector<double> operator-(const cont_st_coord &start) const {
    std::vector<double> st_dist;
    // Spatial distance
    std::vector<double> del_posn;
    std::transform(posn.begin(), posn.end(), start.posn.begin(), std::back_inserter(del_posn),
                   std::minus<double>());
    st_dist[0] =
        std::sqrt(std::inner_product(del_posn.begin(), del_posn.end(), del_posn.begin(), 0));
    // Temporal distance, modulo beta
    st_dist[1] = itime - start.itime;
    return st_dist;
  }
};
typedef std::vector<cont_st_coord> cont_st_coords;

// Returns the position index vector n'_r equivalent to n_r in the first orthant
// of the lattice (relative to the center), so that sqrt(n'_r * n'_r) defines
// the proper distance
const std::vector<int> first_orthant(const std::vector<int> &nr, int n_site_pd) {
  std::vector<int> nr_first_orthant = nr;
  for (std::size_t i = 0; i < nr.size(); ++i) {
    nr_first_orthant[i] = std::min(std::abs(nr[i]), n_site_pd - std::abs(nr[i]));
  }
  return nr_first_orthant;
}

// Returns the momentum index vector n'_k equivalent to n_k in
// the first Brillouin zone of the reciprocal lattice
const std::vector<int> first_brillouin_zone(const std::vector<int> &nk, int n_site_pd) {
  std::vector<int> nk_1BZ = nk;
  // Move each component back into the 1BZ;
  // k_i \in [-\pi / a, \pi / a) => nk_i \in [floor(-N / 2), floor(N / 2) - 1)
  for (std::size_t i = 0; i < nk_1BZ.size(); ++i) {
    while (nk_1BZ[i] >= std::floor(n_site_pd / 2.0)) {
      nk_1BZ[i] -= n_site_pd;
    }
    while (nk_1BZ[i] < std::floor(-n_site_pd / 2.0)) {
      nk_1BZ[i] += n_site_pd;
    }
    // assert((nk_1BZ[i] >= std::floor(-n_site_pd / 2.0)) &&
    //        (nk_1BZ[i] < std::floor(n_site_pd / 2.0)));
  }
  return nk_1BZ;
}

// Class representing a hypercubic lattice space - (imaginary) time coordinate;
// position vectors are given in units of the lattice constant, i.e., they index
// the lattice, and the d-toroidal lattice metric is used for spatial distances
class lat_st_coord {
 public:
  bool debug;
  int id;
  int dim;
  int n_site_pd;
  double beta;
  double lat_const;
  double delta_tau;
  double itime;
  std::vector<int> posn;
  // Custom default constructor definition (so that we can resize a vector of coordinates)
  lat_st_coord() : id(0), lat_const(0.0), itime(0) {}
  // COM constructor definition (coordinates at the origin / user-supplied lattice parameters)
  lat_st_coord(int dim_, int n_site_pd_, double beta_, double delta_tau_, double lat_const_ = 1.0,
               bool debug_ = false)
      : debug(debug_),
        id(0),
        dim(dim_),
        n_site_pd(n_site_pd_),
        beta(beta_),
        lat_const(lat_const_),
        delta_tau(delta_tau_),
        itime(0),
        posn(std::vector<int>(dim_, 0)) {}
  // Standard constructor
  lat_st_coord(int id_, double itime_, std::vector<int> posn_, int n_site_pd_, double beta_,
               double delta_tau_, double lat_const_ = 1.0, bool debug_ = false)
      : debug(debug_),
        id(id_),
        dim(posn_.size()),
        n_site_pd(n_site_pd_),
        beta(beta_),
        lat_const(lat_const_),
        delta_tau(delta_tau_),
        itime(itime_),
        posn(posn_) {}
  // Overload - operator for calculation of lattice spacetime differences (v_end - v_start)
  std::tuple<std::vector<int>, double> operator-(const lat_st_coord &v_start) const {
    // Spatial distance in units of the lattice constant (nvec = rvec / a)
    std::vector<int> del_nr;
    // First fill del_nr with (v_end - v_start), then apply the lattice metric
    // component-wise
    std::transform(posn.begin(), posn.end(), v_start.posn.begin(), std::back_inserter(del_nr),
                   std::minus<int>());
    double del_tau;
    // Enforce normal ordering if this is a density-type loop
    if (id == v_start.id) {
      if (debug) {
        std::cout << "Density loop; setting tau = -delta_{tau} (normal-ordering)!" << std::endl;
      }
      del_tau = -delta_tau;
    } else {
      del_tau = itime - v_start.itime;
    }
    // Now we can return a tuple (r_end - r_start, tau_end - tau_start)
    return std::make_tuple(first_orthant(del_nr, n_site_pd), del_tau);
  }
  void print(std::streambuf *buffer = std::cout.rdbuf(), bool toprule = true,
             bool botrule = true) const {
    std::ostream out(buffer);
    if (toprule) {
      out << "\n" << DIVIDER_MEDIUM << std::endl;
    }
    out << " Lattice spacetime coordinate (ID #" << id << ")"
        << "\n " << BULLET_PT << " Position indices: (";
    for (std::size_t i = 0; i < posn.size(); ++i) {
      if (i == posn.size() - 1) {
        out << posn[i] << ")" << std::endl;
      } else {
        out << posn[i] << ", ";
      }
    }
    out << " " << BULLET_PT << " Rescaled imaginary time (tau / beta): " << (itime / beta)
        << std::endl;
    if (botrule) {
      out << DIVIDER_MEDIUM << std::endl;
    }
  }
};
typedef std::vector<lat_st_coord> lat_st_coords;

// Multiband, product basis interaction refinement of a spacetime coordinate
// (carries two band indices and one product basis index)
class mb_pb_lat_st_vertex : public lat_st_coord {
 public:
  int pb_idx;   // Product basis index
  int band_in;  // Incoming/outgoing fermionic band indices
  int band_out;
  mb_pb_lat_st_vertex() = default;
  // mb_pb_lat_st_vertex(int band_in_, int band_out_, int pb_idx_, int dim_, int n_site_pd_,
  //                     double beta_, double delta_tau_, double lat_const_ = 1.0, bool debug_ =
  //                     false)
  //     : pb_idx(pb_idx_), band_in(band_in_), band_out(band_out_) {}
  mb_pb_lat_st_vertex(int pb_idx_, int band_in_, int band_out_, int id_, double itime_,
                      std::vector<int> posn_, int n_site_pd_, double beta_, double delta_tau_,
                      double lat_const_ = 1.0, bool debug_ = false)
      : lat_st_coord(id_, itime_, posn_, n_site_pd_, beta_, delta_tau_, lat_const_, debug_),
        pb_idx(pb_idx_),
        band_in(band_in_),
        band_out(band_out_) {}
};
typedef std::vector<mb_pb_lat_st_vertex> mb_pb_lat_st_vertices;

// Multiband refinement of a spacetime coordinate (carries two band indices)
class mb_lat_st_vertex : public lat_st_coord {
 public:
  int band_in;  // Incoming/outgoing fermionic band indices
  int band_out;
  mb_lat_st_vertex() = default;
  // mb_lat_st_vertex(int band_in_, int band_out_, int dim_, int n_site_pd_, double beta_,
  //                 double delta_tau_, double lat_const_ = 1.0, bool debug_ = false)
  //     : band_in(band_in_), band_out(band_out_) {}
  mb_lat_st_vertex(int band_in_, int band_out_, int id_, double itime_, std::vector<int> posn_,
                   int n_site_pd_, double beta_, double delta_tau_, double lat_const_ = 1.0,
                   bool debug_ = false)
      : lat_st_coord(id_, itime_, posn_, n_site_pd_, beta_, delta_tau_, lat_const_, debug_),
        band_in(band_in_),
        band_out(band_out_) {}
};
typedef std::vector<mb_lat_st_vertex> mb_lat_st_vertices;

// Calculate the lattice spacetime difference v1.posn - v2.posn (modulo lattice BC)
std::vector<int> lat_diff(lat_st_coord v1, lat_st_coord v2) {
  // Spatial distance in units of the lattice constant (nvec = rvec / a)
  std::vector<int> del_nr;
  // First, fill del_nr with (posn_2 - posn_1)
  std::transform(v2.posn.begin(), v2.posn.end(), v1.posn.begin(), std::back_inserter(del_nr),
                 std::minus<int>());
  // Now return the vector for (posn_final - posn_init), after
  // enforcing the appropriate lattice boundary conditions
  return first_orthant(del_nr, v1.n_site_pd);
}

// Calculate the spatial distance |r1 - r2| using the appropriate lattice metric,
// in units of the lattice constant (rescale by v1.lat_const for absolute distance)
double lat_dist(lat_st_coord v1, lat_st_coord v2) {
  // First, get the spacetime difference vector (v1 - v2) in the first orthant
  const std::vector<int> &del_nr = lat_diff(v1, v2);
  // Now, calculate the distance, sqrt(r12 * r12)
  double del_r_mag = 0;
  for (const int &del_nr_i : del_nr) {
    del_r_mag += std::pow(del_nr_i, 2.0);
  }
  return std::sqrt(del_r_mag);
}

// Checks whether two sites are nearest neighbors
bool nearest_neighbors(lat_st_coord v1, lat_st_coord v2) {
  // First, get the spacetime difference vector (v1 - v2) in the first orthant
  const std::vector<int> &del_nr = lat_diff(v1, v2);
  // If (del_nr * del_nr) is one, the two sites are nearest neighbors
  return are_close(std::inner_product(del_nr.begin(), del_nr.end(), del_nr.begin(), 0), 1.0);
}

// Overload for a call directly on a lattice space-time coordinate (returns a copy)
lat_st_coord first_orthant(const lat_st_coord &coord) {
  lat_st_coord coord_shifted = coord;
  for (std::size_t i = 0; i < coord.posn.size(); ++i) {
    coord_shifted.posn[i] =
        std::min(std::abs(coord.posn[i]), coord.n_site_pd - std::abs(coord.posn[i]));
  }
  return coord_shifted;
}

// Class representing a hypercubic lattice Matsubara 4-vector;
// momentum 3-vectors are given in units of the reciprocal
// lattice spacing, i.e., they index the 1BZ.
class lat_mf_coord {
 public:
  int id;
  double imfreq;  // The imaginary part of the Matsubara frequency (i imfreq)
  std::vector<int> mom;
  // Lattice variables
  int n_site_pd;
  double lat_const;
  lat_mf_coord(int id_, double imfreq_, std::vector<int> mom_, int n_site_pd_,
               double lat_const_ = 1.0)
      : id(id_), imfreq(imfreq_), mom(mom_), n_site_pd(n_site_pd_), lat_const(lat_const_) {}
  // Default constructor
  lat_mf_coord() : id(0), imfreq(0), lat_const(1.0) {}
  // Overload - operator for calculation of lattice Matsubara 4-vector
  // differences (v_end - v_start)
  std::tuple<std::vector<int>, double> operator-(const lat_mf_coord &v_start) const {
    // Spatial distance in units of the lattice constant (nvec = rvec / a)
    std::vector<int> del_nk;
    // First fill del_nk with (v_end - v_start), then apply the reciprocal
    // lattice metric component-wise
    std::transform(mom.begin(), mom.end(), v_start.mom.begin(), std::back_inserter(del_nk),
                   std::minus<int>());
    // Now we can return a tuple (r_end - r_start, ifreq_end - ifreq_start)
    return std::make_tuple(first_brillouin_zone(del_nk, n_site_pd), imfreq - v_start.imfreq);
  }
  void print(std::streambuf *buffer = std::cout.rdbuf(), bool toprule = true,
             bool botrule = true) const {
    std::ostream out(buffer);
    if (toprule) {
      out << "\n" << DIVIDER_MEDIUM << std::endl;
    }
    out << " Lattice momentum-time coordinate (ID #" << id << ")" << std::endl;
    out << " " << BULLET_PT << " Momentum indices: (";
    for (std::size_t i = 0; i < mom.size(); ++i) {
      if (i == mom.size() - 1) {
        out << mom[i] << ")" << std::endl;
      } else {
        out << mom[i] << ", ";
      }
    }
    out << " " << BULLET_PT << " Matsubara frequency Im(inu): " << imfreq << std::endl;
    if (botrule) {
      out << DIVIDER_MEDIUM << std::endl;
    }
  }
};
typedef std::vector<lat_mf_coord> lat_mf_coords;

std::tuple<std::vector<int>, double> test_lat_mf_diff(lat_mf_coord v1, lat_mf_coord v2) {
  return v1 - v2;
}

// Calculate the lattice momentum-frequency 'distance' dist(v1, v2) \equiv (k_12, ik_12)
std::tuple<double, double> lat_mf_dist(lat_mf_coord v1, lat_mf_coord v2) {
  // First, get the momentum-time difference (v1 - v2)
  std::vector<int> del_nk;
  double del_freq;
  std::tie(del_nk, del_freq) = (v1 - v2);
  // Now, apply the norm function to the momentum difference
  double del_k_mag = 0;
  for (const auto &del_nk_i : del_nk) {
    del_k_mag += std::pow(del_nk_i, 2.0);
  }
  double mom_scale = (2.0 * M_PI) / (v1.n_site_pd * v1.lat_const);
  del_k_mag = mom_scale * std::sqrt(del_k_mag);
  return std::make_tuple(del_k_mag, del_freq);
}

// Overload for a call directly on a lattice momentum-time coordinate
// (returns a copy)
lat_mf_coord first_brillouin_zone(const lat_mf_coord &coord) {
  lat_mf_coord coord_shifted = coord;
  // Move each component back into the 1BZ;
  // k_i \in [-\pi / a, \pi / a) => nk_i \in [floor(-N / 2), floor(N / 2) - 1)
  for (std::size_t i = 0; i < coord.mom.size(); ++i) {
    while (coord_shifted.mom[i] >= std::floor(coord.n_site_pd / 2.0)) {
      coord_shifted.mom[i] -= coord.n_site_pd;
    }
    while (coord_shifted.mom[i] < std::floor(-coord.n_site_pd / 2.0)) {
      coord_shifted.mom[i] += coord.n_site_pd;
    }
    // assert((coord_shifted.mom[i] >= std::floor(-coord.n_site_pd / 2.0)) &&
    //        (coord_shifted.mom[i] < std::floor(coord.n_site_pd / 2.0)));
  }
  return coord_shifted;
}

namespace develop {

#if BOOST_VERSION >= 106500
typedef boost::math::arcsine_distribution<double> Arcsin_dist;
Arcsin_dist arcsin_dist(0.0, 1.0);
// Arcsine distribution for imaginary time generation
double arcsin_gen(Rand_engine rand_gen) {
  return boost::math::quantile(arcsin_dist, std_uniform(rand_gen));
}
#endif

// Class representing a hypercubic lattice space - (imaginary) time coordinate;
// position vectors are given in units of the lattice constant, i.e., they index
// the lattice, and the d-toroidal lattice metric is used for spatial distances
class fcc_lat_st_coord {
 public:
  int id;
  double itime;
  std::vector<int> posn;
  // Lattice parameters
  int dim = 3;
  int n_site_pd;
  double lat_const;
  double two_pi_a = 2.0 * M_PI / lat_const;
  // Contains the list of normalized direct lattice basis vectors
  // (in units of the lattice constant)
  std::vector<std::vector<double>> r_basis = {
      {+0.5, +0.5, +0.0}, {+0.5, +0.0, +0.5}, {+0.0, +0.5, +0.5}};
  // Contains the list of reciprocal lattice basis vectors
  // (in units of (2 pi / a))
  std::vector<std::vector<double>> k_basis = {
      {+0.5, +0.5, -0.5}, {+0.5, -0.5, +0.5}, {-0.5, +0.5, +0.5}};
  // Custom default constructor (so that we can resize a vector of coordinates)
  fcc_lat_st_coord() : id(0), itime(0), n_site_pd(0), lat_const(0) {}
  // Standard constructor
  fcc_lat_st_coord(int id_, double itime_, std::vector<int> posn_, int n_site_pd_,
                   double lat_const_ = 1.0)
      : id(id_), itime(itime_), posn(posn_), n_site_pd(n_site_pd_), lat_const(lat_const_) {}
  // Overload - operator for calculation of lattice spacetime differences (v_end
  // - v_start)
  std::tuple<std::vector<int>, double> operator-(const fcc_lat_st_coord &v_start) const {
    // Spatial distance in units of the lattice constant (nvec = rvec / a)
    std::vector<int> del_nr;
    // First fill del_nr with (v_end - v_start),
    // then apply the lattice metric component-wise
    std::transform(posn.begin(), posn.end(), v_start.posn.begin(), std::back_inserter(del_nr),
                   std::minus<int>());
    // Now we can return a tuple (r_end - r_start, tau_end - tau_start)
    return std::make_tuple(lat_bc(del_nr), itime - v_start.itime);
  }
  // Apply the lattice boundary conditions to a distance vector;
  // they are the same as for a square d-torus
  std::vector<int> lat_bc(std::vector<int> &nr) const {
    std::vector<int> nr_lat_proj = nr;
    for (std::size_t i = 0; i < nr.size(); ++i) {
      nr_lat_proj[i] = std::min(std::abs(nr[i]), n_site_pd - std::abs(nr[i]));
    }
    return nr_lat_proj;
  }
};
typedef std::vector<fcc_lat_st_coord> fcc_lat_st_coords;

// TODO: refactor this to avoid unnecessary calculation
//       of time difference and std tie/ignore!
//
// Calculate the spatial distance |r1 - r2| using the appropriate
// (fcc) lattice metric with basis
double lat_dist(fcc_lat_st_coord v1, fcc_lat_st_coord v2) {
  // First, get the spacetime difference vector (v1 - v2) in the first orthant
  std::vector<int> del_nr;
  std::tie(del_nr, std::ignore) = (v1 - v2);
  // Now, calculate the distance, sqrt(r_12 * r_12), where
  // r_i = \sum^{d}_{i=1} n_i \vec{a_i}
  double del_r_mag = 0;
  // for (const auto &del_nr_i : del_nr) {
  for (int i = 0; i < v1.dim; ++i) {
    double del_ri = 0;
    for (int j = 0; j < v1.dim; ++j) {
      del_ri += del_nr[i] * v1.r_basis[i][j];
    }
    del_r_mag += std::pow(del_ri, 2.0);
  }
  del_r_mag = v1.lat_const * std::sqrt(del_r_mag);
  return del_r_mag;
}

}  // end namespace develop

namespace deprecated {

// Defines the permutation group representation of a graph, consisting
// of one fermionic (psi) and one bosonic (phi) connection row vector
typedef std::vector<std::vector<int>> graph_pg;
typedef std::vector<graph_pg> graphs_pg;

// Diagram pool class (set of graphs at fixed order, associated
// diagram info and current/proposal spacetime coordinates)
// using pg graph representation
struct diagram_pool_pg {
  double s_ferm;
  int order;
  int n_verts;
  int n_diags;
  int n_legs;
  int n_intn;
  int n_times;
  int n_posns;
  int n_spins_max;
  graphs_pg graphs;
  std::vector<int> symm_factors;
  std::vector<int> n_loops;
  std::vector<std::vector<std::vector<int>>> loops;
  std::vector<std::vector<std::vector<int>>> neighbors;
  // We must supply an explicit default constructor
  diagram_pool_pg()
      : s_ferm(),
        order(),
        n_legs(),
        n_intn(),
        n_times(),
        n_posns(),
        graphs(),
        symm_factors(),
        n_loops(),
        loops(),
        neighbors() {}
  // Diagram pool constructor for (spinless) G0W0 integrators
  diagram_pool_pg(double s_ferm_, int order_, int n_legs_, int n_intn_, int n_times_, int n_posns_,
                  const graphs_pg &graphs_, const std::vector<int> &symm_factors_,
                  const std::vector<int> &n_loops_,
                  const std::vector<std::vector<std::vector<int>>> &neighbors_)
      : s_ferm(s_ferm_),
        order(order_),
        n_legs(n_legs_),
        n_intn(n_intn_),
        n_times(n_times_),
        n_posns(n_posns_),
        graphs(graphs_),
        symm_factors(symm_factors_),
        n_loops(n_loops_),
        neighbors(neighbors_) {
    n_verts = 2 * order_;
    n_diags = graphs_.size();
  }
  // Diagram pool constructor for (Extended) Hubbard integrators
  diagram_pool_pg(double s_ferm_, int order_, int n_legs_, int n_intn_, int n_times_, int n_posns_,
                  const graphs_pg &graphs_, const std::vector<int> &symm_factors_,
                  const std::vector<int> &n_loops_,
                  const std::vector<std::vector<std::vector<int>>> &loops_,
                  const std::vector<std::vector<std::vector<int>>> &neighbors_)
      : s_ferm(s_ferm_),
        order(order_),
        n_legs(n_legs_),
        n_intn(n_intn_),
        n_times(n_times_),
        n_posns(n_posns_),
        graphs(graphs_),
        symm_factors(symm_factors_),
        n_loops(n_loops_),
        loops(loops_),
        neighbors(neighbors_) {
    n_verts = 2 * order_;
    n_diags = graphs_.size();
    n_spins_max = *max_element(std::begin(n_loops_), std::end(n_loops_));
  }
};
typedef std::vector<diagram_pool_pg> diagram_pools_pg;

}  // end namespace deprecated
