#pragma once
#include "diagmc_includes.hpp"

/* cmath constant definitions */
#define _USE_MATH_DEFINES

/* Typedefs */
typedef std::bitset<16> IntBits;
// Distribution / random number generator typedefs
typedef boost::random::mt19937 Rand_engine;
typedef boost::math::binomial Binom_dist;
typedef boost::random::binomial_distribution<int> Binom_gen;
typedef boost::random::uniform_int_distribution<int> DUnif_gen;
typedef boost::random::uniform_real_distribution<double> Unif_gen;
typedef boost::random::discrete_distribution<int, double> Discr_gen;

/* Global variables */
// Build the random generator and distributions to be used in the Metropolis
// step and configuration updates. We seed the random number
// generator with the standard 64-bit Mersenne twister
DUnif_gen coin_flip(0, 1);
DUnif_gen roll_1d3(0, 2);
DUnif_gen roll_1d4(0, 3);
Unif_gen std_uniform(0.0, 1.0);

// For the following distributions, the parameters are
// determined at runtime (construction) or are variable
DUnif_gen select_posn_component;
DUnif_gen select_modifiable;
Discr_gen posn_shift_gen;
Binom_gen lat_binomial;
Binom_dist lat_binomial_dist;

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
  // The two external legs are always unpinned,
  // except for static self energy diagrams
  if ((diag_type != 2) && (i_vert < n_legs)) {
    return false;
  }
  // For dynamic self energy diagrams, the two external
  // boson lines break the alternating convention, and
  // hence are hard-coded as special cases here
  else if ((diag_type == 3) && (i_vert == 2)) {
    return IntBits(i_constr).test(0);
  } else if ((diag_type == 3) && (i_vert == 3)) {
    return IntBits(i_constr).test(1);
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
    return IntBits(i_constr).test((i_vert - i_first_bos) / 2);
  }
}

bool string_contains(std::string string, std::string substring) {
  return (string.find(substring) != std::string::npos);
}

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

// Macros to define JSON (de)serialization methods to_json/from_json
// for each mcmc config group (all children/parent structs) concisely
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::diag_config, diag_type, subspaces, norm_space_weight,
                                   order, n_legs, n_intn, n_times, n_posns)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::mcmc_config, debug, verbose, normalize,
                                   save_serial, use_batch_U, n_warm, n_meas, n_skip, n_threads,
                                   n_nu_meas, n_k_meas, max_posn_shift, job_id, save_dir, save_name)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::phys_config, dim, n_site, n_site_pd,
                                   n_site_irred, num_elec, lat_const, lat_length, vol_lat,
                                   target_mu, target_n0, mu_tilde, mu, n0, rs, ef, beta, t_hop,
                                   s_ferm, U_loc, U_batch)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config::propr_config, delta_tau, n_nu, n_tau,
                                   job_id, save_dir)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(hub_2dsqlat_mcmc_config, diag, mcmc, phys, propr)

// Perform some simple checks on H5 data/pred types (to avoid unexpected casting)
template <typename Tstd, typename Th5 = H5::PredType>
void check_h5type(std::string name, H5::DataType datatype, Th5 predtype) {
  if (datatype.getClass() != predtype.getClass()) {
    throw std::runtime_error("Unable to read " + name + ", incorrect data-type");
  }
  if (datatype.getSize() != sizeof(Tstd)) {
    throw std::runtime_error("Unable to read " + name + ", incorrect precision");
  }
  return;
}

// Load an attribute from an HDF5 location and return data of corresponding predtype
// NOTE: may throw an exception, which should be caught!
template <typename Tattr, typename Tloc = H5::Group>
Tattr load_h5_attribute(const std::string &attr_name, const Tloc &h5loc) {
  Tattr attr_buffer;
  if (!(std::is_base_of<H5::Group, Tloc>::value || std::is_same<H5::DataSet, Tloc>::value)) {
    throw not_implemeted_error(
        "Invalid H5 object supplied; should be an H5 file, group, or dataset.");
  }
  if (std::is_same<Tattr, bool>::value) {
    H5::Attribute attr = h5loc.openAttribute(attr_name);
    H5::DataType attr_type = attr.getDataType();
    check_h5type<bool>(attr_name, attr_type, H5::PredType::NATIVE_HBOOL);
    attr.read(H5::PredType::NATIVE_HBOOL, &attr_buffer);
  } else if (std::is_same<Tattr, std::string>::value) {
    H5::Attribute attr = h5loc.openAttribute(attr_name);
    H5::StrType attr_type = attr.getStrType();
    H5::StrType h5str_type(0, H5T_VARIABLE);
    check_h5type<const char *, H5::StrType>(attr_name, attr_type, h5str_type);
    attr.read(attr_type, attr_buffer);
  } else if (std::is_integral<Tattr>::value) {
    H5::Attribute attr = h5loc.openAttribute(attr_name);
    H5::DataType attr_type = attr.getDataType();
    check_h5type<long>(attr_name, attr_type, H5::PredType::NATIVE_LONG);
    attr.read(attr_type, &attr_buffer);
  } else if (std::is_floating_point<Tattr>::value) {
    H5::Attribute attr = h5loc.openAttribute(attr_name);
    H5::DataType attr_type = attr.getDataType();
    check_h5type<double>(attr_name, attr_type, H5::PredType::NATIVE_DOUBLE);
    attr.read(attr_type, &attr_buffer);
  } else {
    throw not_implemeted_error("Attribute storage for H5 type of attr_buffer '" + attr_name +
                               "' not yet implemented.");
  }
  return attr_buffer;
}

// Check if a specified attribute in an H5File is equal to an expected value
template <typename T>
bool h5_attribute_equals(const T &expected_value, const std::string &attr_name,
                         const std::string &filename, const H5::H5File &h5file) {
  const T param_found = load_h5_attribute<T, H5::H5File>(attr_name, h5file);
  if (param_found != expected_value) {
    return false;
  }
  return true;
}

// Add a parameter to an HDF5 location as an attribute
// NOTE: may throw an exception, which should be caught!
template <typename Tattr, typename Tloc = H5::Group>
void add_attribute_h5(Tattr param, std::string param_name, Tloc h5loc) {
  if (!(std::is_base_of<H5::Group, Tloc>::value || std::is_same<H5::DataSet, Tloc>::value)) {
    throw not_implemeted_error(
        "Invalid H5 object supplied; should be an H5 file, group, or dataset.");
  }
  H5std_string attr_name(param_name);
  H5::DataSpace attr_space(H5S_SCALAR);
  if (std::is_same<Tattr, bool>::value) {
    H5::Attribute attr = h5loc.createAttribute(attr_name, H5::PredType::NATIVE_HBOOL, attr_space);
    attr.write(H5::PredType::NATIVE_HBOOL, &param);
  } else if (std::is_same<Tattr, std::string>::value) {
    H5::StrType h5str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute attr = h5loc.createAttribute(attr_name, h5str_type, attr_space);
    attr.write(h5str_type, &param);
  } else if (std::is_integral<Tattr>::value) {
    H5::Attribute attr = h5loc.createAttribute(attr_name, H5::PredType::NATIVE_INT, attr_space);
    attr.write(H5::PredType::NATIVE_INT, &param);
  } else if (std::is_floating_point<Tattr>::value) {
    H5::Attribute attr = h5loc.createAttribute(attr_name, H5::PredType::NATIVE_DOUBLE, attr_space);
    attr.write(H5::PredType::NATIVE_DOUBLE, &param);
  } else {
    throw not_implemeted_error("Attribute storage for H5 type of param '" + param_name +
                               "' not yet implemented.");
  }
  return;
}

// Represent a 2D lattice of arbitrary objects using
// a variable-size contiguous (1D) std::vector
template <typename T>
class lattice_2d {
 public:
  // int N_i;
  int N_j;
  std::vector<T> data;
  // Constructors
  lattice_2d() = default;
  lattice_2d(int N_i_, int N_j_, const std::vector<T> &data_)
      : /* N_i(N_i_), */ N_j(N_j_), data(data_) {}
  // Index the 2D lattice
  constexpr const T &operator()(int i, int j) const { return data[j + N_j * i]; }
};

// Represent a 3D lattice of arbitrary objects using
// a variable-size contiguous (1D) std::vector
template <typename T>
class lattice_3d {
 public:
  // int N_i;
  int N_j;
  int N_k;
  std::vector<T> data;
  // Constructors
  lattice_3d() = default;
  lattice_3d(int N_i_, int N_j_, int N_k_, const std::vector<T> &data_)
      : /* N_i(N_i_), */ N_j(N_j_), N_k(N_k_), data(data_) {}
  // Index the 3D lattice
  constexpr const T &operator()(int i, int j, int k) const { return data[k + N_k * (j + N_j * i)]; }
};

// 2D mesh class (contains the function data on the mesh, and the grids for each variable)
struct fmesh_2d {
  // Fields
  std::vector<std::vector<double>> data;
  std::vector<double> x_grid;
  std::vector<double> y_grid;
  // Constructor
  fmesh_2d(const std::vector<std::vector<double>> &data_, const std::vector<double> &x_grid_,
           const std::vector<double> &y_grid_)
      : data(data_), x_grid(x_grid_), y_grid(y_grid_) {}
};

// 2D interpolant class (used, e.g., to define spatially continuous objects from r-grid data)
class interp_2d {
 public:
  // Fields
  fmesh_2d f_mesh;
  // Constructor
  interp_2d(const fmesh_2d &f_mesh_) : f_mesh(f_mesh_) {}
  // Use bilinear interpolation to evaluate the interpoland at any point
  double eval(const std::vector<double> &point) const { return bilinear_interp(f_mesh, point); }
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
    return sign * bilinear_interp(f_mesh, point_shifted);
  }

 private:
  // Bilinear interpolation function
  double bilinear_interp(const fmesh_2d &f_mesh, const std::vector<double> &point) const {
    // Define the x and y values at which to evaluate the function
    double x = point[0], y = point[1];
    // If the values are outside the range of the x and y grids,
    // return 0 (i.e., use extrapolation with a fill value of zero)
    if ((x < f_mesh.x_grid.front()) || (x > f_mesh.x_grid.back())) {
      return 0;
    }
    if ((y < f_mesh.y_grid.front()) || (y > f_mesh.y_grid.back())) {
      return 0;
    }
    // Identify the nearest mesh neighbors of the evaluation point (x, y)
    std::vector<double>::const_iterator iter_x2;
    std::vector<double>::const_iterator iter_y2;
    iter_x2 = std::lower_bound(f_mesh.x_grid.begin(), f_mesh.x_grid.end(), x);
    iter_y2 = std::lower_bound(f_mesh.y_grid.begin(), f_mesh.y_grid.end(), y);
    // Special case: if the evaluation point contains f_mesh.x_grid[-1]
    //  or f_mesh.y_grid[-1], shift the bounding square left by 1 manually
    // (std::next not applicable if lower bound gives last point in grid)
    if (iter_x2 == f_mesh.x_grid.begin()) {
      iter_x2 = std::next(iter_x2);
    }
    if (iter_y2 == f_mesh.y_grid.begin()) {
      iter_y2 = std::next(iter_y2);
    }
    // Now, we can reliably get the last two corners
    // of the bounding box, including edge cases
    // std::vector<double>::const_iterator iter_x2 = std::next(iter_x1) -
    // f_mesh.x_grid.begin(); std::vector<doublidx_x2e>::const_iterator iter_y2
    // = std::next(iter_y1) - f_mesh.y_grid.begin();
    auto iter_x1 = std::prev(iter_x2);
    auto iter_y1 = std::prev(iter_y2);
    // Get the actual indices associated with each iterator
    int idx_x1 = iter_x1 - f_mesh.x_grid.begin();
    int idx_x2 = iter_x2 - f_mesh.x_grid.begin();
    int idx_y1 = iter_y1 - f_mesh.y_grid.begin();
    int idx_y2 = iter_y2 - f_mesh.y_grid.begin();
    // Precompute the grid points x_1, x_2, y_1, y_2
    double x_1 = f_mesh.x_grid[idx_x1];
    double x_2 = f_mesh.x_grid[idx_x2];
    double y_1 = f_mesh.y_grid[idx_y1];
    double y_2 = f_mesh.y_grid[idx_y2];
    // Precompute the function values at the grid points f_11, f_12, f_21, f_22
    double f_11 = f_mesh.data[idx_x1][idx_y1];
    double f_12 = f_mesh.data[idx_x1][idx_y2];
    double f_21 = f_mesh.data[idx_x2][idx_y1];
    double f_22 = f_mesh.data[idx_x2][idx_y2];
    // Use bilinear interpolation to extrapolate
    // the function value at the evaluation point
    return (f_11 * (x_2 - x) * (y_2 - y) + f_12 * (x_2 - x) * (y - y_1) +
            f_21 * (x - x_1) * (y_2 - y) + f_22 * (x - x_1) * (y - y_1)) /
           ((x_2 - x_1) * (y_2 - y_1));
  }
};

// 1D mesh class (contains the 1D grid and function data)
struct fmesh_1d {
  // Fields
  std::vector<double> data;
  std::vector<double> x_grid;
  // Constructor
  fmesh_1d(const std::vector<double> &data_, const std::vector<double> &x_grid_)
      : data(data_), x_grid(x_grid_) {}
};

// 1D interpolant class (used, e.g. to define
// continuous-time objects from tau grid data)
class interp_1d {
 public:
  // Fields
  fmesh_1d f_mesh;
  // Constructor
  interp_1d(const fmesh_1d &f_mesh_) : f_mesh(f_mesh_) {}
  // An explicit default constructor
  interp_1d() : f_mesh({}, {}) {}
  // Use bilinear interpolation to evaluate the interpoland at any point
  double eval(double point) const { return linear_interp(f_mesh, point); }
  // Linear interpolation function; approximates f(x) from mesh data.
  // NOTE: This algorithm applies regardless of mesh uniformity!
  double linear_interp(const fmesh_1d &f_mesh, double x) const {
    // If the values are outside the range of the x and y grids,
    // return 0 (i.e., use extrapolation with a fill value of zero)
    if ((x < f_mesh.x_grid.front()) || (x > f_mesh.x_grid.back())) {
      return 0;
    }
    // Identify the nearest mesh neighbors of the evaluation point (x, y)
    std::vector<double>::const_iterator iter_x2;
    iter_x2 = std::lower_bound(f_mesh.x_grid.begin(), f_mesh.x_grid.end(), x);
    // Special case: if the evaluation point contains f_mesh.x_grid[-1],
    // shift the bounding points left by 1 manually (std::next not
    // applicable if lower bound gives last point in grid)
    if (iter_x2 == f_mesh.x_grid.begin()) {
      iter_x2 = std::next(iter_x2);
    }
    // Now, we can reliably get the last two corners
    // of the bounding box, including edge cases
    auto iter_x1 = std::prev(iter_x2);
    // Get the actual indices associated with each iterator
    int idx_x1 = iter_x1 - f_mesh.x_grid.begin();
    int idx_x2 = iter_x2 - f_mesh.x_grid.begin();
    // Precompute the grid points x_1 and x_2
    double x_1 = f_mesh.x_grid[idx_x1];
    double x_2 = f_mesh.x_grid[idx_x2];
    // Precompute the function values at the grid points f_1 and f_2
    double f_1 = f_mesh.data[idx_x1];
    double f_2 = f_mesh.data[idx_x2];
    // Use bilinear interpolation to extrapolate
    // the function value at the evaluation point
    return (f_1 * (x_2 - x) + f_2 * (x - x_1)) / (x_2 - x_1);
  }
};

// For a bosonic Green's function interpolant (periodic), the period is beta
class b_interp_1d : public interp_1d {
 public:
  // Fields
  double beta;
  // Constructor
  b_interp_1d(const fmesh_1d &f_mesh_, double beta_) : interp_1d(f_mesh_), beta(beta_) {}
  // Evaluates the periodic extension of the bosonic Green's function
  // object using information on the principle interval [0, beta).
  double p_eval(double point) const { return eval(pymod<double>(point, beta)); }
};

// For a fermionic Green's function interpolant (antiperiodic), the period is beta
class f_interp_1d : public interp_1d {
 public:
  // Fields
  double beta;
  // Constructor
  f_interp_1d(const fmesh_1d &f_mesh_, double beta_) : interp_1d(f_mesh_), beta(beta_) {}
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

// Continuous-time bosonic lattice Green's function types
typedef lattice_2d<b_interp_1d> lattice_2d_b_interp;
typedef lattice_3d<b_interp_1d> lattice_3d_b_interp;

// Continuous-time fermionic lattice Green's function types
typedef lattice_2d<f_interp_1d> lattice_2d_f_interp;
typedef lattice_3d<f_interp_1d> lattice_3d_f_interp;

// Defines the (split bosonic/fermionic) edge list representation of a set of graphs; since
// (wlog) the bosonic edges are assumed equal for all graphs, we only define the list once
typedef std::array<int, 2> edge_t;
typedef std::vector<edge_t> edge_list;
typedef std::vector<edge_list> edge_lists;

// A pool of graphs in the edge list representation
// (assumed to share a common bosonic edge basis)
struct graphs_el {
  // Fields
  edge_list b_edge_list;
  edge_lists f_edge_lists;
  // Constructors
  graphs_el() = default;
  graphs_el(const edge_list &b_edge_list_, const edge_lists &f_edge_lists_)
      : b_edge_list(b_edge_list_), f_edge_lists(f_edge_lists_) {}
};

// A pool of 3-point vertices each consisting of one boson
// and two fermion edges (directed in/out of the base vertex)
struct vertices_3pt_el {
  // Fields
  edge_list b_edge_list;
  edge_lists f_edge_in_lists;
  edge_lists f_edge_out_lists;
  // Constructors
  vertices_3pt_el() : b_edge_list({}), f_edge_in_lists({}), f_edge_out_lists({}) {}
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
  // Fields
  double s_ferm;
  int order;
  int n_verts;
  int n_diags;
  int n_legs;
  int n_intn;
  int n_times;
  int n_posns;
  int n_spins_max;
  graphs_el graphs;
  vertices_3pt_el nn_vertices;
  std::vector<int> symm_factors;
  std::vector<int> n_loops;
  std::vector<std::vector<std::vector<int>>> loops;
  // Constructors
  diagram_pool_el();
  // Constructor for the case of spinless (Hubbard, G0W0) integrators
  diagram_pool_el(const hub_2dsqlat_mcmc_config &config_, const graphs_el &graphs_,
                  const std::vector<int> &symm_factors_, const std::vector<int> &n_loops_,
                  const vertices_3pt_el &nn_vertices_ = vertices_3pt_el())
      : s_ferm(config_.phys.s_ferm),
        order(config_.diag.order),
        n_legs(config_.diag.n_legs),
        n_intn(config_.diag.n_intn),
        n_times(config_.diag.n_times),
        n_posns(config_.diag.n_posns),
        graphs(graphs_),
        symm_factors(symm_factors_),
        n_loops(n_loops_),
        nn_vertices(nn_vertices_),
        n_verts(2 * config_.diag.order),
        n_diags(graphs_.f_edge_lists.size()) {}
};
typedef std::vector<diagram_pool_el> diagram_pools_el;

// An explicit default constructor (a 0-dimensional diagram pool)
diagram_pool_el::diagram_pool_el()
    : s_ferm(0.5),
      order(0),
      n_legs(0),
      n_intn(0),
      n_times(0),
      n_posns(0),
      graphs(),
      nn_vertices(),
      n_verts(0),
      n_diags(1),
      n_spins_max(0),
      symm_factors({1}),
      n_loops({0}),
      loops({}) {}

// Class representing a space - (imaginary) time coordinate
class st_coord {
 public:
  // Fields
  int id;
  double itime;
  std::vector<double> posn;
  // Constructor
  st_coord(int id_, double itime_, std::vector<double> posn_)
      : id(id_), itime(itime_), posn(posn_) {}
  // Overload - operator for calculation of spacetime 'distance' (positional
  // distance, temporal difference)
  std::vector<double> operator-(const st_coord &start) const {
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
typedef std::vector<st_coord> st_coords;

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
class hc_lat_st_coord {
 public:
  // Fields
  bool debug;
  int id;
  double itime;
  std::vector<int> posn;
  // Lattice variables
  int dim;
  int n_site_pd;
  int constr_posn_shift = -2;
  double beta;
  double delta_tau;
  double lat_const;
  // Use a custom default constructor
  hc_lat_st_coord();
  // COM constructor declaration (coordinates at the origin and user-supplied
  // lattice parameters)
  hc_lat_st_coord(int dim_, int n_site_pd_, double beta_, double delta_tau_,
                  double lat_const_ = 1.0, bool debug_ = false);
  // Constructor
  hc_lat_st_coord(int id_, double itime_, std::vector<int> posn_, int n_site_pd_, double beta_,
                  double delta_tau_, double lat_const_ = 1.0, bool debug_ = false)
      : id(id_),
        itime(itime_),
        posn(posn_),
        n_site_pd(n_site_pd_),
        beta(beta_),
        delta_tau(delta_tau_),
        lat_const(lat_const_),
        debug(debug_),
        dim(posn_.size()) {}
  // Overload - operator for calculation of lattice spacetime differences (v_end - v_start)
  std::tuple<std::vector<int>, double> operator-(const hc_lat_st_coord &v_start) const {
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
      out << "\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          << std::endl;
    }
    out << " Lattice spacetime coordinate (ID #" << id << ")" << std::endl;
    if (constr_posn_shift != -2) {
      out << " \u2022 Constrained position shift (Delta): " << constr_posn_shift << std::endl;
    }
    out << " \u2022 Position indices: (";
    for (int i = 0; i < posn.size(); ++i) {
      if (i == posn.size() - 1) {
        out << posn[i] << ")" << std::endl;
      } else {
        out << posn[i] << ", ";
      }
    }
    out << " \u2022 Rescaled imaginary time (tau / beta): " << (itime / beta) << std::endl;
    if (botrule) {
      out << "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          << std::endl;
    }
  }
};
typedef std::vector<hc_lat_st_coord> hc_lat_st_coords;

// Custom default constructor definition (so that we can resize a vector of coordinates)
hc_lat_st_coord::hc_lat_st_coord() : id(0), itime(0), lat_const(0.0) {}

// COM constructor definition (coordinates at the origin and user-supplied
// lattice parameters)
hc_lat_st_coord::hc_lat_st_coord(int dim_, int n_site_pd_, double beta_, double delta_tau_,
                                 double lat_const_ /* = 1.0 */, bool debug_ /* = false */)
    : id(0),
      itime(0),
      posn(std::vector<int>(dim_, 0)),
      n_site_pd(n_site_pd_),
      beta(beta_),
      delta_tau(delta_tau_),
      lat_const(lat_const_),
      debug(debug_) {}

std::tuple<std::vector<int>, double> test_lat_st_diff(hc_lat_st_coord v1, hc_lat_st_coord v2) {
  return v1 - v2;
}

// Calculate the lattice spacetime difference v1.posn - v2.posn (modulo lattice BC)
std::vector<int> lat_diff(hc_lat_st_coord v1, hc_lat_st_coord v2) {
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
double lat_dist(hc_lat_st_coord v1, hc_lat_st_coord v2) {
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
bool nearest_neighbors(hc_lat_st_coord v1, hc_lat_st_coord v2) {
  // First, get the spacetime difference vector (v1 - v2) in the first orthant
  const std::vector<int> &del_nr = lat_diff(v1, v2);
  // If (del_nr * del_nr) is one, the two sites are nearest neighbors
  return are_close(std::inner_product(del_nr.begin(), del_nr.end(), del_nr.begin(), 0), 1.0);
}

// Overload for a call directly on a lattice space-time coordinate (returns a copy)
hc_lat_st_coord first_orthant(const hc_lat_st_coord &coord) {
  hc_lat_st_coord coord_shifted = coord;
  for (std::size_t i = 0; i < coord.posn.size(); ++i) {
    coord_shifted.posn[i] =
        std::min(std::abs(coord.posn[i]), coord.n_site_pd - std::abs(coord.posn[i]));
  }
  return coord_shifted;
}

// Class representing a hypercubic lattice Matsubara 4-vector;
// momentum 3-vectors are given in units of the reciprocal
// lattice spacing, i.e., they index the 1BZ.
class hc_lat_mf_coord {
 public:
  // Fields
  int id;
  double imfreq;  // The imaginary part of the Matsubara frequency (i imfreq)
  std::vector<int> mom;
  // Lattice variables
  int n_site_pd;
  double lat_const;
  // Constructor
  hc_lat_mf_coord(int id_, double imfreq_, std::vector<int> mom_, int n_site_pd_,
                  double lat_const_ = 1.0)
      : id(id_), imfreq(imfreq_), mom(mom_), n_site_pd(n_site_pd_), lat_const(lat_const_) {}
  // Default constructor
  hc_lat_mf_coord() : id(0), imfreq(0), lat_const(1.0) {}
  // Overload - operator for calculation of lattice Matsubara 4-vector
  // differences (v_end - v_start)
  std::tuple<std::vector<int>, double> operator-(const hc_lat_mf_coord &v_start) const {
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
      out << "\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          << std::endl;
    }
    out << " Lattice momentum-time coordinate (ID #" << id << ")" << std::endl;
    out << " \u2022 Momentum indices: (";
    for (int i = 0; i < mom.size(); ++i) {
      if (i == mom.size() - 1) {
        out << mom[i] << ")" << std::endl;
      } else {
        out << mom[i] << ", ";
      }
    }
    out << " \u2022 Matsubara frequency Im(inu): " << imfreq << std::endl;
    if (botrule) {
      out << "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
             "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
          << std::endl;
    }
  }
};
typedef std::vector<hc_lat_mf_coord> hc_lat_mf_coords;

std::tuple<std::vector<int>, double> test_lat_mf_diff(hc_lat_mf_coord v1, hc_lat_mf_coord v2) {
  return v1 - v2;
}

// Calculate the lattice momentum-frequency 'distance' dist(v1, v2) \equiv (k_12, ik_12)
std::tuple<double, double> lat_mf_dist(hc_lat_mf_coord v1, hc_lat_mf_coord v2) {
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
hc_lat_mf_coord first_brillouin_zone(const hc_lat_mf_coord &coord) {
  hc_lat_mf_coord coord_shifted = coord;
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
  // Fields
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
  // Use a custom default constructor
  fcc_lat_st_coord();
  // Constructor
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

// Custom default constructor (so that we can resize a vector of coordinates)
fcc_lat_st_coord::fcc_lat_st_coord() : id(0), itime(0), n_site_pd(0), lat_const(0) {}

std::tuple<std::vector<int>, double> test_lat_st_diff(fcc_lat_st_coord v1, fcc_lat_st_coord v2) {
  return v1 - v2;
}

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
  for (std::size_t i = 0; i < v1.dim; ++i) {
    double del_ri = 0;
    for (std::size_t j = 0; j < v1.dim; ++j) {
      del_ri += del_nr[i] * v1.r_basis[i][j];
    }
    del_r_mag += std::pow(del_ri, 2.0);
  }
  del_r_mag = v1.lat_const * std::sqrt(del_r_mag);
  return del_r_mag;
}

}  // namespace develop

namespace deprecated {

// Defines the permutation group representation of a graph, consisting
// of one fermionic (psi) and one bosonic (phi) connection row vector
typedef std::vector<std::vector<int>> graph_pg;
typedef std::vector<graph_pg> graphs_pg;

// Diagram pool class (set of graphs at fixed order, associated
// diagram info and current/proposal spacetime coordinates)
// using pg graph representation
struct diagram_pool_pg {
  // Fields
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
  diagram_pool_pg();
  // Constructor for the case of (spinless) G0W0 integrators
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
  // Constructor for the case of (Extended) Hubbard integrators
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
// We must supply an explicit default constructor
diagram_pool_pg::diagram_pool_pg()
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

}  // namespace deprecated
