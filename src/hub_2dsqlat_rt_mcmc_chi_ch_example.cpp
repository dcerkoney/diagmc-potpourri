#include "diagmc_hubbard_2dsqlat.hpp"

#ifdef _MPI
#include <mpi.h>
#endif

// Defines the measurement type: a Matsubara (4-momentum) correlation function
//                               for the Hubbard model on a 2D square lattice
using meas_t = mcmc_cfg_2d_sq_hub_mf_meas::meas_t;

// Simplify namespace for convenience
using json = nlohmann::json;

// Hard-coded parameters for a test calculation
namespace test_input {

/* MCMC parameters */
bool debug = false;       // Debug-level verbosity
bool verbose = true;      // Summarize after every half-millionth measurement
bool normalize = true;    // Should the integration results be normalized?
bool save_serial = true;  // Should the thread results be individually saved?
bool batch_U = false;     // Should we perform a batch calculation for multiple U values?
int n_warm = 100000;      // Number of steps in the burn-in / warmup phase
int n_meas = 5000000;     // Total number of steps in the measurement phase (including skips)
int n_skip = 1;           // Number of steps to skip in ancy points to measure at
int n_nu_meas = 1;        // Number of external frequency points to measure at
int max_posn_shift = 3;   // Variable maximum step size in local position component shifts

/* Lattice parameters */
int dim = 2;         // Spatial dimensionality of the problem
int n_site_pd = 30;  // Number of lattice sites along a single direction (i.e., L / a)
int n_site = static_cast<int>(std::pow(n_site_pd, dim));  // Total number of lattice sites
// Cutoff for (BC) 'irreducible' lattice distances
int n_site_irred = static_cast<int>(std::floor(n_site_pd / 2) + 1);
int n_tau = (1 << 10);  // (= 2^10) Number of points in the imaginary-time mesh [0, beta)
double ef = 0.0;           // Fermi energy
double beta = 10.0;        // Inverse temperature
double t_hop = 1.0;        // Nearest-neighbor hopping parameter t
double U_loc = 1.0;        // Hubbard U
double s_ferm = 0.5;       // Fermion spin
double lat_const = 1.0;    // Hypercubic lattice constant a
double delta_tau = 1e-10;  // Numerical imaginary time infinitesimal (defined by tau mesh)

// List of U values for (optional) batch-mode calculation
std::vector<double> U_batch = {0.1, 0.5, 1.0, 2.0, 4.0};

/* Order-dependent (fluctuating) lattice parameters */
double n0 = 1.0;  // Initial lattice electron density n0, set to half-filling
double lat_length = lat_const * n_site_pd;   // Lattice length in Bohr radii
double vol_lat = std::pow(lat_length, dim);  // Volume of the lattice
double rs = rad_d_ball(1.0 / n0, dim);       // Wigner-Seitz radius (for HEG correspondence)
int num_elec = static_cast<int>(std::round(n_site * n0));  // Number of electrons in the lattice

/* Diagram parameters */
int order = 3;             // Order n in perturbation theory, measurement space V_n
int n_legs = 2;            // Number of external legs in the V_n measurement
int n_intn = order - 1;    // Number of interaction lines in all V_n graphs (order in U)
int n_times = n_intn + 1;  // One modifiable time per internal line (static U), plus outgoing leg
int n_posns = n_intn + 1;  // One modifiable position per internal line (local U), plus outgoing leg
// Normalization diagram weight D_0 from quadratic fit vs. U at
// beta = 10 and beta at U = 1 to produce ~80-90% sample time in V_n
double D_0 = (-0.000143769 + 0.0143555 * U_loc + 0.0571346 * U_loc * U_loc) *
             (-0.0630317 + 0.0794737 * beta + 0.00206871 * beta * beta);

std::string save_name = "chi_ch_hub_2dsqlat";  // Prefix for saved files
std::string diag_type = "charge_poln";         // The class of measurement diagram(s)
std::vector<int> subspaces = {0, order};       // V = V_0 \oplus V_n

/* Derived parameters loaded from Green's function data */
double mu_tilde;  // The HF renormalized, aka reduced chemical potential,
                  // is mu_tilde \approx mu - (U n / 2) (exact in the
                  // thermodynamic limit); we implement the HF resummation
                  // by working as a function of (fixed) reduced chemical
                  // potential, i.e., we define G_HF(mu) := G_0(mu_tilde)
double mu;        // Physical chemical potential corresponding to mu_tilde

}  // namespace test_input

// Check if the params used for G_0 data match the current test parameters
bool params_consistent(std::string filename, bool main_thread) {
  // Open H5 file
  H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);
  // Check global G_0 attributes for consistency with test params
  const long dim = load_attribute_h5<long, H5::H5File>("dim", file);
  if (!(dim == test_input::dim)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename << "' has inconsistent dim params: ("
                << dim << ", " << test_input::dim << ")" << std::endl;
    }
    return false;
  }
  const long n_tau = load_attribute_h5<long, H5::H5File>("n_tau", file);
  if (!(n_tau == test_input::n_tau)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename << "' has inconsistent n_tau params: ("
                << n_tau << ", " << test_input::n_tau << ")" << std::endl;
    }
    return false;
  }
  const long n_site_pd = load_attribute_h5<long, H5::H5File>("n_site_pd", file);
  if (!(n_site_pd == test_input::n_site_pd)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename
                << "' has inconsistent n_site_pd params: (" << n_site_pd << ", "
                << test_input::n_site_pd << ")" << std::endl;
    }
    return false;
  }
  const double lat_const = load_attribute_h5<double, H5::H5File>("lat_const", file);
  if (!(lat_const == test_input::lat_const)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename
                << "' has inconsistent lat_const params: (" << lat_const << ", "
                << test_input::lat_const << ")" << std::endl;
    }
    return false;
  }
  const double t_hop = load_attribute_h5<double, H5::H5File>("t_hop", file);
  if (!(t_hop == test_input::t_hop)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename << "' has inconsistent t_hop params: ("
                << t_hop << ", " << test_input::t_hop << ")" << std::endl;
    }
    return false;
  }
  const double U_loc = load_attribute_h5<double, H5::H5File>("U_loc", file);
  if (!(U_loc == test_input::U_loc)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename << "' has inconsistent U_loc params: ("
                << U_loc << ", " << test_input::U_loc << ")" << std::endl;
    }
    return false;
  }
  const double beta = load_attribute_h5<double, H5::H5File>("beta", file);
  if (!(beta == test_input::beta)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename << "' has inconsistent beta params: ("
                << beta << ", " << test_input::beta << ")" << std::endl;
    }
    return false;
  }
  const double n0 = load_attribute_h5<double, H5::H5File>("n0", file);
  if (!(n0 == test_input::n0)) {
    if (test_input::debug && main_thread) {
      std::cout << "Green's function data at '" << filename << "' has inconsistent n0 params: ("
                << n0 << ", " << test_input::n0 << ")" << std::endl;
    }
    return false;
  }
  return true;
}

// Loads the bare (Hartree) lattice Green's function data from HDF5
lattice_2d_f_interp load_g0_h5(std::string filename, bool debug = false) {
  // Use test input parameters
  using namespace test_input;

  int ibuffer;
  double dbuffer;

  // Open H5 file
  H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);

  // Load chemical potential attributes into test parameters
  mu = load_attribute_h5<double, H5::H5File>("mu", file);
  mu_tilde = load_attribute_h5<double, H5::H5File>("mu_tilde", file);

  ////////////////////////
  // Load tau mesh data //
  ////////////////////////

  H5::DataSet dataset = file.openDataSet("tau_mesh");
  H5::DataSpace dataspace = dataset.getSpace();
  H5::DataType datatype = dataset.getDataType();
  check_h5type<double>("tau_mesh", datatype, H5::PredType::NATIVE_DOUBLE);

  // Tau mesh is assumed rank 1 (1D)
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims, NULL);
  size_t size = static_cast<size_t>(dims[0]);

  // Read the (flattened) lattice G_0 data
  std::vector<double> tau_mesh(size);
  dataset.read(tau_mesh.data(), datatype);
  const double delta_tau = tau_mesh[1];

  if (debug) {
    for (auto& tau : tau_mesh) {
      std::cout.precision(16);
      std::cout << tau << std::endl;
    }
    std::cout << delta_tau << std::endl;
  }

  ///////////////////////////
  // Load lattice G_0 data //
  ///////////////////////////

  dataset = file.openDataSet("lat_g0_rt_data");
  dataspace = dataset.getSpace();
  datatype = dataset.getDataType();
  check_h5type<double>("lat_g0_rt_data", datatype, H5::PredType::NATIVE_DOUBLE);

  // Get rank, dimensions and size of the data
  int g0_rank = dataspace.getSimpleExtentNdims();
  hsize_t dims_g0[g0_rank];
  dataspace.getSimpleExtentDims(dims_g0, NULL);
  // Deduce the total size of the data array
  size = 0;
  for (int i = 0; i < g0_rank; ++i) {
    size += static_cast<size_t>(dims_g0[i]);
  }

  // Read the (flattened) lattice G_0 data
  std::vector<double> lat_g0_vec(size);
  dataset.read(lat_g0_vec.data(), datatype);

  if (debug) {
    for (auto& i : lat_g0_vec) {
      std::cout << i << std::endl;
    }
  }

  H5::Attribute shape_attr = dataset.openAttribute("shape");
  H5::DataType shape_type = shape_attr.getDataType();
  dataspace = shape_attr.getSpace();
  check_h5type<long>("shape", shape_type, H5::PredType::NATIVE_LONG);

  // Shape array is rank 1 (1D)
  dataspace.getSimpleExtentDims(dims, NULL);
  size = static_cast<size_t>(dims[0]);
  assert(size == 3);

  // Read lattice G_0 data attribute 'shape'
  std::vector<long> shape(size);
  shape_attr.read(shape_type, shape.data());

  // G_0(r, tau) mesh is (d + 1)-dimensional
  int dim = shape.size() - 1;
  int n_site_irred = static_cast<int>(std::floor(n_site_pd / 2) + 1);
  assert(shape == std::vector<long>({n_site_irred, n_site_irred, n_tau}));

  if (debug) {
    std::cout << "\nG_0 data shape = ( ";
    for (auto i : shape) {
      std::cout << i << " ";
    }
    std::cout << ")\n" << std::endl;
  }

  //////////////////////////////////
  // Build G_0 interpolant matrix //
  //////////////////////////////////

  // Build a vector of linear 1D interpolants from flattened lattice G_0 data
  std::vector<f_interp_1d> lat_g0_vec_interp;
  for (int i = 0; i < n_site_irred; ++i) {
    for (int j = 0; j < n_site_irred; ++j) {
      std::vector<double> g0_tau_data;
      for (int k = 0; k < n_tau; ++k) {
        g0_tau_data.push_back(lat_g0_vec[k + n_tau * (j + n_site_irred * i)]);
      }
      lat_g0_vec_interp.push_back(f_interp_1d(fmesh_1d(g0_tau_data, tau_mesh), beta));
    }
  }
  lattice_2d_f_interp lat_g0_r_tau(n_site_irred, n_site_irred, lat_g0_vec_interp);
  return lat_g0_r_tau;
}

#ifdef _MPI
// Aggregate MPI results, compute the standard error over threads, and save the results to HDF5
void aggregate_and_save(int mpi_size, int mpi_rank, int mpi_main,
                        mcmc_cfg_2d_sq_hub_mf_meas integrator, std::string job_id = "",
                        std::string filename = "") {
  // Use test input parameters
  using namespace test_input;

  bool main_thread = (mpi_rank == mpi_main);

  const mcmc_lat_ext_hub_params& params = integrator.params;
  bool normalized = integrator.normalized;
  meas_t thread_data;
  if (normalized) {
    thread_data = integrator.meas_means;
  } else {
    thread_data = integrator.meas_sums;
  }

  // Default save name
  if (filename.empty()) {
    filename = "mcmc_run";
  }

  // To avoid recording the full covariance matrix, we assume a complex "error circle",
  // i.e., we define the error using the norm squared means; this can heavily
  // overestimate the error of the smaller component if, e.g., the real/imag
  // scales differ greatly.
  typedef std::vector<std::vector<double>> err_t;
  err_t thread_normsqdata;
  for (size_t i = 0; i < subspaces.size(); i++) {
    const std::vector<std::complex<double>>& ss_data = thread_data[i];
    std::vector<double> ss_normsqdata;
    for (size_t j = 0; j < ss_data.size(); j++) {
      // std defines norm as the field norm (Euclidean norm squared), |z|^2
      ss_normsqdata.push_back(std::norm(ss_data[j]));
    }
    thread_normsqdata.push_back(ss_normsqdata);
  }
  assert(thread_normsqdata.size() == thread_data.size());
  assert(thread_normsqdata[1].size() == thread_data[1].size());

  // Get the (Bessel-corrected) standard error over threads for each subspace measurement
  meas_t meas_data;
  err_t meas_err;
  for (size_t i = 0; i < subspaces.size(); i++) {
    // Sum over threads
    if (main_thread) {
      MPI_Reduce(MPI_IN_PLACE, thread_data[i].data(), thread_data[i].size(), MPI_DOUBLE_COMPLEX,
                 MPI_SUM, mpi_main, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, thread_normsqdata[i].data(), thread_normsqdata[i].size(), MPI_DOUBLE,
                 MPI_SUM, mpi_main, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(thread_data[i].data(), thread_data[i].data(), thread_data[i].size(),
                 MPI_DOUBLE_COMPLEX, MPI_SUM, mpi_main, MPI_COMM_WORLD);
      MPI_Reduce(thread_normsqdata[i].data(), thread_normsqdata[i].data(),
                 thread_normsqdata[i].size(), MPI_DOUBLE, MPI_SUM, mpi_main, MPI_COMM_WORLD);
    }
    if (main_thread) {
      std::vector<std::complex<double>> ss_meas_data;
      std::vector<double> ss_meas_err;
      // Now normalize and calculate the error for each data point
      for (size_t j = 0; j < thread_data[i].size(); j++) {
        // Sample mean
        ss_meas_data.push_back(thread_data[i][j] / static_cast<double>(mpi_size));
        if (subspaces[i] != 0) {
          // Naive sample variance is s^2_N = <(x - <x>)^2> = <x^2> - <x>^2
          double naive_sample_variance = (thread_normsqdata[i][j] / static_cast<double>(mpi_size)) -
                                         std::norm(ss_meas_data[j]);
          // Unbiased (Bessel-corrected) sample variance is s^2_{N-1} = s^2_N * (N / (N - 1)),
          // giving a standard error estimate s_{N-1} = s_N * sqrt(N / (N - 1))
          ss_meas_err.push_back(
              std::sqrt(naive_sample_variance * mpi_size / static_cast<double>(mpi_size - 1)));
        }
      }
      meas_data.push_back(ss_meas_data);
      if (subspaces[i] != 0) {
        meas_err.push_back(ss_meas_err);
      }
    }
  }
  if (debug) {
    std::cout << "Done building meas data/errs." << std::endl;
  }
  if (main_thread) {
    // Open the output H5 file for writing; if it already exists, don't rewrite over it!
    H5::H5File h5file;
    int dup_count = 0;
    bool is_duplicate = false;
    std::string full_fpath = filename + ".h5";
    if (!job_id.empty()) {
      full_fpath = job_id + "/" + filename + "_" + job_id + ".h5";
    }
    do {
      try {
        H5::Exception::dontPrint();
        h5file = H5::H5File(full_fpath, H5F_ACC_EXCL);
        is_duplicate = false;
      } catch (H5::FileIException error) {
        // error.printErrorStack();
        std::cerr << "File '" << full_fpath << "' already exists." << std::endl;
        is_duplicate = true;
        ++dup_count;
        filename = filename + "_" + std::to_string(dup_count);
      }
      if (dup_count > 10) {
        throw std::runtime_error("Unable to save data, too many duplicates!");
      }
    } while (is_duplicate);
    std::cout << "\nWriting data to H5 file '" + full_fpath + "'..." << std::endl;

    // Save MCMC parameters as HDF5 attributes
    params.save_to_h5<H5::H5File>(h5file);
    add_attribute_h5<bool>(normalized, "normalized", h5file);
    add_attribute_h5<int>(integrator.n_ascend, "n_ascend", h5file);
    add_attribute_h5<int>(integrator.n_descend, "n_descend", h5file);
    add_attribute_h5<int>(integrator.n_mutate, "n_mutate", h5file);
    add_attribute_h5<int>(integrator.max_order, "max_order", h5file);
    add_attribute_h5<int>(integrator.v_meas_ext, "v_meas_ext", h5file);
    add_attribute_h5<int>(integrator.n_subspaces, "n_subspaces", h5file);
    add_attribute_h5<int>(static_cast<int>(integrator.timestamp), "timestamp", h5file);
    add_attribute_h5<double>(integrator.norm_const, "norm_const", h5file);
    add_attribute_h5<double>(integrator.d0_weight, "d0_weight", h5file);
    add_attribute_h5<std::string>(integrator.diag_typestring, "diag_typestring", h5file);
    if (!integrator.name.empty()) {
      add_attribute_h5<std::string>(integrator.name, "integrator_name", h5file);
    }

    // Build an HDF5 compound datatype for complex numbers out of complex_t structs
    typedef struct complex_t {
      double re;
      double im;
      complex_t(double re_, double im_) : re(re_), im(im_) {}
    } complex_t;
    H5::CompType hcomplextype(sizeof(complex_t));
    hcomplextype.insertMember("re", HOFFSET(complex_t, re), H5::PredType::NATIVE_DOUBLE);
    hcomplextype.insertMember("im", HOFFSET(complex_t, im), H5::PredType::NATIVE_DOUBLE);

    // Write the subspace measurement results to labeled datasets
    for (size_t i = 1; i < subspaces.size(); i++) {
      // Save zeroth statistical moment (measurement mean, or if unnormalized, sum)
      const std::string data_postfix = (normalized && (i > 0)) ? "_meas_mean" : "_meas_sum";
      H5std_string ss_data_name("V" + std::to_string(subspaces[i]) + data_postfix);
      if (verbose) {
        std::cout << "Saving " << ss_data_name << "...";
      }
      // Convert the subspace data to a vector of complex_t's
      std::vector<complex_t> ss_data;
      for (size_t j = 0; j < meas_data[i].size(); j++) {
        ss_data.push_back(complex_t(meas_data[i][j].real(), meas_data[i][j].imag()));
      }
      // Write the zeroth statistical moment to H5 file
      int rank = 1;
      hsize_t dim[] = {meas_data[i].size()};
      H5::DataSpace dataspace(rank, dim);
      H5::DataSet dataset = h5file.createDataSet(ss_data_name, hcomplextype, dataspace);
      dataset.write(ss_data.data(), hcomplextype);
      if (verbose) {
        std::cout << "done!" << std::endl;
        if (debug) {
          for (complex_t data_pt : ss_data) {
            std::cout << data_pt.re << " + " << data_pt.im << "i" << std::endl;
          }
        }
      }
      if (subspaces[i] != 0) {
        // Save first statistical moment (sample standard error, with Bessel's correction)
        H5std_string ss_err_name("V" + std::to_string(subspaces[i]) + "_meas_stderr");
        if (verbose) {
          std::cout << "Saving " << ss_err_name << "...";
        }
        // Write the first statistical moment to H5 file
        rank = 1;
        dim[0] = {meas_err[i - 1].size()};
        H5::DataSpace dataspace2(rank, dim);
        H5::DataSet dataset2 =
            h5file.createDataSet(ss_err_name, H5::PredType::NATIVE_DOUBLE, dataspace2);
        dataset2.write(meas_err[i - 1].data(), H5::PredType::NATIVE_DOUBLE);
        if (verbose) {
          std::cout << "done!" << std::endl;
        }
      }
    }
  }
}
#endif

int main(int argc, char* argv[]) {
  // Use test input parameters
  using namespace test_input;

  // Default MPI params (for serial runs)
  int mpi_size = 1;
  int mpi_rank = 0;
  int mpi_main = 0;
  bool main_thread = true;

#ifdef _MPI
  // Get MPI parameters for parallel runs
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  mpi_main = 0;
  main_thread = (mpi_rank == mpi_main);
  // Allow regular/debug level verbosity only on the main thread
  if (!main_thread) {
    debug = false;
    verbose = false;
  }
  // Don't save serial results if we run with MPI
  save_serial = false;
#endif

  // Parse the config file (JSON with Comments) as JSON
  std::ifstream config_file("config.jsonc");
  json config = jsonc_parse(config_file);

  // Write parsed JSON to file for debugging
  std::ofstream jout_file(".parsed_config.json");
  jout_file << std::setw(4) << config << std::endl;

  std::vector<double> U_list;
  if (batch_U) {
    U_list = {U_batch};
  } else {
    U_list = {U_loc};
  }
  for (double this_U : U_list) {
    // Update test param U value
    U_loc = this_U;

    // Update normalization diagram weight test param D_0 value
    D_0 = (-0.000143769 + 0.0143555 * this_U + 0.0571346 * this_U * this_U) *
          (-0.0630317 + 0.0794737 * beta + 0.00206871 * beta * beta);

    std::ofstream logfile;
    std::time_t starttime = std::time(nullptr);

    // Job ID defined via UNIX timestamp
    std::string job_id = "run_" + std::to_string(starttime);

    if (main_thread) {
      // Make subdirectory for results (if it doesn't already exist)
      std::filesystem::create_directory(job_id);
      logfile.open(job_id + "/" + save_name + "_" + job_id + ".log");
      logfile << "MPI run with " << mpi_size << " thread(s)" << std::endl;
      std::cout << "MPI run with " << mpi_size << " thread(s)" << std::endl;
      // Write start time to logfile and stdout
      logfile << "Job started at:\t" << std::asctime(std::localtime(&starttime))
              << "UNIX timestamp:\t" << starttime << std::endl;
      logfile.close();
      std::cout << "Job started at:\t" << std::asctime(std::localtime(&starttime))
                << "UNIX timestamp:\t" << starttime << std::endl;
    }

    try {
      // Find and load Green's function data consistent with the current test parameters;
      // we look recursively for any h5 files in the "propagators" parent directory.
      lattice_2d_f_interp lat_g0_r_tau;
      std::string propr_dirname = "propagators";
      for (const auto& rdir_entry : std::filesystem::recursive_directory_iterator(propr_dirname)) {
        // If this is a subdirectory, continue iterating
        if (rdir_entry.is_directory()) {
          continue;
        }
        if (debug) {
          std::cout << rdir_entry << std::endl;
        }
        // Get the path string for this directory entry (a file)
        const std::string& path_string = rdir_entry.path().string();
        // If this is a Green's function HDF5 file with consistent parameters, load it
        // if (H5::H5File::isHdf5(path_string.c_str()) && (path_string.find("g0") !=
        // std::string::npos) &&
        if (H5::H5File::isHdf5(path_string) && (path_string.find("g0") != std::string::npos) &&
            params_consistent(path_string, main_thread)) {
          if (main_thread) {
            std::cout << "\nFound consistent Green's function data in propagator subdirectory '"
                      << path_string << "', loading it...";
          }
          lat_g0_r_tau = load_g0_h5(path_string);
          if (main_thread) {
            std::cout << "done!" << std::endl;
          }
          break;
        }
      }
      if (lat_g0_r_tau.data.empty()) {
        throw std::logic_error("No applicable Green's function data found!");
      }

      // Build V_0 diagram pool (empty)
      const diagram_pool_el diags_v0;

      // Build V_3 diagram pool; there are 5 nonzero topologies at third order (O(U^2))
      const edge_list b_edge_list = {{2, 3}, {4, 5}};  // bos_edges(D^i_n) (common basis)
      const edge_lists f_edge_lists = {
          {{0, 4}, {4, 2}, {2, 1}, {1, 0}, {3, 5}, {5, 3}},  // ferm_edges(D^1_n)
          {{1, 4}, {4, 2}, {2, 0}, {0, 1}, {3, 5}, {5, 3}},  // ferm_edges(D^2_n)
          {{0, 4}, {4, 1}, {1, 2}, {2, 0}, {3, 5}, {5, 3}},  // ferm_edges(D^3_n)
          {{0, 4}, {4, 2}, {2, 0}, {1, 3}, {3, 5}, {5, 1}},  // ferm_edges(D^4_n)
          {{0, 4}, {4, 2}, {2, 0}, {1, 5}, {5, 3}, {3, 1}},  // ferm_edges(D^5_n)
      };
      const graphs_el graphs_vn(b_edge_list, f_edge_lists);
      const std::vector<int> symm_factors = {1, 1, 1, 1, 1};
      const std::vector<int> n_loops = {2, 2, 2, 2, 2};
      const vertices_3pt_el nn_vertices(f_edge_lists);
      const diagram_pool_el diags_vn(s_ferm, order, n_legs, n_intn, n_times, n_posns, graphs_vn,
                                     symm_factors, n_loops, nn_vertices);
      if (debug && main_thread) {
        double n_diags = 5;
        double n_edges = 2 * order;
        for (size_t i = 0; i < n_diags; i++) {
          std::cout << "\nDiagram #" << i << ":\n";
          for (size_t j = 0; j < n_edges; j++) {
            edge_t f_edge_in = nn_vertices.f_edge_in_lists[i][j];
            edge_t f_edge_out = nn_vertices.f_edge_out_lists[i][j];
            std::cout << "Fermionic edges in 3-vertex with base v_" << i << ": ";
            std::cout << "[" << f_edge_in[0] << ", " << f_edge_in[1] << "], ";
            std::cout << "[" << f_edge_out[0] << ", " << f_edge_out[1] << "], " << std::endl;
          }
        }
      }

      // Combined diagram pools
      const diagram_pools_el diag_pools = {diags_v0, diags_vn};

      // External measurement frequencies
      std::vector<double> nu_list;
      for (int m = 0; m < n_nu_meas; m++) {
        nu_list.push_back((2 * m) * M_PI / beta);  // i nu_m = i (2 m pi T)
      }

      // External measurement k-path coordinate indices;
      // we measure the susceptibility at all k-points along the
      // G-X-M-G high-symmetry path in the first Brilloin zone
      std::vector<std::vector<int>> path_nk_coords;
      const int n_bz_edge = static_cast<int>(std::floor(n_site_pd / 2.0));
      const int n_k_meas = 3 * n_bz_edge;
      std::vector<int> ik_G_X(n_bz_edge);                  // Excludes duplicate endpoint X
      std::vector<int> ik_X_M(n_bz_edge);                  // Excludes duplicate endpoint M
      std::vector<int> ik_M_G(n_bz_edge);                  // Excludes duplicate endpoint G
      std::iota(std::begin(ik_G_X), std::end(ik_G_X), 0);  // ik_G_X = range(0, n_bz_edge)
      std::iota(std::begin(ik_X_M), std::end(ik_X_M), 0);  // ik_X_M = range(0, n_bz_edge)
      std::iota(std::rbegin(ik_M_G), std::rend(ik_M_G),
                1);  // ik_M_G = range(1, n_bz_edge)[::-1]
      for (const auto nk_x : ik_G_X) {
        path_nk_coords.push_back({nk_x, 0});
      }
      for (const auto nk_y : ik_X_M) {
        path_nk_coords.push_back({n_bz_edge, nk_y});
      }
      for (const auto nk_xy : ik_M_G) {
        path_nk_coords.push_back({nk_xy, nk_xy});
      }
      // Total number of k-path coordinates
      assert(n_k_meas == path_nk_coords.size());

      // Build the external momentum-frequency measurement coordinates;
      // they are explicitly labeled, so we store them as a 1D array
      hc_lat_mf_coords mf_meas_coords;
      const int n_meas_coords = n_nu_meas * n_k_meas;
      for (int i = 0; i < n_nu_meas; i++) {
        for (int j = 0; j < n_k_meas; j++) {
          const int id = i * n_k_meas + j;
          mf_meas_coords.push_back(
              hc_lat_mf_coord(id, nu_list[i], path_nk_coords[j], n_site_pd, lat_const));
        }
      }
      assert(n_meas_coords == mf_meas_coords.size());
      if (debug) {
        // Print the list of current spacetime coordinates
        for (size_t i = 0; i < mf_meas_coords.size(); i++) {
          const bool toprule = (i == 0);
          mf_meas_coords[i].print(std::cout.rdbuf(), toprule);
        }
      }

      // Initialize the MCMC observables {Tr[sgn(D_0)], Tr[sgn(D_n) exp(-iq * r + iq_m * tau)]}
      meas_t meas_sums = {{0}, std::vector<std::complex<double>>(n_meas_coords, 0)};
      assert(meas_sums.size() == subspaces.size());

      // Build MCMC parameter class using test parameters
      const mcmc_lat_ext_hub_params params(
          mcmc_lattice_params(dim, n_warm, n_meas, n_skip, n_site_pd, num_elec, vol_lat, ef, mu, rs,
                              beta, t_hop, delta_tau, lat_const),
          max_posn_shift, n_nu_meas, n_k_meas, mu_tilde, this_U);

      // Finally, build the MCMC integrator object
      mcmc_cfg_2d_sq_hub_mf_meas mcmc_integrator(params, lat_g0_r_tau, D_0, diag_type, diag_pools,
                                                 subspaces, mf_meas_coords, meas_sums, verbose,
                                                 debug, starttime);

      // Now integrate! (optionally, saving thread subresults to hdf5 (if save_serial = true))
      mcmc_integrator.integrate(job_id, save_name, normalize, save_serial, main_thread);

      // // Integration dry run (results summarized but unsaved)
      // mcmc_integrator.integrate();

#ifdef _MPI
      if (mpi_size == 1) {
        // We cannot compute error bars for an MPI run with one thread (serial)
        mcmc_integrator.save(job_id, save_name);
      } else {
        // Otherwise, compute standard error over MPI threads and save the result with error bars
        aggregate_and_save(mpi_size, mpi_rank, mpi_main, mcmc_integrator, job_id, save_name);
      }
#endif
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }

    if (main_thread) {
      // Write end/elapsed time to logfile and stdout
      std::time_t endtime = std::time(nullptr);
      logfile.open(job_id + "/" + save_name + "_" + job_id + ".log", std::ofstream::app);
      logfile << "\nJob ended at:\t" << std::asctime(std::localtime(&endtime)) << "Elapsed time:\t"
              << std::difftime(endtime, starttime) << " seconds" << std::endl;
      logfile.close();
      std::cout << "\nJob ended at:\t" << std::asctime(std::localtime(&endtime))
                << "Elapsed time:\t" << std::difftime(endtime, starttime) << " seconds"
                << std::endl;
    }
  }

#ifdef _MPI
  MPI_Finalize();
#endif
  return 0;
}