#include "diagmc_hubbard_2dsqlat.hpp"
#include "diagmc_includes.hpp"
#include "diagmc_tools.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

// Filesystem features are part of the standard library as of C++17
#if HAVE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#elif HAVE_EXPTL_FS
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#elif HAVE_BOOST_FS
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif

// We use the ordered_json type to preserve insertion order
using json = nlohmann::json;

// Defines the measurement type: a Matsubara (4-momentum) correlation function
//                               for the Hubbard model on a 2D square lattice
using meas_t = mcmc_cfg_2d_sq_hub_mf_meas::meas_t;

// Update the original YAML config file with parameters which may have been edited
// during the MCMC run (namely: {norm_space_weight, n_k_meas, save_serial, job_id, save_dir}).
void update_yaml_config(const hub_2dsqlat_mcmc_config& cfg,
                        const std::string& yaml_filename = "config.yml") {
  std::string tmp_filename = ".tmpfile";
  std::ofstream tempfile(tmp_filename);
  std::vector<std::string> update_groups = {"diag", "mcmc", "phys"};
  // Avoid precision loss on conversion of double value norm_space_weight to string
  std::stringstream norm_space_weight_full;
  norm_space_weight_full << std::setprecision(20) << cfg.diag.norm_space_weight;
  // A dictionary of the parameters to be updated
  std::map<std::string, std::string> param_updates = {
      {"norm_space_weight", norm_space_weight_full.str()},
      {"n_k_meas", std::to_string(cfg.mcmc.n_k_meas)},
      {"save_serial", cfg.mcmc.save_serial ? "true" : "false"},
      {"job_id", std::to_string(cfg.mcmc.job_id.value())},
      {"save_dir", cfg.mcmc.save_dir.value()},
  };
  // Read through the YAML file line-by-line
  std::string line;
  std::string current_group = "None";
  std::ifstream yaml_config(yaml_filename);
  // Matches any unindented YAML key, i.e., an option group
  const std::regex group_marker("^\\w+:\\s*");
  while (std::getline(yaml_config, line)) {
    // Strip comments from the current line
    std::string line_no_comments = line.substr(0, line.find("#"));
    // Update the current option group as we read
    // through the file if we encounter a new one
    if (std::regex_search(line_no_comments, group_marker)) {
      current_group = line_no_comments;
      // Remove the semi-colon from the group name
      current_group.pop_back();
    }
    // Only search for param updates in updatable groups
    // (for partial support of duplicate param names in separate groups)
    if (std::find(update_groups.begin(), update_groups.end(), current_group) !=
        update_groups.end()) {
      for (const auto& [param, new_value] : param_updates) {
        if (string_contains(line_no_comments, param + ":")) {
          std::string indented_key = line.substr(0, line.find(":"));
          // Preserve inline comments during replacement
          std::string inline_comments = "";
          if (string_contains(line, "#")) {
            inline_comments = line.substr(line.find("#") - 1);
          }
          line = indented_key + ": " + new_value + inline_comments;
          break;
        }
      }
    }
    // Write this (possibly updated) line to the tempfile
    tempfile << line << std::endl;
  }
  // Finally, move the tempfile into the original config file
  fs::rename(tmp_filename, yaml_filename);
  return;
}

// Check if the given propagator data is physically consistent with the current run configuration
bool consistent_propagator_data(const std::string& h5_filename, const json& config,
                                bool verbose = true) {
  // Open H5 file containing the propagator data
  H5::H5File file = H5::H5File(h5_filename, H5F_ACC_RDONLY);
  // Deserialize the propagator configuration data to
  // JSON from its corresponding H5 (string) attribute
  H5std_string cfg = load_h5_attribute<std::string, H5::H5File>("config", file);
  if (verbose) {
    std::cout << "Loaded propagator config from HDF5..." << std::endl;
  }
  json propr_config = json::parse(cfg);
  if (verbose) {
    std::cout << "Parsed propagator config to JSON:" << std::endl;
    std::cout << propr_config << std::endl;
  }
  // Run and propagator configurations should not differ physically
  if ((config["phys"] != propr_config["phys"]) || (config["propr"] != propr_config["propr"])) {
    if (verbose) {
      // Diff each relevant group of the current run and propagator data JSON objects
      json diff_phys_params = json::diff(config["phys"], propr_config["phys"]);
      json diff_propr_params = json::diff(config["propr"], propr_config["propr"]);
      // Print any difference(s) in parameters
      std::cout << "Physical parameters for the current run and the propagator data at '"
                << h5_filename << "' differ as follows:\n";
      if (!diff_phys_params.empty()) {
        std::cout << R"(diff config["phys"] propr_config["phys"] =)"
                  << "\n"
                  << std::setw(4) << diff_phys_params << std::endl;
      }
      if (!diff_propr_params.empty()) {
        std::cout << R"(diff config["propr"] propr_config["propr"] =)"
                  << "\n"
                  << std::setw(4) << diff_propr_params << std::endl;
      }
    }
    return false;
  }
  if (verbose) {
    std::cout << "Found consistent propagator data!" << std::endl;
  }
  return true;
}

// Loads the lattice Green's function data from HDF5
lattice_2d_f_interp load_greens_function_h5(const std::string& h5_filename,
                                            const hub_2dsqlat_mcmc_config& cfg) {
  int ibuffer;
  double dbuffer;

  // Open H5 file
  H5::H5File file = H5::H5File(h5_filename, H5F_ACC_RDONLY);

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
  std::size_t size = static_cast<std::size_t>(dims[0]);

  // Read the (flattened) lattice G_0 data
  std::vector<double> tau_mesh(size);
  dataset.read(tau_mesh.data(), datatype);
  const double delta_tau = tau_mesh[1];

  if (cfg.mcmc.debug) {
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
    size += static_cast<std::size_t>(dims_g0[i]);
  }

  // Read the (flattened) lattice G_0 data
  std::vector<double> lat_g0_vec(size);
  dataset.read(lat_g0_vec.data(), datatype);

  if (cfg.mcmc.debug) {
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
  size = static_cast<std::size_t>(dims[0]);
  assert(size == 3);

  // Read lattice G_0 data attribute 'shape'
  std::vector<long> shape(size);
  shape_attr.read(shape_type, shape.data());

  if (cfg.mcmc.debug) {
    std::cout << "\nG_0 data shape = ( ";
    for (const auto& i : shape) {
      std::cout << i << " ";
    }
    std::cout << ")\n" << std::endl;
  }

  // G_0(r, tau) mesh is (d + 1)-dimensional
  const int n_site_irred = cfg.phys.n_site_irred;
  assert(shape == std::vector<long>({n_site_irred, n_site_irred, n_tau}));

  //////////////////////////////////
  // Build G_0 interpolant matrix //
  //////////////////////////////////

  // Build a vector of linear 1D interpolants from flattened lattice G_0 data
  const int n_tau = cfg.propr.n_tau;
  const double beta = cfg.phys.beta;
  std::vector<f_interp_1d> lat_g0_vec_interp;
  for (std::size_t i = 0; i < n_site_irred; ++i) {
    for (std::size_t j = 0; j < n_site_irred; ++j) {
      std::vector<double> g0_tau_data;
      for (std::size_t k = 0; k < n_tau; ++k) {
        g0_tau_data.push_back(lat_g0_vec[k + n_tau * (j + n_site_irred * i)]);
      }
      lat_g0_vec_interp.push_back(f_interp_1d(fmesh_1d(g0_tau_data, tau_mesh), beta));
    }
  }
  lattice_2d_f_interp lat_g0_r_tau(n_site_irred, n_site_irred, lat_g0_vec_interp);
  return lat_g0_r_tau;
}

#ifdef HAVE_MPI
// Aggregate MPI results, compute the standard error over threads, and save the results to HDF5
void aggregate_and_save(int mpi_size, int mpi_rank, int mpi_main,
                        const mcmc_cfg_2d_sq_hub_mf_meas& integrator) {
  bool is_main_thread = (mpi_rank == mpi_main);
  bool normalized = integrator.normalized;

  // Reference to integrator cfg for brevity
  const hub_2dsqlat_mcmc_config& cfg = integrator.cfg;

  meas_t thread_data;
  if (normalized) {
    thread_data = integrator.meas_means;
  } else {
    thread_data = integrator.meas_sums;
  }

  // To avoid recording the full covariance matrix, we assume a complex "error circle",
  // i.e., we define the error using the norm squared means; this can heavily
  // overestimate the error of the smaller component if, e.g., the real/imag
  // scales differ greatly.
  typedef std::vector<std::vector<double>> err_t;
  err_t thread_normsqdata;
  for (std::size_t i = 0; i < cfg.diag.subspaces.size(); i++) {
    const std::vector<std::complex<double>>& ss_data = thread_data[i];
    std::vector<double> ss_normsqdata;
    for (std::size_t j = 0; j < ss_data.size(); j++) {
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
  for (std::size_t i = 0; i < cfg.diag.subspaces.size(); i++) {
    // Sum over threads
    if (is_main_thread) {
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
    if (is_main_thread) {
      std::vector<std::complex<double>> ss_meas_data;
      std::vector<double> ss_meas_err;
      // Now normalize and calculate the error for each data point
      for (std::size_t j = 0; j < thread_data[i].size(); j++) {
        // Sample mean
        ss_meas_data.push_back(thread_data[i][j] / static_cast<double>(mpi_size));
        if (cfg.diag.subspaces[i] != 0) {
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
      if (cfg.diag.subspaces[i] != 0) {
        meas_err.push_back(ss_meas_err);
      }
    }
  }
  if (cfg.mcmc.debug && is_main_thread) {
    std::cout << "Done building meas data/errs." << std::endl;
  }
  if (is_main_thread) {
    // Open the output H5 file for writing; if it already exists, don't rewrite over it!
    H5::H5File h5file;
    int dup_count = 0;
    bool is_duplicate = false;
    std::string filename = cfg.mcmc.save_dir.value() + "/" + cfg.mcmc.save_name + "_run_" +
                           std::to_string(cfg.mcmc.job_id.value());
    std::string extension = ".h5";
    do {
      try {
        H5::Exception::dontPrint();
        h5file = H5::H5File(filename + extension, H5F_ACC_EXCL);
        is_duplicate = false;
      } catch (const H5::FileIException& error) {
        error.printErrorStack();
        std::cerr << "File '" << filename + extension << "' already exists." << std::endl;
        is_duplicate = true;
        ++dup_count;
        extension = "_" + std::to_string(dup_count) + ".h5";
      }
      if (dup_count > 10) {
        throw std::runtime_error("Unable to save data, too many duplicates!");
      }
    } while (is_duplicate);
    std::cout << "\nWriting data to H5 file '" + h5file.getFileName() + "'..." << std::endl;

    // Save the run cfg to an HDF5 string attribute as serialized JSON
    add_attribute_h5<std::string>(static_cast<json>(cfg).dump(), "config", h5file);

    // Save extra MCMC parameters as HDF5 attributes
    add_attribute_h5<bool>(normalized, "normalized", h5file);
    add_attribute_h5<int>(integrator.n_ascend, "n_ascend", h5file);
    add_attribute_h5<int>(integrator.n_descend, "n_descend", h5file);
    add_attribute_h5<int>(integrator.n_mutate, "n_mutate", h5file);
    add_attribute_h5<int>(integrator.max_order, "max_order", h5file);
    add_attribute_h5<int>(integrator.v_meas_ext, "v_meas_ext", h5file);
    add_attribute_h5<int>(integrator.n_subspaces, "n_subspaces", h5file);
    add_attribute_h5<double>(integrator.norm_const, "norm_const", h5file);
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
    for (std::size_t i = 1; i < cfg.diag.subspaces.size(); i++) {
      // Save zeroth statistical moment (measurement mean, or if unnormalized, sum)
      const std::string data_postfix = (normalized && (i > 0)) ? "_meas_mean" : "_meas_sum";
      H5std_string ss_data_name("V" + std::to_string(cfg.diag.subspaces[i]) + data_postfix);
      if (cfg.mcmc.verbose) {
        std::cout << "Saving " << ss_data_name << "...";
      }
      // Convert the subspace data to a vector of complex_t's
      std::vector<complex_t> ss_data;
      for (std::size_t j = 0; j < meas_data[i].size(); j++) {
        ss_data.push_back(complex_t(meas_data[i][j].real(), meas_data[i][j].imag()));
      }
      // Write the zeroth statistical moment to H5 file
      int rank = 1;
      hsize_t dim[] = {meas_data[i].size()};
      H5::DataSpace dataspace(rank, dim);
      H5::DataSet dataset = h5file.createDataSet(ss_data_name, hcomplextype, dataspace);
      dataset.write(ss_data.data(), hcomplextype);
      if (cfg.mcmc.verbose) {
        std::cout << "done!" << std::endl;
        if (cfg.mcmc.debug) {
          for (complex_t data_pt : ss_data) {
            std::cout << data_pt.re << " + " << data_pt.im << "i" << std::endl;
          }
        }
      }
      if (cfg.diag.subspaces[i] != 0) {
        // Save first statistical moment (sample standard error, with Bessel's correction)
        H5std_string ss_err_name("V" + std::to_string(cfg.diag.subspaces[i]) + "_meas_stderr");
        if (cfg.mcmc.verbose) {
          std::cout << "Saving " << ss_err_name << "...";
        }
        // Write the first statistical moment to H5 file
        rank = 1;
        dim[0] = {meas_err[i - 1].size()};
        H5::DataSpace dataspace2(rank, dim);
        H5::DataSet dataset2 =
            h5file.createDataSet(ss_err_name, H5::PredType::NATIVE_DOUBLE, dataspace2);
        dataset2.write(meas_err[i - 1].data(), H5::PredType::NATIVE_DOUBLE);
        if (cfg.mcmc.verbose) {
          std::cout << "done!" << std::endl;
        }
      }
    }
  }
}
#endif

int main(int argc, char* argv[]) {
  try {
    // Default MPI params (for serial runs)
    int mpi_size = 1;
    int mpi_rank = 0;
    int mpi_main = 0;
    bool is_main_thread = true;

    // Load configuration from JSON
    std::ifstream config_file("config.json");
    json j_config;
    config_file >> j_config;

    // Deserialize to a config object
    auto cfg = j_config.get<hub_2dsqlat_mcmc_config>();

#ifdef HAVE_MPI
    // Get MPI parameters for parallel runs
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    mpi_main = 0;
    is_main_thread = (mpi_rank == mpi_main);
    // Allow regular/debug level verbosity only on the main thread
    bool debug;
    bool verbose;
    if (is_main_thread) {
      debug = cfg.mcmc.debug;
      verbose = cfg.mcmc.verbose;
    } else {
      debug = false;
      verbose = false;
    }
    // Update the config info with the number of MPI threads in this run
    cfg.mcmc.n_threads = mpi_size;
    // Don't save serial results if we run with MPI
    cfg.mcmc.save_serial = false;
#endif

    if (is_main_thread) {
      std::cout << "Loaded config from JSON:" << std::endl;
      std::cout << std::setw(4) << j_config << "\n" << std::endl;
    }

    std::vector<double> U_list;
    if (cfg.mcmc.use_batch_U) {
      U_list = {cfg.phys.U_batch};
    } else {
      U_list = {cfg.phys.U_loc};
    }
    for (double this_U : U_list) {
      // Update test param U value
      cfg.phys.U_loc = this_U;

      // Update normalization diagram weight test param norm_space_weight (D_0) value
      cfg.diag.norm_space_weight =
          (-0.000143769 + 0.0143555 * this_U + 0.0571346 * this_U * this_U) *
          (-0.0630317 + 0.0794737 * cfg.phys.beta + 0.00206871 * cfg.phys.beta * cfg.phys.beta);

      // Open a logfile for this job
      std::ofstream logfile;

      // Get the job's start time
      std::time_t starttime = std::time(nullptr);

      // Job ID and default MCMC save directory are defined on
      // runtime via UNIX timestamp; add them to the config object
      cfg.mcmc.job_id = starttime;

      // Fall back to default save name/directory values if necessary
      if (cfg.mcmc.save_name.empty()) {
        cfg.mcmc.save_name = "meas_hub_2dsqlat";
      }
      // Update the job save directory if necessary, but don't overwrite an existing
      // user-defined save_dir value unless it is of the default format 'run_{job_id}'
      if (!cfg.mcmc.save_dir.has_value() || cfg.mcmc.save_dir.value().empty() ||
          string_contains(cfg.mcmc.save_dir.value(), "run_")) {
        cfg.mcmc.save_dir = "run_" + std::to_string(starttime);
      }

      if (is_main_thread) {
        // Make subdirectory for results (if it doesn't already exist)
        fs::create_directory(cfg.mcmc.save_dir.value());
        // logfile.open("test.log");
        logfile.open(cfg.mcmc.save_dir.value() + "/" + cfg.mcmc.save_name + "_run_" +
                            std::to_string(cfg.mcmc.job_id.value()) + ".log");
        logfile << "MPI run with " << mpi_size << " thread(s)" << std::endl;
        std::cout << "MPI run with " << mpi_size << " thread(s)" << std::endl;
        // Write start time to logfile and stdout
        logfile << "Job started at:\t" << std::asctime(std::localtime(&starttime))
                << "UNIX timestamp:\t" << starttime << std::endl;
        logfile.close();
        std::cout << "Job started at:\t" << std::asctime(std::localtime(&starttime))
                  << "UNIX timestamp:\t" << starttime << std::endl;
      }

      // Find and load Green's function data consistent with the current test parameters;
      // we look recursively for any h5 files in the "propagators" parent directory.
      lattice_2d_f_interp lat_g0_r_tau;
      for (const auto& dir_entry : fs::recursive_directory_iterator(cfg.propr.save_dir)) {
        // If this is a subdirectory, continue iterating
        if (fs::is_directory(dir_entry)) {
          continue;
        }
        if (debug) {
          std::cout << dir_entry << std::endl;
        }
        // Get the path string for this directory entry (a file)
        const std::string& path_string = dir_entry.path().string();
        // If this is a Green's function HDF5 file with consistent parameters, load it
        if (H5::H5File::isHdf5(path_string) && string_contains(path_string, "g0") &&
            consistent_propagator_data(path_string, j_config, debug)) {
          if (is_main_thread) {
            std::cout << "\nFound consistent Green's function data in propagator subdirectory '"
                      << path_string << "', loading it...";
          }
          lat_g0_r_tau = load_greens_function_h5(path_string, cfg);
          if (is_main_thread) {
            std::cout << "done!" << std::endl;
          }
          break;
        }
      }
      if (lat_g0_r_tau.data.empty()) {
        throw std::runtime_error("No applicable Green's function data found!");
      }

      // Build the measurement (V_3) diagram pool; there
      // are 5 nonzero topologies at third order (O(U^2))
      std::ifstream graph_info("graph_info.json");
      json j_graph_info;
      graph_info >> j_graph_info;

      const auto j_graphs_vn = j_graph_info.at("graphs");

      // Build the normalization (V_0) diagram pool (empty)
      const diagram_pool_el diags_v0;

      // All graph topologies share a common bosonic edge list
      const edge_list b_edge_list = j_graphs_vn.at("bos_edges_common").get<edge_list>();

      // Loop over topologies to load the fermionic edge lists
      edge_lists f_edge_lists;
      for (const auto& [key, value] : j_graphs_vn.at("topologies").items()) {
        f_edge_lists.push_back(value.get<edge_list>());
      }

      // Additional graph info
      const auto n_loops = j_graph_info.at("n_loops").get<std::vector<int>>();
      const auto symm_factors = j_graph_info.at("symm_factors").get<std::vector<int>>();

      // Store nearest-neighbor vertices redundantly for fast local updates at high orders
      const vertices_3pt_el nn_vertices(f_edge_lists);

      // Build the measurement diagram pool (a collection of diagrams, loops, and n.n. vertices)
      const graphs_el graphs_vn(b_edge_list, f_edge_lists);
      const diagram_pool_el diags_vn(cfg, graphs_vn, symm_factors, n_loops, nn_vertices);

      // Build the full set of subspace diagram pools to be passed to the integrator
      const diagram_pools_el diag_pools = {diags_v0, diags_vn};

      if (debug) {
        double n_diags = 5;
        double n_edges = 2 * cfg.diag.order;
        for (std::size_t i = 0; i < n_diags; i++) {
          std::cout << "\nDiagram #" << i << ":\n";
          for (std::size_t j = 0; j < n_edges; j++) {
            edge_t f_edge_in = nn_vertices.f_edge_in_lists[i][j];
            edge_t f_edge_out = nn_vertices.f_edge_out_lists[i][j];
            std::cout << "Fermionic edges in 3-vertex with base v_" << i << ": ";
            std::cout << "[" << f_edge_in[0] << ", " << f_edge_in[1] << "], ";
            std::cout << "[" << f_edge_out[0] << ", " << f_edge_out[1] << "], " << std::endl;
          }
        }
      }

      // Compute the external measurement Matsubara frequencies
      int eta;
      if (cfg.diag.diag_type == "charge_poln") {
        eta = 0;
      } else if (string_contains(cfg.diag.diag_type, "self_en")) {
        eta = 1;
      } else {
        throw std::invalid_argument(
            "Invalid diagram type '" + cfg.diag.diag_type +
            R"(', choose from: {"charge_poln", "self_en_stat", "self_en_dyn"}.)");
      }
      std::vector<double> nu_list;
      for (int m = 0; m < cfg.mcmc.n_nu_meas; m++) {
        nu_list.push_back((2 * m + eta) * M_PI / cfg.phys.beta);
      }

      // Load the measurement k-path coordinate indices from JSON
      std::ifstream k_path_file("k_path_info.json");
      json k_path_info;
      k_path_file >> k_path_info;
      const auto path_nk_coords = k_path_info.at("k_path").get<std::vector<std::vector<int>>>();
      assert(cfg.mcmc.n_k_meas == path_nk_coords.size());
      
      // Build the external momentum-frequency measurement coordinates;
      // they are explicitly labeled, so we store them as a 1D array
      hc_lat_mf_coords mf_meas_coords;
      const int n_meas_coords = cfg.mcmc.n_nu_meas * cfg.mcmc.n_k_meas;
      int id = 0;
      for (const auto& nu : nu_list) {
        for (const auto& nk_vec : path_nk_coords) {
          mf_meas_coords.push_back(
              hc_lat_mf_coord(id, nu, nk_vec, cfg.phys.n_site_pd, cfg.phys.lat_const));
          ++id;
        }
      }
      assert(n_meas_coords == mf_meas_coords.size());
      if (debug) {
        // Print the list of current spacetime coordinates
        for (std::size_t i = 0; i < mf_meas_coords.size(); i++) {
          const bool toprule = (i == 0);
          mf_meas_coords[i].print(std::cout.rdbuf(), toprule);
        }
      }

      // Initialize the MCMC observables {Tr[sgn(D_0)], Tr[sgn(D_n) exp(-iq * r + iq_m * tau)]}
      meas_t meas_sums = {{0}, std::vector<std::complex<double>>(n_meas_coords, 0)};
      assert(meas_sums.size() == subspaces.size());

      // Finally, build the MCMC integrator object
      mcmc_cfg_2d_sq_hub_mf_meas mcmc_integrator(cfg, lat_g0_r_tau, diag_pools, mf_meas_coords,
                                                 meas_sums);

      // Now integrate! (optionally, saving thread subresults to hdf5 (if save_serial = true))
      mcmc_integrator.integrate(is_main_thread);

#ifdef HAVE_MPI
      if (mpi_size == 1) {
        // We cannot compute error bars for an MPI run with one thread (serial)
        mcmc_integrator.save();
      } else {
        // Otherwise, compute standard error over MPI threads and save the result with error bars
        aggregate_and_save(mpi_size, mpi_rank, mpi_main, mcmc_integrator);
      }
#endif

      if (is_main_thread) {
        // Convert the updated config object to JSON and write back to file
        json updated_j_config = cfg;
        std::ofstream outfile("config.json");
        outfile << std::setw(4) << updated_j_config << std::endl;

        // Also update relevant entries in the YAML config file (by hand)
        update_yaml_config(cfg);

        // Copy all config / JSON files to the run's save directory
        for (const auto& filename :
             {"config.yml", "config.json", "graph_info.json", "k_path_info.json"}) {
          fs::copy_file(filename, cfg.mcmc.save_dir.value() + "/" + filename);
        }

        // Write end/elapsed time to logfile and stdout
        std::time_t endtime = std::time(nullptr);
        // logfile.open("test.log", std::ofstream::app);
        logfile.open(cfg.mcmc.save_dir.value() + "/" + cfg.mcmc.save_name + "_run_" +
                         std::to_string(cfg.mcmc.job_id.value()) + ".log",
                     std::ofstream::app);
        logfile << "\nJob ended at:\t" << std::asctime(std::localtime(&endtime))
                << "Elapsed time:\t" << std::difftime(endtime, starttime) << " seconds"
                << std::endl;
        logfile.close();
        std::cout << "\nJob ended at:\t" << std::asctime(std::localtime(&endtime))
                  << "Elapsed time:\t" << std::difftime(endtime, starttime) << " seconds"
                  << std::endl;
      }
    }

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}