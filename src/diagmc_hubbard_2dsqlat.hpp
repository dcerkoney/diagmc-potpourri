#pragma once
#include "diagmc_includes.hpp"
#include "diagmc_tools.hpp"

// Configuration object for real-space MCMC for the Hubbard model on the 2d
// square lattice; it is assumed the measurement is a scalar, or a one-particle
// correlation function that is a function of an external Matsubara 4-vector.
class mcmc_cfg_2d_sq_hub_mf_meas {
 private:
  Rand_engine rand_gen;

 public:
  using meas_t = std::vector<std::vector<std::complex<double>>>;
  // Fields
  mcmc_lat_ext_hub_params params;
  bool work_finished = false;
  bool normalized = false;
  bool debug = false;
  bool verbose = true;
  bool using_local_wf;
  int i_step = 0;
  int n_saved = 0;
  int n_accepted = 0;
  // For checking d0 value / detailed balance
  int n_ascend = 0;
  int n_descend = 0;
  int n_mutate = 0;
  // Indices for accessing the current/proposal subspaces
  int idx_ss_curr = 0;
  int idx_ss_prop = 0;
  int max_order;  // Maximum diagram order in the run
  int n_ext;      // Number of externally-constrained vertices (includies internal
                  // vertices tethered to external ones)
  int diag_type;
  int n_subspaces;
  // The proposal probability ratio is P(nu | nu') / P(nu' | nu), where
  // nu' and nu are the new and old integration variables, respectively.
  // For the Metropolis algorithm, we only need to store the ratio.
  double proposal_ratio;
  double weight_curr;
  double weight_prop;
  // Normalization constant for the MCMC integration
  double norm_const = 1;
  // Scalar weight D_0 for the dummy subspace V_0
  double d0_weight;
  std::vector<int> ss_n_saved;
  std::vector<int> ss_sign_samples;
  // Vector denoting the (diagram) order n of all MCMC subspaces V_n
  std::vector<int> subspaces;
  // Current weights for each diagram topology in all subspaces V_n
  std::vector<std::vector<double>> ss_diag_weights_curr;
  std::vector<std::vector<double>> ss_diag_weights_prop;
  // List of diagram pools for each non-trivial subspace
  diagram_pools_el ss_diags;
  // The name of this mcmc integrator
  std::string name = "";
  std::time_t timestamp;
  // The diagram type should be one of the following:
  // ['vacuum', 'charge_poln', 'self_en_stat', 'self_en_dyn']
  std::string diag_typestring;
  std::vector<int> constraints;
  std::vector<int> mates;
  std::vector<int> modifiables;
  hc_lat_st_coords coords_curr;
  hc_lat_st_coords coords_prop;
  lattice_2d_f_interp lat_g0_r_tau;
  // Measurement coordinates in the Matsubara representation
  hc_lat_mf_coords mf_meas_coords;
  // Measurement results for each subspace;
  // in general, the values can be complex
  meas_t meas_sums;   // raw sums
  meas_t meas_means;  // normalized mean data
  // Constructor
  mcmc_cfg_2d_sq_hub_mf_meas(const mcmc_lat_ext_hub_params &params_,
                             const lattice_2d_f_interp &lat_g0_r_tau_, const double d0_weight_,
                             const std::string &diag_typestring_, const diagram_pools_el &ss_diags_,
                             const std::vector<int> &subspaces_,
                             const hc_lat_mf_coords &mf_meas_coords_, const meas_t &meas_sums_,
                             const bool verbose_ = false, const bool debug_ = false,
                             const std::time_t &timestamp_ = 0)
      // Parent constructor
      : params(params_),
        lat_g0_r_tau(lat_g0_r_tau_),
        d0_weight(d0_weight_),
        diag_typestring(diag_typestring_),
        ss_diags(ss_diags_),
        subspaces(subspaces_),
        mf_meas_coords(mf_meas_coords_),
        meas_sums(meas_sums_),
        verbose(verbose_),
        debug(debug_),
        timestamp(timestamp_) {
    max_order = subspaces_.back();
    n_subspaces = subspaces_.size();
    ss_n_saved.assign(n_subspaces, 0);
    ss_sign_samples.assign(n_subspaces, 0);
    // We currently require a constant subspace V_0 for normalization
    if (subspaces[0] != 0) {
      throw not_implemeted_error(
          "Non-constant subspace normalization techniques not yet implemented!");
    }
    // The turning point for the local ratio method of calculating the weight
    // function occurs when n_ferm_lines > 8, i.e., for diagrams above 4th order
    using_local_wf = (max_order > 4);
    if (using_local_wf) {
      ss_diag_weights_prop.resize(n_subspaces);
      ss_diag_weights_prop[0] = {d0_weight};
      for (size_t i = 1; i < n_subspaces; i++) {
        std::vector<double> row;
        for (size_t j = 0; j < ss_diags[i].n_diags; j++) {
          row.push_back(0.0);
        }
        ss_diag_weights_prop[i] = row;
      }
    }
    // Convert from diagram typestring to an integer convention for compactness.
    // This is especially important for calls to constexpr functions taking the
    // diagram type as an input, e.g., the function 'is_pinned()'.
    std::map<std::string, int> dt_map = {
        {"vacuum", 0}, {"charge_poln", 1}, {"self_en_stat", 2}, {"self_en_dyn", 3}};
    if (dt_map.count(diag_typestring) == 0) {
      throw std::invalid_argument("Invalid diagram type \'" + diag_typestring + "\'!");
    }
    diag_type = dt_map[diag_typestring];
    if (diag_type == 2) {
      std::cout << "Warning: static self-energy measurement is an untested option!" << std::endl;
    }
    // if (diag_type > 2) {
    //   throw not_implemeted_error("Self energy weight function not yet implemented.");
    // }
    // Define the standard (diagram type dependent!) bases for vertex constraints;
    // unconstrained vertices are denoted by a -1 value in the list
    switch (diag_type) {
      case 0:  // Vacuum diagrams have no external vertices
        constraints = {};
        break;
      case 1:  // The two external legs are untethered
        constraints = {-1, -1};
        break;
      case 2:  // The two external legs are tethered to each other
        constraints = {-1, 0};
        break;
      case 3:  // Each external leg is tethered to an internal vertex;
               // by convention, we take the pairs (0,2), (1,3) such
               // that all deviation from the alternating convention
               // is localized to the first 4 vertices in the list
        constraints = {-1, -1, 0, 1};
        break;
    }
    // Store the number of external vertices as a class field
    n_ext = constraints.size();
    // Use a pairwise basis for internal vertex constraints
    for (int i = n_ext; i < 2 * max_order; ++i) {
      constraints.push_back((i % 2 == 1 ? (i - 1) : -1));
    }
    // Also save constrained mates for quick access and diagram-type generality;
    // the constrained mate vertices are: v_mate = argwhere(constraints == v_modif).
    // Analagous to the constraint vector, a -1 represents a mate vertex itself.
    mates.assign(constraints.size(), -1);
    for (size_t i = 0; i < constraints.size(); i++) {
      std::vector<int>::iterator it = std::find(constraints.begin(), constraints.end(), i);
      // Record the mate of each unconstrained vertex we find
      if (it != constraints.end()) {
        mates[i] = std::distance(constraints.begin(), it);
      }
    }
    // Vector holding indices (ID#s) of all modifiable (unconstrained, non-COM)
    // spacetime coordinates, for use with mutate updates
    std::vector<int>::iterator it = constraints.begin() + 1;  // +1 to ignore COM coord
    while ((it = std::find(it, constraints.end(), -1)) != constraints.end()) {
      modifiables.push_back(std::distance(constraints.begin(), it));
      ++it;
    }
    // Set up the lattice binomial generator and distribution; because N_s is
    // even, we bias the distribution so that the mean will end up at the
    // origin, i.e., is at k = N_s / 2
    // int lat_binom_mean = (params_.n_site_pd - 1) / 2.0; (old mean, mu=-0.5)
    int lat_binom_mean = params_.n_site_pd / 2;
    double p_bias = lat_binom_mean / static_cast<double>(params_.n_site_pd - 1);
    lat_binomial.param(Binom_gen::param_type(params_.n_site_pd - 1, p_bias));
    lat_binomial_dist = Binom_dist(params_.n_site_pd - 1, p_bias);
    // For variable maximum step-size position shifts
    double prob_shift = 1.0 / static_cast<double>(2 * params_.max_posn_shift);
    std::vector<double> weights(2 * params_.max_posn_shift + 1, prob_shift);
    // Be sure we have an exactly normalized distribution, and
    // hence include an infinitesimal chance for no shift at all
    weights[params_.max_posn_shift] = 0.0;
    double overflow = 1.0 - std::accumulate(weights.begin(), weights.end(), 0.0);
    weights[params_.max_posn_shift] = overflow;
    assert(std::accumulate(weights.begin(), weights.end(), 0.0) == 1.0);
    posn_shift_gen.param(Discr_gen::param_type(weights));
    // Seed the private random number generator
    // member with a stdlib random device
    std::random_device rd;
    rand_gen.seed(rd());
  }
  // Returns the results of the MCMC run; the default behaviour
  // normalizes the integration by C / N_{s,0} =  D_0 / E_0
  meas_t results(bool normalize = true) {
    if (normalize) {
      // Normalized results already calculated
      if (normalized) {
        return meas_means;
      }
      // Calculate the normalization constant for the MCMC integration
      if (ss_n_saved[0] == 0) {
        std::cout << "Warning: the normalization subspace was unvisited; keeping "
                     "default normalization constant (1) for debugging purposes!"
                  << std::endl;
      } else {
        norm_const = d0_weight / static_cast<double>(ss_n_saved[0]);
      }
      // Get the reweighted (normalized) mean values in all measurement subspaces
      meas_means = meas_sums;
      for (int i_ss = 1; i_ss < n_subspaces; ++i_ss) {
        for (int i_meas = 0; i_meas < meas_means[i_ss].size(); ++i_meas) {
          meas_means[i_ss][i_meas] *= norm_const;
        }
      }
      // Return the normalized measurement results
      normalized = true;
      return meas_means;
    } else {
      // Return the raw measurement results
      return meas_sums;
    }
  }
  // Save measurement results to HDF5
  void save(std::string job_id = "", std::string filename = "") {
    if (filename.empty()) {
      filename = "mcmc_run";
    }
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
    std::cout << "Writing data to H5 file '" + full_fpath + "'..." << std::endl;

    // Save MCMC parameters as HDF5 attributes
    params.save_to_h5<H5::H5File>(h5file);
    add_attribute_h5<int>(normalized, "normalized", h5file);
    add_attribute_h5<int>(n_ascend, "n_ascend", h5file);
    add_attribute_h5<int>(n_descend, "n_descend", h5file);
    add_attribute_h5<int>(n_mutate, "n_mutate", h5file);
    add_attribute_h5<int>(max_order, "max_order", h5file);
    add_attribute_h5<int>(n_ext, "n_ext", h5file);
    add_attribute_h5<int>(n_subspaces, "n_subspaces", h5file);
    add_attribute_h5<int>(static_cast<int>(timestamp), "timestamp", h5file);
    add_attribute_h5<double>(norm_const, "norm_const", h5file);
    add_attribute_h5<double>(d0_weight, "d0_weight", h5file);
    add_attribute_h5<std::string>(diag_typestring, "diag_typestring", h5file);
    if (!name.empty()) {
      add_attribute_h5<std::string>(name, "integrator_name", h5file);
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

    // Save normalized (mean) data if available; otherwise save the raw measurement sums
    const meas_t &data = (normalized) ? meas_means : meas_sums;

    // Write each subspace result to a labeled dataset
    for (size_t i = 0; i < subspaces.size(); i++) {
      const std::string postfix = (normalized && (i > 0)) ? "_meas_mean" : "_meas_sum";
      H5std_string ss_data_name("V" + std::to_string(subspaces[i]) + postfix);
      if (verbose) {
        std::cout << "Saving " << ss_data_name << "...";
      }
      // Convert the subspace data to a vector of complex_t's
      std::vector<complex_t> ss_data;
      for (size_t j = 0; j < data[i].size(); j++) {
        ss_data.push_back(complex_t(data[i][j].real(), data[i][j].imag()));
      }
      // Subspace measurements are always rank 1 (1D)
      const int rank = 1;
      hsize_t dim[] = {data[i].size()};
      H5::DataSpace dataspace(rank, dim);
      H5::DataSet dataset = h5file.createDataSet(ss_data_name, hcomplextype, dataspace);
      dataset.write(ss_data.data(), hcomplextype);
      if (verbose) {
        std::cout << "done!" << std::endl;
        for (complex_t data_pt : ss_data) {
          std::cout << data_pt.re << " + " << data_pt.im << "i" << std::endl;
        }
      }
    }
  }
  // Summarize the current results
  void summarize(std::string job_id = "", std::string filename = "") {
    // Forward stdout to output log h5file if applicable
    std::streambuf *buffer;
    std::ofstream outfile;
    std::string full_fpath = filename + ".log";
    if (filename.empty()) {
      buffer = std::cout.rdbuf();
    } else {
      if (!job_id.empty()) {
        full_fpath = job_id + "/" + filename + "_" + job_id + ".log";
      }
      outfile.open(full_fpath, std::ofstream::app);
      buffer = outfile.rdbuf();
    }
    std::ostream out(buffer);

    if (work_finished) {
      out << "Final results:";
    }
    // Print the integrator name, if applicable
    if (name != "") {
      out << "\n" << name << std::endl;
    }
    // Print the step number
    out << "\n\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n"
        << " MC step #" << i_step
        << "\n\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
           "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n"
        << std::endl;
    // Print subspace info
    std::string ss_string = "In subspace: [";
    for (auto ss : subspaces) {
      ss_string += "V_" + std::to_string(ss);
      if (ss != subspaces.back()) {
        ss_string += " ";
      }
    }
    ss_string += "]\n";
    int slen = ss_string.length();
    for (int i = 0; i < slen; ++i) {
      if (i == slen - 4 * (n_subspaces - idx_ss_curr)) {
        ss_string += "^";
      } else {
        ss_string += " ";
      }
    }
    // ss_string += "\n";
    // out << ss_string << std::endl;
    out << ss_string;

    // Print the list of current spacetime coordinates
    for (size_t i = 0; i < coords_curr.size(); i++) {
      const bool toprule = (i == 0);
      coords_curr[i].print(buffer, toprule);
    }

    // Print the diagram pool in the current subspace
    const diagram_pool_el &diags = ss_diags[idx_ss_curr];
    out << "\nDiagram pool size: " << diags.n_diags << std::endl;
    out << "Diagram pool fermion loop numbers: ( ";
    for (auto n_loop : diags.n_loops) {
      out << n_loop << " ";
    }
    out << ")" << std::endl;
    // if (subspaces[idx_ss_curr] > 0) {
    //   out << "Diagram pool loop list #1: ( ";
    //   for (auto loop : diags.loops[0]) {
    //     out << "( ";
    //     for (auto v : loop) {
    //       out << v << " ";
    //     }
    //     out << ") ";
    //   }
    //   out << ")" << std::endl;
    // }
    // Print the spacetime coordinate constraints
    out << "Constraints on spacetime coordinates: [";
    for (int i = 0; i < constraints.size(); ++i) {
      if (constraints[i] == -1) {
        out << "x";
      } else {
        out << constraints[i];
      }
      if (i != constraints.size() - 1) {
        out << " ";
      }
    }
    out << "]" << std::endl;

    // Measurement summary data
    if (i_step >= params.n_warm) {
      double n_meas_so_far;
      if (work_finished) {
        // If the run is finished, i_step = params.n_meas,
        // and we should not shift by +1 in this case
        n_meas_so_far = params.n_meas;
      } else {
        n_meas_so_far = i_step - params.n_warm + 1;
      }
      // Saved counts (n_saved_V_i)
      out << "\nSubspace saved counts:\t[";
      for (int i = 0; i < n_subspaces; ++i) {
        out << ss_n_saved[i];
        if (i != n_subspaces - 1) {
          out << " ";
        }
      }
      out << "]" << std::endl;
      // Saved ratios (n_saved_V_i / n_meas_steps_curr)
      out << "Subspace saved ratios:\t[";
      for (int i = 0; i < n_subspaces; ++i) {
        out << (ss_n_saved[i] / n_meas_so_far);
        if (i != n_subspaces - 1) {
          out << " ";
        }
      }
      out << "]" << std::endl;
      // Average measured signs (sign_sum_V_i / n_saved_V_i)
      out << "Subspace average measured signs:\t[";
      for (int i = 0; i < n_subspaces; ++i) {
        if (ss_n_saved[i] == 0) {
          out << "N/A";
        } else {
          out << (ss_sign_samples[i] / static_cast<double>(ss_n_saved[i]));
        }
        if (i != n_subspaces - 1) {
          out << " ";
        }
      }
      out << "]" << std::endl;
      // Print the configuration weight
      out << "\nCurrent weight:\t" << weight_curr << std::endl;
      // Print the current overall acceptance/update ratios (excluding warmup
      // steps, and including unmeasured steps)
      out << "Current total acceptance ratio:\t" << (n_accepted / n_meas_so_far) << std::endl;
      out << "Current average ascend ratio:\t" << (n_ascend / n_meas_so_far) << std::endl;
      out << "Current average descend ratio:\t" << (n_descend / n_meas_so_far) << std::endl;
      out << "Current average mutate ratio:\t" << (n_mutate / n_meas_so_far) << std::endl;
    }
    if (work_finished) {
      out << "\nNormalization constant: C / N_{s,0} =" << norm_const << std::endl;
    }
  }
  // Instantiates the Markov chain in the dummy subspace V_0; since the weight
  // D_0 is exactly known, we do not need to seed the coordinates in the setup
  // stage! Also seeds the random number generator(s) to be used in the
  // Metropolis step and updates.
  constexpr void setup() { weight_curr = d0_weight; }
  // Instantiates the Markov chain at some user-supplied phase space coordinate,
  // and determines the associated initial configuration weight. Also seeds the
  // random number generator(s) to be used in the Metropolis step and updates.
  constexpr void setup(const int idx_ss_init, const hc_lat_st_coords &initial_coords) {
    // If the initial subspace is the trivial one, call the appropriate setup
    // method
    if (idx_ss_init == 0) {
      setup();
    } else {
      // Update proposed coordinates and calculate the corresponding weight
      idx_ss_prop = idx_ss_init;
      coords_prop = initial_coords;
      weight_prop = get_weight_prop();  // also updates ss_diag_weights_prop
      // The initial configuration weight is automatically accepted;
      // fill the current variables
      idx_ss_curr = idx_ss_prop;
      coords_curr = coords_prop;
      weight_curr = weight_prop;
      if (using_local_wf) {
        ss_diag_weights_curr = ss_diag_weights_prop;
      }
    }
  }
  // Seed new spacetime proposal coordinates for vertices in the exclusive range
  // [v_start, v_stop)
  int seed_coords_prop(int v_start, int v_stop) {
    // Initialize temporary variables to hold coordinate seed values
    int v_seed_id;
    double v_seed_itime;
    std::vector<int> v_seed_posn(params.dim);
    // We shift to an origin-centered binomial distribution: B(N, 1/2) - N/2
    int lat_offset = -params.n_site_pd / 2;
    // Seed the vertices
    int n_seeded = 0;
    // Seed the center-of-mass coordinate (default constructor with N and beta
    // defined from params)
    if (v_start == 0) {
      coords_prop.push_back(hc_lat_st_coord(params.dim, params.n_site_pd, params.beta,
                                            params.delta_tau, params.lat_const, debug));
      // Start at the next vertex for the usual seeding procedure to follow
      ++v_start;
      ++n_seeded;
    }
    for (int i = v_start; i < v_stop; ++i) {
      // Seed spacetime variables for new unconstrained vertices
      if (constraints[i] == -1) {
        if (debug) {
          std::cout << "Seeding vertex v_" << i << "..." << std::endl;
        }
        v_seed_id = i;
        v_seed_itime = params.beta * std_uniform(rand_gen);
        for (int j = 0; j < params.dim; ++j) {
          v_seed_posn[j] = lat_binomial(rand_gen);
          proposal_ratio /= boost::math::pdf(lat_binomial_dist, v_seed_posn[j]);
          v_seed_posn[j] = pymod<int>(v_seed_posn[j] + lat_offset, params.n_site_pd);
        }
        // Construct the v_seed coordinate object and add it to the list
        hc_lat_st_coord v_seed(v_seed_id, v_seed_itime, v_seed_posn, params.n_site_pd, params.beta,
                               params.delta_tau, params.lat_const, debug);
        coords_prop.push_back(v_seed);
        ++n_seeded;
      }
      // Due to the equal time and position constraints imposed by the Hubbard
      // interaction, constrained vertices differ only from their partners by ID#
      else {
        if (debug) {
          std::cout << "Setting constrained vertex v_" << i << " from v_" << constraints[i] << "..."
                    << std::endl;
        }
        hc_lat_st_coord v_constr = coords_prop[constraints[i]];
        v_constr.id = i;
        coords_prop.push_back(v_constr);
      }
    }
    return n_seeded;
  }
  // Propose to move to the next order in the list of subspaces
  void ascend() {
    if (debug) {
      std::cout << "Trying ascend update..." << std::endl;
    }
    // The proposal subspace is one higher in the list
    idx_ss_prop = idx_ss_curr + 1;
    // Get the number of new interaction lines (i.e., vertex pairs)
    const int n_new_lines = subspaces[idx_ss_prop] - subspaces[idx_ss_curr];
    const int n_verts_curr = 2 * subspaces[idx_ss_curr];
    const int n_new_verts = 2 * n_new_lines;
    // There is one extra time for outgoing vertex of polarization diagrams
    const int n_new_times = n_new_lines + (diag_type == 1);
    // The proposal ratio is the product of all inverse generation probabilities
    // of the new variables; for the temporal part of the distribution, the
    // factor of -1 accounts for the fixed COM coordinate
    proposal_ratio = std::pow(params.beta, n_new_times - 1);
    // Seed the new internal vertices (range(i) defined s.t. new ids start at
    // n_verts_curr), and update the proposal ratio to include the distribution
    // of each spatial variable
    int n_seeded = 0;
    coords_prop = coords_curr;
    n_seeded += seed_coords_prop(n_verts_curr, n_verts_curr + n_new_verts);
    if (debug) {
      std::cout << "n_seeded = " << n_seeded << std::endl;
      std::cout << "n_new_times = " << n_new_times << std::endl;
    }
    assert(n_seeded == n_new_times);
    // Update the proposal diagram weight
    weight_prop = get_weight_prop();
    if (debug) {
      std::cout << "weight_prop = " << weight_prop << std::endl;
    }
  }
  // Propose to move to the previous order in the list of subspaces
  void descend() {
    if (debug) {
      std::cout << "Trying descend update..." << std::endl;
    }
    // The proposal subspace is one lower in the list
    idx_ss_prop = idx_ss_curr - 1;
    // Get the number of removed interaction lines (i.e., vertex pairs)
    const int n_rem_lines = subspaces[idx_ss_curr] - subspaces[idx_ss_prop];
    const int n_verts_curr = 2 * subspaces[idx_ss_curr];
    const int n_verts_prop = 2 * subspaces[idx_ss_prop];
    const int n_rem_verts = 2 * n_rem_lines;
    // There is one extra time for outgoing vertex of polarization diagrams
    const int n_rem_times = n_rem_lines + (diag_type == 1);
    // We shift to an origin-centered binomial distribution: B(N, 1/2) - N/2
    int lat_offset = -params.n_site_pd / 2;
    int this_lat_posn;
    // The proposal ratio is the product of all generation probabilities of the
    // variables to be removed; for the temporal part of the distribution, the
    // factor of -1 accounts for the fixed COM coordinate
    proposal_ratio = std::pow(1.0 / params.beta, n_rem_times - 1);
    for (int i = n_verts_curr - n_rem_verts; i < n_verts_curr; ++i) {
      // Only the independent (non-COM and unconstrained)
      // vertices contribute a non-unity proposal distribution
      if ((i == 0) || (constraints[i] != -1)) {
        continue;
      }
      for (int j = 0; j < params.dim; ++j) {
        // std::cout << "current coordinate index: " << i << std::endl;
        this_lat_posn = coords_curr[i].posn[j];
        // Shift back to a binomial distribution centered at N / 2 to compute the pmf correctly
        if (this_lat_posn >= params.n_site_pd / 2.0) {
          this_lat_posn -= params.n_site_pd;
        }
        if (debug && !(((this_lat_posn - lat_offset) >= 0) &&
                       ((this_lat_posn - lat_offset) < params.n_site_pd))) {
          std::cout << "Current lattice position: " << (this_lat_posn - lat_offset) << std::endl;
          throw std::out_of_range("Bad lattice position (not in [0, N))!");
        }
        proposal_ratio *= boost::math::pdf(lat_binomial_dist, this_lat_posn - lat_offset);
      }
    }
    // Truncate the list of spacetime coordinates to the number of vertices in
    // the proposed order
    coords_prop = coords_curr;
    coords_prop.resize(n_verts_prop);
    // Update the proposal diagram weight
    weight_prop = get_weight_prop();
    if (debug) {
      std::cout << "weight_prop = " << weight_prop << std::endl;
    }
  }
  // Propose to update one of the diagram internal variable(s) (while preserving
  // the subspace)
  void mutate() {
    if (debug) {
      std::cout << "Trying mutate update..." << std::endl;
    }
    // The proposed subspace is the same as the current one
    idx_ss_prop = idx_ss_curr;
    coords_prop = coords_curr;
    int n_times = ss_diags[idx_ss_prop].n_times;
    int n_posns = ss_diags[idx_ss_prop].n_posns;
    // Select internal variable(s) to alter; the total number of
    // modifiables is: (d+1)*n, where n is the subspace order
    int n_modifiables = (params.dim + 1) * modifiables.size();
    // Adjust the select_modifiable distribution parameters
    select_modifiable.param(DUnif_gen::param_type(0, n_modifiables - 1));
    // Alter the selected diagram variable(s), taking care of equal time and
    // position constraints
    int selection = select_modifiable(rand_gen);
    int v_modif;
    // Reseed the selected time (clean update); this leaves the state probability unchanged
    if (selection < n_times) {
      // Get selected modifiable vertex ID
      v_modif = modifiables[selection];
      // Update selected vertex's time with a local shift
      coords_prop[v_modif].itime += (params.beta / 10.0) * (2 * std_uniform(rand_gen) - 1);
      coords_prop[v_modif].itime = pymod<double>(coords_prop[v_modif].itime, params.beta);
      // Enforce the equal-time constraint on mate(v_{modif}) if applicable
      if (mates[v_modif] != -1) {
        coords_prop[mates[v_modif]].itime = coords_prop[v_modif].itime;
      }
      // Since the times are generated uniformly, the proposal ratio is 1
      proposal_ratio = 1.0;
    }
    // Move the selected position index to a nearest neighbor
    else {
      // Shift the selection so that it indexes the total number of spacetime vertex components;
      // hence, the integer division (selection // dim) indexes the spacetime vertices, and the
      // remainder (selection % dim) indexes the vertex components.
      selection = selection - n_times;
      // Get selected modifiable vertex ID and index for position component (axis)
      v_modif = modifiables[selection / params.dim];
      int i_axis = selection % params.dim;
      if (debug) {
        std::cout << "selection = " << selection << std::endl;
        std::cout << "v_modif = " << v_modif << std::endl;
        std::cout << "i_axis = " << i_axis << std::endl;
      }
      // Now, update the selected vertex's position allowing for
      // local position shifts up to 3 nn away (improves ergodicity)
      int posn_offset = static_cast<int>(std::floor((posn_shift_gen.max() + 1) / 2));
      int posn_shift = posn_shift_gen(rand_gen) - posn_offset;
      if (debug) {
        std::cout << "Trying position shift = " << posn_shift << std::endl;
      }
      coords_prop[v_modif].posn[i_axis] += posn_shift;
      // Enforce the lattice periodicity, n_i \in [0, N)
      coords_prop[v_modif].posn[i_axis] =
          pymod<int>(coords_prop[v_modif].posn[i_axis], params.n_site_pd);
      // Enforce the equal-position constraint on mate(v_{modif}) if applicable
      if (mates[v_modif] != -1) {
        coords_prop[mates[v_modif]].posn[i_axis] = coords_prop[v_modif].posn[i_axis];
      }
      // Since the proposal probabilities are symmetric
      // for a local update, the proposal ratio is 1
      proposal_ratio = 1.0;
    }
    // Update the proposal diagram weight
    weight_prop = get_weight_prop(v_modif);
    if (debug) {
      std::cout << "weight_prop = " << weight_prop << std::endl;
    }
  }
  // Measurement function; aggregates data related to the observable(s) of interest
  void measure() {
    if (debug) {
      std::cout << "Measuring observable(s)..." << std::endl;
    }
    // Get the current diagram sign; we store it as a double so that
    // multiplication with complex numbers is defined (not so for int)
    double sign_curr = sgn(weight_curr);
    // Get the current subspace index, and
    // update the subspace/total saved counts
    ss_sign_samples[idx_ss_curr] += sign_curr;
    ++ss_n_saved[idx_ss_curr];
    ++n_saved;
    // The scalar diagram is sign-definite, so just increment
    // the measurement sum (i.e., visitation count) in V_0
    if (subspaces[idx_ss_curr] == 0) {
      meas_sums[idx_ss_curr][0] += 1.0;  // Only one sum in the list
      return;
    }
    // For vacuum diagrams, simply increment by the current diagram sign (no FT
    // factors)
    if (diag_type == 0) {
      meas_sums[idx_ss_curr][0] += sign_curr;  // Only one sum in the list
    }
    // NOTE: Assumes inversion symmetry holds (hence exp(-ipr) -> cos(pr)),
    //       which need not be true for a general lattice problem...
    else {
      double p_dot_r;
      double nu_tau;
      // Defines the imaginary unit i
      const std::complex<double> ii(0.0, 1.0);
      // The current outgoing external spacetime coordinate; we perform the
      // Fourier transform to the Matsubara representation by MC, i.e., we
      // sample it; note that by convention, for any correlation function, the
      // second vertex in the list (v = 1) is the outgoing leg.
      const hc_lat_st_coord &v_out_rt = coords_curr[n_ext - 1];
      // Increment each measurement sum with the current diagram sign and associated FT factor
      for (std::size_t j = 0; j < meas_sums[idx_ss_curr].size(); ++j) {
        // The current measurement coordinate for the
        // external vertex in the Matsubara representation
        const hc_lat_mf_coord &v_out_mf_j = mf_meas_coords[j];
        // The FT factor is: e^{-i (p * r - nu * tau)}
        p_dot_r = 0.0;
        nu_tau = v_out_mf_j.imfreq * v_out_rt.itime;
        // for (std::size_t k = 0; k < params.dim; ++k) {
        //   p_dot_r += v_out_mf_j.mom[k] * v_out_rt.posn[k];
        // }
        // p_dot_r *= (2.0 * M_PI / static_cast<double>(params.n_site_pd));
        // Since the stored vectors mom and posn are index vectors,
        // include the missing scale factor (2 pi a / L) = (2 pi / N)
        p_dot_r = (2.0 * M_PI / static_cast<double>(params.n_site_pd)) *
                  std::inner_product(v_out_mf_j.mom.begin(), v_out_mf_j.mom.end(),
                                     v_out_rt.posn.begin(), 0.0);
        // Explicitly enforce the time-reversal symmetry (TRS) for
        // charge (and longitudinal spin) polarization measurements
        if (diag_type == 1) {
          meas_sums[idx_ss_curr][j] += sign_curr * std::cos(p_dot_r) * std::cos(nu_tau);
        }
        // The self-energy measurement is complex, as it is not TRS
        else {
          meas_sums[idx_ss_curr][j] += sign_curr * std::cos(p_dot_r) * std::exp(ii * nu_tau);
        }
      }
    }
  }
  // Propose a new MCMC state by randomly selecting an update (ascend, descend, or mutate)
  int propose() {
    if (debug) {
      std::cout << "\nProposing an update..." << std::endl;
    }
    // First, pick a random update from the following types: (0) ascend, (1) descend, (2) mutate
    int selection = roll_1d3(rand_gen);
    // Ascend update was selected
    if (selection == 0) {
      // If we are in the top subspace, we cannot ascend
      if (idx_ss_curr == n_subspaces - 1) {
        // Update not applicable, so the proposal weight is zero.
        // We set the proposal distribution to 1 to avoid undefined
        // behavior or division by zero in the acceptance ratio calculation
        weight_prop = 0.0;
        proposal_ratio = 1.0;
        if (debug) {
          std::cout << "Can't ascend!" << std::endl;
        }
      }
      // Generate the proposal configuration, as well as its probability and weight
      else {
        ascend();
      }
    }
    // Descend update was selected
    else if (selection == 1) {
      // If we are in the bottom subspace, we cannot descend
      if (idx_ss_curr == 0) {
        // Update not applicable, so the proposal weight is zero
        weight_prop = 0.0;
        proposal_ratio = 1.0;
        if (debug) {
          std::cout << "Can't descend!" << std::endl;
        }
      }
      // Generate the proposal configuration, as well as its probability and weight
      else {
        descend();
      }
    }
    // Mutate update selected
    else {
      // If we are in the trivial subspace V_0 (scalar configuration), there is nothing to mutate
      if (subspaces[idx_ss_curr] == 0) {
        // Update not applicable, so the proposal weight is zero
        weight_prop = 0.0;
        proposal_ratio = 1.0;
        if (debug) {
          std::cout << "Can't mutate!" << std::endl;
        }
      }
      // Generate the proposal configuration, as well as its probability and weight
      else {
        mutate();
      }
    }
    return selection;
  }
  // Blank Metropolis step for debug purposes
  constexpr void blank_step() { ++i_step; }
  // Do a single Metropolis step; we propose a new configuration,
  // then either accept or reject the move probabalistically.
  void step() {
    // Now, propose an update, then either accept or reject it according to the
    // Metropolis weight
    int selection = propose();
    double r = std_uniform(rand_gen);
    double acceptance_ratio = std::min(1.0, std::abs(weight_prop / weight_curr) * proposal_ratio);
    if (debug) {
      std::cout << "proposal_ratio = " << proposal_ratio << std::endl;
    }
    if (r <= acceptance_ratio) {
      if (debug) {
        std::cout << "Move accepted (p_accept = " << acceptance_ratio << ")" << std::endl;
      }
      // Update the current subspace, coordinates, weight,
      // probability of generation, and acceptance count
      idx_ss_curr = idx_ss_prop;
      coords_curr = coords_prop;
      weight_curr = weight_prop;
      if (using_local_wf) {
        // Update individual subspace diagram weights
        ss_diag_weights_curr = ss_diag_weights_prop;
      }
      // Only update the acceptance count after warmup steps
      if (i_step >= params.n_warm) {
        ++n_accepted;
        // For adjusting d0 value / checking detailed balance
        if (selection == 0) {
          ++n_ascend;
        } else if (selection == 1) {
          ++n_descend;
        } else {
          ++n_mutate;
        }
      }
    } else if (debug) {
      std::cout << "Move rejected (p_accept = " << acceptance_ratio << ")" << std::endl;
    }
    // After the warm-up period, perform a measurement every n_skip steps
    if ((i_step >= params.n_warm) && ((i_step - params.n_warm + 1) % params.n_skip == 0)) {
      measure();
    }
    ++i_step;
  }
  // Do the full MCMC walk
  meas_t integrate(std::string job_id = "", std::string outfile = "", bool normalize = true,
                   bool save_serial = false, bool main_thread = true) {
    setup();
    while (i_step < params.n_warm + params.n_meas) {
      // Do a Metropolis-Hastings step
      step();
      // Save MCMC state information for every millionth measurement to logfile,
      // and print it to stdout if the verbosity is high
      if (main_thread && (i_step >= params.n_warm) && (i_step % 1000000 == 0)) {
        summarize(job_id, outfile);  // print to logfile
        if (verbose) {
          summarize();  // print to stdout
        }
      }
    }
    assert(i_step == params.n_warm + params.n_meas);
    // Normalize the measurement data and mark the integration as finished
    const meas_t &meas_data = results(normalize);
    work_finished = true;
    // Optionally save the results of this thread to H5
    if (save_serial) {
      save(job_id, outfile);
    }
    // Summarize integration results to logfile in the main thread only
    if (main_thread) {
      summarize(job_id, outfile);  // print to logfile
      summarize();                 // print to stdout
    }
    return meas_data;
  }
  // Calculates the configuration proposal weight for the appropriate diagram type
  double get_weight_prop() {
    if (idx_ss_prop == 0) {
      return d0_weight;
    } else {
      return dn_weight();
    }
  }
  // Calculates the configuration proposal weight for mutation-type updates
  double get_weight_prop(int v_mutated) {
    if (idx_ss_prop == 0) {
      return d0_weight;
    } else if (!using_local_wf) {
      return dn_weight();
    } else {
      return dn_weight_mutate(v_mutated);
    }
  }
  // Proposal weight function for diagrams in the Hubbard model
  double dn_weight() {
    // Initialize temporary variables to hold position / time differences;
    // the vector del_n = (del_n_i, del_n_j) indexes G^{ij}_0(tau) in each
    // spatial direction along the lattice
    double tot_dn_weight = 0.0;
    // Add the fermionic connections to the weight for every diagram
    const diagram_pool_el &diags = ss_diags[idx_ss_prop];
    // Constant prefactors
    double prefactor = (2 * diags.s_ferm + 1) * std::pow(-params.U_loc, diags.n_intn) *
                       std::pow(params.lat_const, params.dim * diags.n_posns);
    // Get a reference to the bosonic edge list(s) (same for all diagrams!)
    for (int i_diag = 0; i_diag < diags.n_diags; ++i_diag) {
      // std::cout << "\nDiagram #" << i_diag << ":" << std::endl;
      // Initialize this diagram, including the fermion loop factor (-1) ^ F
      double this_weight = std::pow(-1.0, diags.n_loops[i_diag]);
      // Get a reference to the fermionic edge list for this diagram
      const edge_list &ferm_edges = diags.graphs.f_edge_lists[i_diag];
      // Loop over fermionic edges {(v_out, v_in)},
      // v_out (= v_start) = edge[0], v_in (= v_end) = edge[1]
      for (edge_t edge : ferm_edges) {
        // Evaluate the Green's function interpolant over
        // the spacetime distance associated with this line
        double del_tau;
        std::vector<int> del_nr(params.dim);
        std::tie(del_nr, del_tau) = coords_prop[edge[1]] - coords_prop[edge[0]];
        this_weight *= lat_g0_r_tau(del_nr[0], del_nr[1]).ap_eval(del_tau);
      }
      if (using_local_wf) {
        // Record the full weight of this diagram, with prefactors
        ss_diag_weights_prop[idx_ss_prop][i_diag] = prefactor * this_weight;
      }
      // Add to the total integrand weight
      tot_dn_weight += this_weight;
    }
    // Finally, return the result (with additional constant prefactors and
    // Feynman rules), and multiply by the Hubbard interaction term
    return prefactor * tot_dn_weight;
  }
  double dn_weight_mutate(int v_modif) {
    std::vector<int> mutated_verts = {v_modif, mates[v_modif]};
    // Remove mate vertex from the list if we didn't select a pair,
    // i.e., if it is nonexistent (a -1 in mate vector)
    if (mates[v_modif] == -1) {
      mutated_verts.resize(1);
    }
    // Build the weight for each diagram in the pool
    const diagram_pool_el &diags = ss_diags[idx_ss_prop];
    // Calculate 3-vertex at base v_modif for curr & prop st-coordinates
    double tot_dn_weight = 0.0;
    for (int i_diag = 0; i_diag < diags.n_diags; ++i_diag) {
      std::array<double, 2> vert_weights = {0.0, 0.0};  // {vweight_curr, vweight_prop}
      // Get a reference to the two fermionic edges
      edge_list f_edges;
      for (int v_base : mutated_verts) {
        const edge_t &f_edge_in = diags.nn_vertices.f_edge_in_lists[i_diag][v_base];
        const edge_t &f_edge_out = diags.nn_vertices.f_edge_out_lists[i_diag][v_base];
        f_edges.push_back(f_edge_in);
        f_edges.push_back(f_edge_out);
      }
      // Calculate st differences for incoming/outgoing fermion lines
      const std::array<hc_lat_st_coords, 2> &coords{coords_curr, coords_prop};
      for (std::size_t i = 0; i < 2; ++i) {  // (coords.size() = 2)
        double ferm_io_weight = 1.0;
        for (edge_t f_edge : f_edges) {
          double del_tau;
          std::vector<int> del_nr(params.dim);
          // Get the spacetime difference (using lattice spacing units)
          std::tie(del_nr, del_tau) = coords[i][f_edge[1]] - coords[i][f_edge[0]];
          // Evaluate the Green's function interpolant over
          // the spacetime distance associated with this line
          ferm_io_weight *= lat_g0_r_tau(del_nr[0], del_nr[1]).ap_eval(del_tau);
        }
        // Fermionic part of the 3-vertex weight
        vert_weights[i] = ferm_io_weight;
      }
      // NOTE: constant prefactors including the bosonic contributions U cancel in ratio
      double this_weight =
          (vert_weights[1] / vert_weights[0]) * ss_diag_weights_curr[idx_ss_curr][i_diag];
      // Record the full weight of this diagram
      ss_diag_weights_prop[idx_ss_prop][i_diag] = this_weight;
      // Add to the total integrand weight
      tot_dn_weight += this_weight;
    }
    return tot_dn_weight;
  }
};
