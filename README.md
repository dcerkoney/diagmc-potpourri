![banner](https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/banner.png?raw=true)

# diagmc-potpourri
A Markov chain Monte Carlo library for various low-order Feynman diagram computations.

![chi_2_ch](https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/chi_2_ch_2dsqhub_run_1629109813.png?raw=true)

<p align="middle">
  <img src="https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/static_chi_ch_2dsqhub_run_1629109813.png?raw=true" width="270" />
  <img src="https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/re_sigma_2_2dsqhub_run_1629106445.png?raw=true" width="270" /> 
  <img src="https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/sigma_2_iom0_2dsqhub_run_1629106445.png?raw=true" width="270" />
</p>

<p align="middle">
  <img src="https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/im_sigma_2_loc_2dsqhub_run_1629107326.png?raw=true" width="270" /> 
  <img src="https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/re_sigma_2_loc_2dsqhub_run_1629107326.png?raw=true" width="270" />
  <img src="https://github.com/dcerkoney/diagmc-potpourri/blob/e1538d14651915a346723ca66f72bc97c08ed54c/.readme/sigma_2_iom0_2dsqhub_run_1629107326.png?raw=true" width="270" />
</p>

<!-- DEPENDENCIES -->
## Dependencies

### 1. HDF5 >= 1.12

Earlier versions of HDF5 (e.g. 1.10) may work as well, but are untested. The code was tested against HDF5 1.12 built from scratch (following the documentation [here](https://portal.hdfgroup.org/display/support/HDF5+1.12.0)). Alternatively, one could try installing prebuilt binaries, e.g.,
   ```sh
    sudo apt-get install libhdf5-serial-dev
   ```
To use the pre- and post-processing scripts, the [Python HDF5 library](https://docs.h5py.org/en/stable/quick.html) is also required:
   ```sh
    pip install h5py
   ```
   
### 2. Boost [math](https://www.boost.org/doc/libs/1_76_0/libs/math/doc/html/index.html), [random](https://www.boost.org/doc/libs/1_76_0/doc/html/boost_random.html), and (optional requirement) [filesystem](https://www.boost.org/doc/libs/1_75_0/libs/filesystem/doc/index.htm) Libraries

Either install the libraries individually via
   ```sh
    sudo apt-get install libboost-math-dev libboost-random-dev
   ```
or install Boost in its entirety,
   ```sh
    sudo apt-get install libboost-all-dev
   ```
For backwards compatibility with some very old compilers (e.g., GCC < 5.3), the boost filesystem libraries are an additional requirement,
   ```sh
    sudo apt-get install libboost-system-dev libboost-filesystem-dev
   ```
  
### 3. [ruamel.yaml](https://pypi.org/project/ruamel.yaml/)
Used for YAML config files with round-trip comment support.
   ```sh
    pip3 install ruamel.yaml 
   ```

### 4. (Optional) [jsbeautifier](https://pypi.org/project/jsbeautifier/)
Used in pre/post-processing to improve human readability of JSON info & config files, if available.
   ```sh
    pip install jsbeautifier
   ```

### 5. (Optional) Open MPI or equivalent
Open MPI is packaged with most Linux distros nowadays; you can check that it is installed via, e.g., the following:
   ```sh
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
   ```
To link against a specific MPI implementation (e.g., Intel or MPICH2), you may need to update your `PATH` and `LD_LIBRARY_PATH` environment variables accordingly and/or set the appropriate include directories explicitly in [CMakeLists.txt](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/main/src/CMakeLists.txt) such that CMake chooses the correct libraries. For more details, see CMake's implementation of [FindMPI](https://github.com/Kitware/CMake/blob/master/Modules/FindMPI.cmake).

### 6. (Optional) [TRIQS TPRF](https://triqs.github.io/tprf/latest/)

The TRIQS TPRF package is optionally used for benchmarking purposes in the post-processing script 'plot.py' if [`plot_rpa = True`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/f3e4401425171b1bb93c1f6e5d1de278f95d482e/plot.py#L854). It may be installed (along with [TRIQS](https://triqs.github.io/triqs/latest/) itself) via the TRIQS repository,
   ```sh
    sudo apt-get update && sudo apt-get install -y software-properties-common apt-transport-https curl
    source /etc/lsb-release
    curl -L https://users.flatironinstitute.org/~ccq/triqs3/$DISTRIB_CODENAME/public.gpg | sudo apt-key add -
    sudo add-apt-repository "deb https://users.flatironinstitute.org/~ccq/triqs3/$DISTRIB_CODENAME/ /"
    sudo apt-get install triqs_tprf
   ```
For detailed installation instructions, see [here](https://triqs.github.io/triqs/latest/install.html) and [here](https://triqs.github.io/tprf/latest/install.html).


<!-- INSTALLATION -->
## Installation

### 1. Clone the repo:
   ```sh
   git clone https://github.com/dcerkoney/diagmc-potpourri.git
   ```
   
### 2. Navigate to your local project directory and build the executable:
   ```sh
   cd diagmc-potpourri && mkdir build
   (cd build && cmake ../src -DCMAKE_BUILD_TYPE=Release && make -j <NTHREAD>)
   ```
where `<NTHREAD>` is the number of threads to be used for the build step.

<!-- USAGE -->
## Usage

To use the code, first copy the configuration templates for either the charge polarization or self-energy example measurement into `config.yml` and `graph_info.json` in the project directory, e.g.,
   ```sh
cp config_templates/chi_ch_example_config.yml config.yml
cp config_templates/chi_ch_example_graph_info.json graph_info.json
   ```   
edit the input parameters in [config.yml](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/tree/main/config.yml) as desired, and then run the helper script:
  ```sh
   python3 run_diagmc.py <CMDLINE_ARGS>
  ```
The usage details are accessible as follows:
   ```
python3 run_diagmc.py -h
Usage: run_diagmc.py [ options ]

Options:
  -h, --help            show this help message and exit
  --target_mu=TARGET_MU
                         Target (noninteracting) chemical potential. If
                         supplied, we work at fixed chemical potential and
                         variable density; otherwise, we use a fixed density
                         and variable chemical potential.
  --target_n0=TARGET_N0
                         Target density in units of the lattice constant; since
                         the number of electrons is coarse-grained, the actual
                         density may differ slightly. Default (n0 = 1)
                         corresponds to half-filling (mu0 ~= 0).
  --lat_length=LAT_LENGTH
                         Lattice length in Bohr radii for working at fixed V.
                         If supplied, the lattice constant is deduced from the
                         lattice volume and number of sites per direction.
  --lat_const=LAT_CONST
                         lattice constant in Bohr radii
  --n_site_pd=N_SITE_PD
                         number of lattice sites per direction
  --n_tau=N_TAU         number of tau points in the nonuniform mesh used for
                         downsampling (an even number)
  --n_nu=N_NU           number of bosonic frequency points in the uniform FFT
                         mesh (an even number)
  --dim=DIM             spatial dimension of the lattice (default is 2);
                         allowed values: {2, 3}
  --beta=BETA           inverse temperature in inverse Hartrees
  --t_hop=T_HOP         tight-binding hopping parameter t
  --U_loc=U_LOC         onsite Hubbard interaction in Hartrees
  --config=CONFIG       relative path of the config file to be used (default:
                         'config.yml')
  --propr_save_dir=PROPR_SAVE_DIR   
                         subdirectory to save results to, if applicable
  --plot_g0             generate plots for the lattice Green's function
  --plot_pi0            generate plots for the polarization bubble
  --dry_run             perform a dry run (don't update config file or save
                         propagator data)
   ```
If any applicable propagator data is found (stored by default in propagators/proprs_<JOB_ID>), it will be selected for the run; otherwise, the propagators will be generated by calling the script [generate_propagators.py](generate_propagators.py). Additionally, the script (re)calculates a number of parameters and updates the YAML (user-facing) and JSON (for internal use by the C++ driver) config files. Accordingly, several config parameters need not be specified initially (namely, those with a blank value in the provided [example config templates](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/tree/main/config_templates)). A config option with an explicit `null` value is an optional parameter which may be set by the user. Additionally, several config parameters may optionally be overriden by command line arguments (see above) for convenience.

Several example sets of propagators/results are provided, but in order to run the code for a different set of test parameters, one may need to generate new propagators (i.e., if the C++ MCMC driver complains that a compatible lattice Green's function was not found). The MCMC integrator is compatible with free energy, self energy, and charge/longitudinal spin susceptibility measurements. The provided examples calculate the charge susceptibility and self energy up to 2nd order in U. The charge susceptibility may optionally be compared with the RPA result obtained via the TRIQS TPRF package.

Running both example measurements at the default settings should take no more than a minute or two.
However, reproducing all the figures provided (e.g., for the susceptibilities, by increasing to [`n_meas = 50000000`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/7e51c5410c38abfa65dcaaac425d35e9e3e6a565/config.yml#L26), [`n_nu_meas = 5`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/7e51c5410c38abfa65dcaaac425d35e9e3e6a565/config.yml#L29), and setting [`use_batch_U = true`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/7e51c5410c38abfa65dcaaac425d35e9e3e6a565/config.yml#L24)) will take more time—around half an hour on a typical machine.

Finally, use the [plot.py](plot.py) script for postprocessing:
* To generate plots for all run subdirectories,
   ```sh
    python3 plot.py
   ```
* To generate plots for a specific run subdirectory,
   ```sh
    python3 plot.py RUN_SUBDIR
   ```
* To generate plots for the most recent run,
   ```sh
    python3 plot.py latest
   ```


<!-- CONTACT -->
## Contact

Daniel Cerkoney - dcerkoney@physics.rutgers.edu

Project Link: [https://github.com/dcerkoney/diagmc-potpourri](https://github.com/dcerkoney/diagmc-potpourri)
