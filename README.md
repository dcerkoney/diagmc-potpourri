# diagmc-hubbard-2dsqlat
Low-order diagrammatic Monte-Carlo (DiagMC) for the Hubbard model on the 2D square lattice in real space.

![alt text](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/chi_ch_examples/n=1,%20beta=10,%20U=1/chi_2_ch_2dsqhub_run_1624927311.png?raw=true)

<p align="middle">
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/chi_ch_examples/n=1,%20beta=10,%20U=1/static_chi_ch_2dsqhub_run_1624927311.png?raw=true" width="276" />
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/self_en_examples/n=1,%20beta=2,%20U=2/re_sigma_2_2dsqhub_run_1625109804.png?raw=true" width="276" /> 
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/self_en_examples/n=1,%20beta=2,%20U=2/sigma_2_iom0_2dsqhub_run_1625109804.png?raw=true" width="276" />
</p>

<p align="middle">
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/self_en_examples/n=0.3,%20beta=2,%20U=2/im_sigma_2_loc_2dsqhub_run_1625172983.png?raw=true" width="276" /> 
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/self_en_examples/n=0.3,%20beta=2,%20U=2/re_sigma_2_loc_2dsqhub_run_1625172983.png?raw=true" width="276" />
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/5cdef140969020ceec51c41c87d8479fb56c512b/results/self_en_examples/n=0.3,%20beta=2,%20U=2/sigma_2_iom0_2dsqhub_run_1625172983.png?raw=true" width="276" />
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
### 2. Boost [Math](https://www.boost.org/doc/libs/1_76_0/libs/math/doc/html/index.html) and [Random](https://www.boost.org/doc/libs/1_76_0/doc/html/boost_random.html) Libraries

Either install the libraries individually via0
   ```sh
    sudo apt-get install libboost-math-dev libboost-random-dev
   ```
or install Boost in its entirety,
   ```sh
    sudo apt-get install libboost-all-dev
   ```
### 3. (Optional) Open MPI or equivalent
Open MPI is packaged with most Linux distros nowadays; you can check that it is installed via, e.g., the following:
   ```sh
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
   ```
To link against a specific MPI implementation (e.g., Intel or MPICH2), you may need to update your `PATH` and `LD_LIBRARY_PATH` environment variables accordingly and/or set the appropriate include directories explicitly in [CMakeLists.txt](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/faeb4d796a64d23e2b2df93ce9a36d863556b61e/src/CMakeLists.txt) such that CMake chooses the correct libraries. For more details, see CMake's implementation of [FindMPI](https://github.com/Kitware/CMake/blob/master/Modules/FindMPI.cmake).

### 4. (Optional) [TRIQS TPRF](https://triqs.github.io/tprf/latest/)

The TRIQS TPRF package is optionally used for benchmarking purposes in the post-processing script 'plot.py' if [`plot_rpa = True`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/d8035967acc7e31e4fbbbb16093cc0b762f5004a/plot.py#L888). It may be installed (along with [TRIQS](https://triqs.github.io/triqs/latest/) itself) via
   ```sh
    sudo apt-get install triqs_tprf
   ```
For detailed installation instructions, see [here](https://triqs.github.io/tprf/latest/install.html) and [here](https://triqs.github.io/triqs/latest/install.html).


<!-- INSTALLATION -->
## Installation

### 1. Clone the repo:
   ```sh
   git clone https://github.com/dcerkoney/diagmc-hubbard-2dsqlat.git
   ```
### 2. Navigate to your local project directory and build the executable:
   ```sh
   cd diagmc-hubbard-2dsqlat && mkdir build
   (cd build && cmake ../src -DCMAKE_BUILD_TYPE=Release)
   ```


<!-- USAGE -->
## Usage

To use the code, first edit the test input parameters in [hub_2dsqlat_rt_mcmc_chi_ch_example.cpp](src/hub_2dsqlat_rt_mcmc_chi_ch_example.cpp) and/or [hub_2dsqlat_rt_mcmc_self_en_example.cpp](src/hub_2dsqlat_rt_mcmc_self_en_example.cpp) as desired. The MCMC integrator is compatible with free energy, self energy, and charge/longitudinal spin susceptibility measurements. The provided examples calculate the charge susceptibility and self energy up to 2nd order in U. The charge susceptibility may optionally be compared with the RPA result obtained via the TRIQS TPRF package. Several example sets of propagators/results are provided, but in order to run the code for a different set of test parameters, one may need to generate new propagators (i.e., if the code complains that a compatible lattice Green's function was not found). To this end, edit the [config.yml](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/faeb4d796a64d23e2b2df93ce9a36d863556b61e/config.yml) file as desired, and then run the script [generate_propagators.py](generate_propagators.py). Note that several config parameters are (re)calculated by the script, and need not be specified initially; namely, those with `null` value in the provided  [example config templates](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/tree/main/config_templates). Additionally, several config parameters may optionally be overriden by command line options for convenience; the usage details are accessible as follows:
   ```
python3 generate_propagators.py -h
Usage: generate_propagators.py [ options ]

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
  --save_dir=SAVE_DIR   subdirectory to save results to, if applicable
  --plot_g0             generate plots for the lattice Green's function
  --plot_pi0            generate plots for the polarization bubble
  --dry_run             perform a dry run (don't update config file or save
                        propagator data)
   ```
Then, rebuild the project
   ```sh
   (cd build && cmake ../src -DCMAKE_BUILD_TYPE=Release)
   ```
and run the executable of interest, e.g.:
   ```sh
    ${MPI_PREFIX} ./chi_ch_example
   ```
or
   ```sh
    ${MPI_PREFIX} ./self_en_example
   ```
where `${MPI_PREFIX}` would be unset for a serial run or, say, `mpirun -n 8` for a parallel run with 8 threads.

Running both example measurements at the default settings should take no more than a minute or two.
However, reproducing all the figures provided (e.g., for the susceptibilities, by increasing to [`n_meas = 50000000`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/d8035967acc7e31e4fbbbb16093cc0b762f5004a/src/hub_2dsqlat_rt_mcmc_chi_ch_example.cpp#L21), [`n_nu_meas = 5`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/d8035967acc7e31e4fbbbb16093cc0b762f5004a/src/hub_2dsqlat_rt_mcmc_chi_ch_example.cpp#L23), and setting [`batch_U = true`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/d8035967acc7e31e4fbbbb16093cc0b762f5004a/src/hub_2dsqlat_rt_mcmc_chi_ch_example.cpp#L19)) will take more time—around half an hour on a typical machine.

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

Project Link: [https://github.com/dcerkoney/diagmc-hubbard-2dsqlat](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat)
