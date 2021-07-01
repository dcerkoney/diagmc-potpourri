# diagmc-hubbard-2dsqlat
Low-order diagrammatic Monte-Carlo (DiagMC) for the Hubbard model on the 2D square lattice.

<p align="middle">
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/ae3898311cf53d8a9cbba1fedbe227ca32552141/results/self_en_examples/self_en_dyn_n=2_bHFI.png?raw=true" width="100" />
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/ae3898311cf53d8a9cbba1fedbe227ca32552141/results/chi_ch_examples/charge_poln_diags_n=2_bHFI.png?raw=true" width="728" /> 
</p>

![alt text](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/779b25f5414ce13ac169260bd66a82eafba893a1/results/chi_ch_examples/beta=10,%20U=1/chi_2_ch_2dsqhub_run_1624927311.png?raw=true)

<p align="middle">
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/779b25f5414ce13ac169260bd66a82eafba893a1/results/chi_ch_examples/beta=10,%20U=1/static_chi_ch_2dsqhub_run_1624927311.png?raw=true" width="276" />
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/779b25f5414ce13ac169260bd66a82eafba893a1/results/self_en_examples/beta=2,%20U=2/re_sigma_2_2dsqhub_run_1625109804.png?raw=true" width="276" /> 
  <img src="https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/779b25f5414ce13ac169260bd66a82eafba893a1/results/self_en_examples/beta=2,%20U=2/sigma_2_iom0_2dsqhub_run_1625109804.png?raw=true" width="276" />
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
To link `h5c++` against an alternative MPI compiler (e.g., Intel or MPICH2), you will need to modify the [CXX](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/e00ea5a7d17f2076fe5889a23ca7152f3c5846d3/build/Makefile#L9) and [linker](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/e00ea5a7d17f2076fe5889a23ca7152f3c5846d3/build/Makefile#L10) flags in the Makefile accordingly.

For serial usage, remove the [`D_MPI`](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/e00ea5a7d17f2076fe5889a23ca7152f3c5846d3/build/Makefile#L2) preprocessor macro from the Makefile.

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
### 2. Navigate to your local project directory, then build the executable:
   ```sh
   cd diagmc-hubbard-2dsqlat && make
   ```
   

<!-- USAGE -->
## Usage

To use the code, first edit the test input parameters in [hub_2dsqlat_rt_mcmc_chi_ch_example.cpp](src/hub_2dsqlat_rt_mcmc_chi_ch_example.cpp) and [hub_2dsqlat_rt_mcmc_self_en_example.cpp](src/hub_2dsqlat_rt_mcmc_self_en_example.cpp) as desired. The MCMC integrator is compatible with free energy, self energy, and charge/longitudinal spin susceptibility measurements. The provided examples calculate the charge susceptibility and self energy up to 2nd order in U. The charge susceptibility may optionally be compared with the RPA result obtained via the TRIQS TPRF package. Several example sets of propagators/results are provided, but in order to run the code for a different set of test parameters, one may need to generate new propagators (i.e., if the code complains that a compatible lattice Green's function was not found). To this end, use the script [generate_propagators.py](generate_propagators.py). The usage details are accessible as follows:
   ```
python3 generate_propagators.py -h
Usage: generate_propagators.py [ options ]

Options:
  -h, --help            show this help message and exit
  --dim=DIM             Spatial dimension of the lattice (default is 2);
                         allowed values: {2, 3}.
  --target_n0=TARGET_N0
                         Target density in units of the lattice constant; since
                         the number of electrons is coarse-grained, the actual
                         density may differ slightly. Default (n0 = 1)
                         corresponds to half-filling (mu0 ~= 0).
  --target_mu0=TARGET_MU0
                         Target (noninteracting) chemical potential. If
                         supplied, we work at fixed chemical potential and
                         variable density; otherwise, we use a fixed density
                         and variable chemical potential.
  --t_hop=T_HOP         The tight-binding hopping parameter t.
  --U_loc=U_LOC         Onsite Hubbard interaction in units of t.
  --beta=BETA           Inverse temperature in units of 1/t.
  --n_site_pd=N_SITE_PD
                         Number of sites per direction.
  --lat_const=LAT_CONST
                         Lattice constant in Bohr radii.
  --lat_length=LAT_LENGTH
                         Lattice length in Bohr radii (for working at fixed V:
                         calculate 'a' on-the-fly).
  --n_tau=N_TAU         Number of tau points in the nonuniform mesh used for
                         downsampling (an even number).
  --n_nu=N_NU           Number of bosonic frequency points (an even number).
  --save_dir=SAVE_DIR   Subdirectory to save results to, if applicable
  --save                Save propagator data to h5?
  --overwrite           Overwrite existing propagator data?
  --plot_g0             Option for plotting the lattice Green's functions.
  --plot_pi0            Option for plotting the polarization bubble P_0.
   ```
Then, rebuild the executables and run either of them:
   ```sh
    make && ./hub_2dsqlat_rt_mcmc_chi_ch_example.exe
   ```
or
   ```sh
    make && ./hub_2dsqlat_rt_mcmc_self_en_example.exe
   ```
For a parallel run with, say, 8 threads,
   ```sh
    make && mpirun -n 8 ./hub_2dsqlat_rt_mcmc_chi_ch_example.exe
   ```

Runing the examples at the default settings should take no more than a few minutes.
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
