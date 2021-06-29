# diagmc-hubbard-2dsqlat
Low-order diagrammatic Monte-Carlo (DiagMC) for the Hubbard model on the 2D square lattice.


<!-- DEPENDENCIES -->
## Dependencies

### 1. HDF5 >= 1.12

Earlier versions of HDF5 (e.g. 1.10) may work as well, but are untested. The code was tested against HDF5 1.12 built from scratch (following the documentation [here](https://portal.hdfgroup.org/display/support/HDF5+1.12.0)). Alternatively, one could try installing prebuilt binaries, e.g., on Ubuntu:
   ```sh
    sudo apt-get install libhdf5-serial-dev
   ```
To use the pre- and post-processing scripts, the [Python HDF5 library](https://docs.h5py.org/en/stable/quick.html) is also required:
   ```sh
    pip install h5py
   ```
### 2. Boost [Math](https://www.boost.org/doc/libs/1_76_0/libs/math/doc/html/index.html) and [Random](https://www.boost.org/doc/libs/1_76_0/doc/html/boost_random.html) Libraries

Either install the libraries individually via
   ```sh
    sudo apt-get install libboost-math-dev libboost-random-dev
   ```
or install Boost in its entirety,
   ```sh
    sudo apt-get install libboost-all-dev
   ```
### 3. (Optional) Open MPI (or equivalent)
Open MPI is packaged with most Linux distros nowadays; you can check that it is installed via, e.g., the following:
   ```sh
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
   ```
To link `h5c++` against an alternative MPI compiler (e.g., Intel or MPICH2), you will need to modify the [CXX](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/e00ea5a7d17f2076fe5889a23ca7152f3c5846d3/build/Makefile#L9) and [linker](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/e00ea5a7d17f2076fe5889a23ca7152f3c5846d3/build/Makefile#L10) flags in the Makefile accordingly.

### 4. (Optional) [TRIQS TPRF](https://triqs.github.io/tprf/latest/index.html#)

The TRIQS TPRF package is optionally used for benchmarking purposes in the post-processing script 'plot.py' (if [plot_rpa = True](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/f6be749f6154e362187f15f14b11eaa1ded88616/plot.py#L569)). It may be installed (along with [TRIQS](https://triqs.github.io/triqs/latest/) itself) via
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
   cd diagmc-hubbard-2dsqlat && ./make.sh
   ```
For serial usage, remove the ```D_MPI``` preprocessor macro from the [Makefile](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/e00ea5a7d17f2076fe5889a23ca7152f3c5846d3/build/Makefile#L2).

<!-- USAGE -->
## Usage

To use the code, first edit the test input parameters in 'hub_2dsqlat_rt_mcmc.cpp' as desired. In principle, the MCMC integrator is compatible with free energy, self energy, and polarization measurements, but only the latter have been tested. The provided example calculates the charge polarization up to 2nd order in U, and optionally compares with the RPA result (obtained via the TRIQS TPRF package). Several example sets of propagators/results are provided, but in order to run the code for a different set of test parameters, one may need to generate new propagators (i.e., if the code complains that a compatible lattice Green's function was not found). To this end, use the script [generate_propagators.py](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/9b82d1568875d67482f1bc3a151dabcaa85454f4/generate_propagators.py). The usage details are accessible as follows:
   ```
python3 generate_propagators.py -h
Usage: generate_propagators.py [ options ]

Options:
  -h, --help            show this help message and exit
  --dim=DIM             Spatial dimension of the electron gas (default is 2);
                         allowed values: {2, 3}.
  --target_n0=TARGET_N0
                         Target density in units of the lattice constant; since
                         the number of electrons is coarse-grained, the actual
                         density may differ slightly. Default (n0 = 1)
                         corresponds to half-filling (mu = 0).
  --target_mu0=TARGET_MU0
                         Target (noninteracting) chemical potential. If
                         supplied, we work at fixed chemical potential and
                         variable density; otherwise, we use a fixed density
                         and variable chemical potential.
  --t_hop=T_HOP         The tight-binding hopping parameter t.
  --U_loc=U_LOC         Onsite Hubbard interaction in units of 1/t.
  --beta=BETA           Inverse temperature in units of 1/t.
  --n_site_pd=N_SITE_PD
                         Number of sites per direction.
  --lat_const=LAT_CONST
                         Lattice constant, in Bohr radii (for working at fixed
                         'N' and 'a'; we will calculate 'V' on-the-fly).
  --lat_length=LAT_LENGTH
                         Lattice length, in Bohr radii (for working at fixed V;
                         we will calculate 'a' on-the-fly).
  --n_tau=N_TAU         Number of tau points in the nonuniform mesh used for
                         downsampling (an even number).
  --n_nu=N_NU           Number of bosonic frequency points (an even number).
  --save_dir=SAVE_DIR   Subdirectory to save results to, if applicable
  --save                Save propagator data to h5?
  --overwrite           Overwrite existing propagator data?
  --plot_g0             Option for plotting the lattice Green's functions.
  --plot_pi0            Option for plotting the polarization bubble P_0.
   ```
Then, rebuild the executable and run it:
   ```sh
    ./make.sh && ./hub_2dsqlat_rt_mcmc.exe
   ```
or for a parallel run, e.g. with 8 threads,
   ```sh
    ./make.sh && mpirun -n 8 ./hub_2dsqlat_rt_mcmc.exe
   ```
   
Reproducing all the figures provided (by setting [n_nu_meas = 5](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/9b82d1568875d67482f1bc3a151dabcaa85454f4/src/hub_2dsqlat_rt_mcmc.cpp#L25) and [batch_U = true](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/9b82d1568875d67482f1bc3a151dabcaa85454f4/src/hub_2dsqlat_rt_mcmc.cpp#L21)) will take a while (around half an hour).
For faster runs, try reducing n_meas by a factor of 10 (i.e., set [n_meas = 5000000](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/9b82d1568875d67482f1bc3a151dabcaa85454f4/src/hub_2dsqlat_rt_mcmc.cpp#L23)) and/or calculate only static susceptibilities by leaving [n_nu_meas = 1](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/9b82d1568875d67482f1bc3a151dabcaa85454f4/src/hub_2dsqlat_rt_mcmc.cpp#L25).

Finally, use [plot.py](https://github.com/dcerkoney/diagmc-hubbard-2dsqlat/blob/9b82d1568875d67482f1bc3a151dabcaa85454f4/plot.py) for postprocessing:
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
