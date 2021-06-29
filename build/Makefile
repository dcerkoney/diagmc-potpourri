C++ = h5c++
CFLAGS = -D_MPI -O3 -std=c++17 -fopenmp #-fPIC
BOOST = -lboost_system -lboost_filesystem

BUILDDIR=.
SRCDIR=../src

# Link to both Open MPI and HDF5 libraries
hub_2dsqlat_rt_mcmc : export HDF5_CXX = mpicxx
hub_2dsqlat_rt_mcmc : export HDF5_CLINKER = mpicxx
hub_2dsqlat_rt_mcmc : $(SRCDIR)/hub_2dsqlat_rt_mcmc.cpp
	$(C++) $(CFLAGS) $(SRCDIR)/hub_2dsqlat_rt_mcmc.cpp -o $(BUILDDIR)/hub_2dsqlat_rt_mcmc
	
clean:
	rm -f hub_2dsqlat_rt_mcmc hub_2dsqlat_rt_mcmc.o

