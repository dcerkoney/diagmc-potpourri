cmake_minimum_required(VERSION 3.10)
project(diagmc-hubbard-2dsqlat VERSION 0.1.0)

# Request compilation with C++17 features
set(CMAKE_CXX_STANDARD 17)

# Output executables to parent directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

add_compile_options(-Wall -Wextra -Werror -pedantic)

# Use -Ofast (-O3 and -ffast-math) optimizations in release mode
add_compile_options($<$<CONFIG:RELEASE>:-Ofast>)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add all project executables
add_executable(random random.cpp)
add_executable(hub_2dsqlat_rt_mcmc_cf_meas hub_2dsqlat_rt_mcmc_cf_meas.cpp)
add_executable(mb_hub_2dsqlat_rt_mcmc_cf_meas mb_hub_2dsqlat_rt_mcmc_cf_meas.cpp)

# Require compilation with C++14 features, which we make heavy use of
# target_compile_features(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC cxx_std_14)
# target_compile_features(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC cxx_std_14)

# Require compilation with C++17 features (should we maintain backwards compatibility with C++14?)
target_compile_features(random PUBLIC cxx_std_11)
target_compile_features(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC cxx_std_17)
target_compile_features(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC cxx_std_17)


# Add project include directory
target_include_directories(random PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_include_directories(hub_2dsqlat_rt_mcmc_cf_meas PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_include_directories(mb_hub_2dsqlat_rt_mcmc_cf_meas PRIVATE ${CMAKE_SOURCE_DIR}/../include)

# Require HDF5 and get its main (low-level API) C/C++ bindings
find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_link_libraries(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC ${HDF5_LIBRARIES})
target_link_libraries(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC ${HDF5_LIBRARIES})
target_include_directories(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC ${HDF5_INCLUDE_DIRS})

# Find MPI libraries and C++ bindings
find_package(MPI)
if(MPI_CXX_FOUND)
    target_compile_definitions(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_MPI)
    target_compile_definitions(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_MPI)
    # Link MPI C++ libraries
    target_link_libraries(hub_2dsqlat_rt_mcmc_cf_meas PRIVATE MPI::MPI_CXX)
    target_link_libraries(mb_hub_2dsqlat_rt_mcmc_cf_meas PRIVATE MPI::MPI_CXX)
elseif(HDF5_IS_PARALLEL)
    message(SEND_ERROR "Parallel HDF5 requires linkage to MPI, but no MPI C++ library was found!")
else()
    message(STATUS "No MPI C++ library was found, compiling in serial mode")
endif()

# Check for filesystem capabilities and link to required libraries if necessary
try_compile(HAVE_STD_FS ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build_tests/test_filesystem.cpp)
if(HAVE_STD_FS)
    message(STATUS "Compiler supports filesystem library natively")
    target_compile_definitions(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_STD_FS)
    target_compile_definitions(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_STD_FS)
    # Must link to filesystem library explicitly for GCC < 9
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.0")
            message(STATUS "Linking to filesystem library explicitly (GCC < 9)")
            # Link filesystem libraries
            target_link_libraries(hub_2dsqlat_rt_mcmc_cf_meas PRIVATE stdc++fs)
            target_link_libraries(mb_hub_2dsqlat_rt_mcmc_cf_meas PRIVATE stdc++fs)
        endif()
    endif()
else()
    try_compile(HAVE_EXPTL_FS ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build_tests/test_experimental_filesystem.cpp
                LINK_LIBRARIES stdc++fs)
    if(HAVE_EXPTL_FS)
        message(STATUS "Compiler supports experimental filesystem library, linking to it")
        target_compile_definitions(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_EXPTL_FS)
        target_compile_definitions(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_EXPTL_FS)
        # Link experimental filesystem libraries
        target_link_libraries(hub_2dsqlat_rt_mcmc_cf_meas PRIVATE stdc++fs)
        target_link_libraries(mb_hub_2dsqlat_rt_mcmc_cf_meas PRIVATE stdc++fs)
    else()
        find_package(Boost COMPONENTS filesystem REQUIRED)
        try_compile(HAVE_BOOST_FS ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build_tests/test_boost_filesystem.cpp
                    LINK_LIBRARIES Boost::filesystem)
        if(HAVE_BOOST_FS)
            message(STATUS "Compiler lacks filesystem support, linking to boost filesystem library")
            target_compile_definitions(hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_BOOST_FS)
            target_compile_definitions(mb_hub_2dsqlat_rt_mcmc_cf_meas PUBLIC -DHAVE_BOOST_FS)
            # Link boost filesystem libraries
            target_link_libraries(hub_2dsqlat_rt_mcmc_cf_meas PRIVATE Boost::filesystem)
            target_link_libraries(mb_hub_2dsqlat_rt_mcmc_cf_meas PRIVATE Boost::filesystem)
        else()
            message(SEND_ERROR "No filesystem libraries found!")
        endif()
    endif()
endif()