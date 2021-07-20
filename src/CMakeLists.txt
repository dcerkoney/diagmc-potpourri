cmake_minimum_required(VERSION 3.10)
project(diagmc-hubbard-2dsqlat VERSION 0.1.0)

# Request compilation with C++17 features
set(CMAKE_CXX_STANDARD 17)

# Output executables to parent directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

# Use -Ofast instead of -O3 optimizations in release mode
add_compile_options($<$<CONFIG:RELEASE>:-Ofast>)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add all project executables
add_executable(chi_ch_example hub_2dsqlat_rt_mcmc_chi_ch_example.cpp)
add_executable(self_en_example hub_2dsqlat_rt_mcmc_self_en_example.cpp)

# Require compilation with C++14 features, which we make heavy use of
target_compile_features(chi_ch_example PUBLIC cxx_std_14)
target_compile_features(self_en_example PUBLIC cxx_std_14)

# Add project include directory
target_include_directories(chi_ch_example PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_include_directories(self_en_example PRIVATE ${CMAKE_SOURCE_DIR}/../include)

# Require HDF5 and get its C++ bindings
find_package(HDF5 REQUIRED COMPONENTS C CXX)
target_link_libraries(chi_ch_example PUBLIC ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
target_link_libraries(self_en_example PUBLIC ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})

# Require MPI and get its C++ bindings
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    target_compile_definitions(chi_ch_example PUBLIC -DHAVE_MPI)
    target_compile_definitions(self_en_example PUBLIC -DHAVE_MPI)
else()
    message(SEND_ERROR "No MPI C++ library found!")
endif()
# Link MPI C++ libraries
target_link_libraries(chi_ch_example PRIVATE MPI::MPI_CXX)
target_link_libraries(self_en_example PRIVATE MPI::MPI_CXX)

# Check for filesystem capabilities and link to required libraries if necessary
try_compile(HAVE_STD_FS ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build_tests/test_filesystem.cpp)
if(HAVE_STD_FS)
    message(STATUS "Compiler supports filesystem library natively")
    target_compile_definitions(chi_ch_example PUBLIC -DHAVE_STD_FS)
    target_compile_definitions(self_en_example PUBLIC -DHAVE_STD_FS)
    # Must link to filesystem library explicitly for GCC < 9
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.0")
            message(STATUS "Linking to filesystem library explicitly (GCC < 9)")
            # Link filesystem libraries
            target_link_libraries(chi_ch_example PRIVATE stdc++fs)
            target_link_libraries(self_en_example PRIVATE stdc++fs)
        endif()
    endif()
else()
    try_compile(HAVE_EXPTL_FS ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build_tests/test_experimental_filesystem.cpp
                LINK_LIBRARIES stdc++fs)
    if(HAVE_EXPTL_FS)
        message(STATUS "Compiler supports experimental filesystem library, linking to it")
        target_compile_definitions(chi_ch_example PUBLIC -DHAVE_EXPTL_FS)
        target_compile_definitions(self_en_example PUBLIC -DHAVE_EXPTL_FS)
        # Link experimental filesystem libraries
        target_link_libraries(chi_ch_example PRIVATE stdc++fs)
        target_link_libraries(self_en_example PRIVATE stdc++fs)
    else()
        find_package(Boost COMPONENTS filesystem REQUIRED)
        try_compile(HAVE_BOOST_FS ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/build_tests/test_boost_filesystem.cpp
                    LINK_LIBRARIES Boost::filesystem)
        if(HAVE_BOOST_FS)
            message(STATUS "Compiler lacks filesystem support, linking to boost filesystem library")
            target_compile_definitions(chi_ch_example PUBLIC -DHAVE_BOOST_FS)
            target_compile_definitions(self_en_example PUBLIC -DHAVE_BOOST_FS)
            # Link boost filesystem libraries
            target_link_libraries(chi_ch_example PRIVATE Boost::filesystem)
            target_link_libraries(self_en_example PRIVATE Boost::filesystem)
        else()
            message(SEND_ERROR "No filesystem libraries found!")
        endif()
    endif()
endif()