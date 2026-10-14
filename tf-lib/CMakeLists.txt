cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++17  -march=native")

project(tile-fusion-lib CXX C)


if(NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib)
endif()
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY lib)
endif()

SET(MKL_INTERFACE_FULL intel_lp64)
find_package(MKL REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH)
#if(NOT TARGET MKL::MKL)

#endif()
if(TARGET MKL::MKL)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
endif ()
Python_add_library(tf_lib MODULE WITH_SOABI
        main.cpp
        Functions.cpp
        TorchBindings.cpp
)

target_include_directories(tf_lib PRIVATE
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
        ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(tf_lib PRIVATE
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
        pybind11::headers
        Python::Module
        OpenMP::OpenMP_CXX
        $<LINK_ONLY:MKL::MKL>
)
