
project(cuda-spmm-impl LANGUAGES  C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_ARCHITECTURES 50 61 75 80 86)
#set(CMAKE_CUDA_ARCHITECTURES 50 75)
# TODO: after you had cmake 3.24, you can use the below line so no need to enter the version
#set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES native)

if (CUDA_FOUND AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    enable_language(CUDA)

else()
    message(STATUS "CUDA not found, proceeding without CUDA support")
    # Set the project languages to only CXX if CUDA is not found
endif()

add_executable(spmm_demo_gpu
        SpMMCudaDemo.cu
        )

set_target_properties(spmm_demo_gpu PROPERTIES CUDA_COMPILE_FLAGS "${CUDA_COMPILE_FLAGS}"
        )

target_include_directories(spmm_demo_gpu
        PRIVATE
        ${SYM_AGG}/include
        ${SYM_FUS}/include
        )
target_link_libraries(spmm_demo_gpu
        cusparse
        cublas
        CUDA::cusparse
        fusion
        aggregation
        swbench::swbench
        )

#if(OpenMP_CXX_FOUND)
#    target_link_libraries(spmv_demo_gpu OpenMP::OpenMP_CXX)
#endif()