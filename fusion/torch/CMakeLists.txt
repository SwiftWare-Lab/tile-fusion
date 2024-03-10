cmake_minimum_required(VERSION 3.22)
project(fused_gcn)

set(CMAKE_CXX_STANDARD 17)
#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES 61)
#endif()
find_package(Python OPTIONAL_COMPONENTS Development)
find_package(Torch OPTIONAL_COMPONENTS)
find_package( OpenMP REQUIRED)
include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
if(NOT TARGET MKL::MKL)

    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
endif()

if(TARGET MKL::MKL AND TARGET Python)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    add_executable(fused_gcn GCNIntegratedTraining.cpp)
    #    target_compile_features(example_app PRIVATE cxx_std_11)

    #    add_executable(example-app torchTest.cpp
    #            Torch_GCN_Layer_Utils.h
    #            Backend_Utils.h)
    target_link_libraries(fused_gcn PRIVATE
            ${TORCH_LIBRARIES}
            Python::Python
            OpenMP::OpenMP_CXX
            $<LINK_ONLY:MKL::MKL>
            swbench::swbench
            fusion)
    target_include_directories(fused_gcn PUBLIC
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_directories(fused_gcn PUBLIC
            ${SYM_AGG}/include
            ${SYM_FUS}/include
    )
#    add_executable(wrapper_bench TorchWrapperBenchmark.cpp)
#    target_link_libraries(wrapper_bench PRIVATE
#            ${TORCH_LIBRARIES}
#            Python::Python
#            OpenMP::OpenMP_CXX
#            $<LINK_ONLY:MKL::MKL>
#            swbench::swbench
#            fusion)
#    target_include_directories(wrapper_bench PUBLIC
#            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
#            ${TORCH_INCLUDE_DIRS}
#    )
#    target_compile_definitions(fused_gcn
#            PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    #    set_property(TARGET example_app PROPERTY CXX_STANDARD 11)

endif ()