
if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG REQUIRED)
endif()


add_library(spmm_spmm_demo_utils STATIC
        SpMM_SpMM_Demo_Utils.cpp)

target_link_libraries(spmm_spmm_demo_utils
        fusion
        swbench::swbench
        MKL::MKL
        )

include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

project(examples)

add_executable(dag_decomp
        dag_decomposition_demo.cpp
        )
target_link_libraries(dag_decomp
        fusion
        )

add_executable(spmm_spmm_fusion
        SpMM_SpMM_Demo.cpp
        )
target_link_libraries(spmm_spmm_fusion
        fusion
        swbench::swbench
        OpenMP::OpenMP_CXX
        spmm_spmm_demo_utils
        )

add_executable(fused_vs_mkl
        Fused_vs_MKL_Demo.cpp
        )

target_link_libraries(fused_vs_mkl
        fusion
        swbench::swbench
        OpenMP::OpenMP_CXX
        spmm_spmm_demo_utils
        )


#if(TARGET MKL::MKL)
#    add_definitions(-DMKL)
#    message(STATUS "${MKL_IMPORTED_TARGETS}")
#    add_executable(dense_test
#            dense_test.cpp
#            )
#    target_include_directories(dense_test PRIVATE
#            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
#            sympiler/aggregation/include/
#            )
#    target_link_libraries(dense_test PRIVATE
#            $<LINK_ONLY:MKL::MKL>
#            aggregation
#            )
#
#    add_executable(sparse_mkl_test
#            sparse_mkl_test.cpp
#            )
#    target_include_directories(sparse_mkl_test PRIVATE
#            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
#            sympiler/aggregation/include/
#            )
#    target_link_libraries(sparse_mkl_test PRIVATE
#            aggregation
#            $<LINK_ONLY:MKL::MKL>
#            )

#endif()


#target_compile_options(dag_decomp PRIVATE
#        -march=native
#        )
#target_include_directories(psc_example PRIVATE
#        sympiler/include
#        sympiler/aggregation/include
#        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
#        )
#target_link_libraries(psc_example PRIVATE
#        aggregation
#        $<LINK_ONLY:MKL::MKL>
#        )