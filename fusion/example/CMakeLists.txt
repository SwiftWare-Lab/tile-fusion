


#if(NOT TARGET MKL::MKL)
#    set(MKL_INTERFACE "lp64")
#    find_package(MKL CONFIG REQUIRED)
#    add_definitions( "-DMKL")
#endif()


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
        )
add_executable(spmm_spmm_example
        SpMM_SpMM_Tutorial_Demo.cpp
        )
        
target_link_libraries(spmm_spmm_example
        fusion
        swbench::swbench
        OpenMP::OpenMP_CXX
        )

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