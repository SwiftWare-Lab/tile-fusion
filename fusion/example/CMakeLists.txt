
include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

project(examples)

if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
endif()

if(TARGET MKL::MKL)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    add_executable(fused_vs_mkl
            Fused_vs_MKL_Demo.cpp
            )
    target_include_directories(fused_vs_mkl PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            )
    target_link_libraries(fused_vs_mkl PRIVATE
            fusion
            $<LINK_ONLY:MKL::MKL>
            swbench::swbench
            OpenMP::OpenMP_CXX
            )

    add_executable(gcn_demo
            ../gcn/GNN_GCN_Demo.cpp
    )

    target_include_directories(gcn_demo PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )

    target_link_libraries(gcn_demo
            fusion
            swbench::swbench
            $<LINK_ONLY:MKL::MKL>
    )

    add_executable(gcn_layer_demo
            ../gcn/GCN_Layer_Demo.cpp
            )

    target_include_directories(gcn_layer_demo PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            )

    target_link_libraries(gcn_layer_demo
            fusion
            swbench::swbench
            $<LINK_ONLY:MKL::MKL>
            )


endif()

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

add_executable(fusion_profiler
        Fusion_Profiler.cpp
        )
target_link_libraries(fusion_profiler
        fusion
        swbench::swbench
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