
include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

project(gcn)



# add a definition to the project

if(${SWIFT_BLAS_BACKEND} STREQUAL "MKL")
    if(NOT TARGET MKL::MKL)
        set(MKL_INTERFACE "lp64")
        find_package(MKL CONFIG OPTIONAL_COMPONENTS)
    endif()
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    set( BLAS_INC_DIR
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    set( BLAS_LINK_FLAGS
            $<LINK_ONLY:MKL::MKL>)
elseif (${SWIFT_BLAS_BACKEND} STREQUAL "FLEXIBLAS")
    add_definitions(-DFLEXIBLAS)
    message(STATUS "FLEXIBLAS")
    set( BLAS_LINK_FLAGS
            flexiblas )
endif ()
    if(TARGET MKL::MKL)
        add_definitions(-DMKL)
        message(STATUS "${MKL_IMPORTED_TARGETS}")

        add_executable(gcn_demo
                MultiLayer/GCN_Multi_Layer_Demo.cpp
        )

        target_include_directories(gcn_demo PRIVATE
                ${BLAS_INC_DIR}
        )

        target_link_libraries(gcn_demo
                fusion
                swbench::swbench
                OpenMP::OpenMP_CXX
                #$<LINK_ONLY:MKL::MKL>
                ${BLAS_LINK_FLAGS}
        )

        add_executable(gcn_layer_demo
                SingleLayer/GCN_Layer_Demo.cpp
        )

        target_include_directories(gcn_layer_demo PRIVATE
                #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
                ${BLAS_INC_DIR}
        )

        target_link_libraries(gcn_layer_demo
                fusion
                swbench::swbench
                OpenMP::OpenMP_CXX
                #$<LINK_ONLY:MKL::MKL>
                ${BLAS_LINK_FLAGS}
        )

        add_executable(gcn_layer_sp_demo
                SingleLayer/GCN_Layer_SP_Demo.cpp
        )

        target_include_directories(gcn_layer_sp_demo PRIVATE
                #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
                ${BLAS_INC_DIR}
        )

        target_link_libraries(gcn_layer_sp_demo
                fusion
                swbench::swbench
                OpenMP::OpenMP_CXX
                #$<LINK_ONLY:MKL::MKL>
                ${BLAS_LINK_FLAGS}
        )

        add_executable(gcn_verify_demo
                GCN_Verify_With_PyG_Demo.cpp
        )

        target_include_directories(gcn_verify_demo PRIVATE
                #$<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
                ${BLAS_INC_DIR}
        )

        target_link_libraries(gcn_verify_demo
                fusion
                swbench::swbench
                OpenMP::OpenMP_CXX
                #$<LINK_ONLY:MKL::MKL>
                ${BLAS_LINK_FLAGS}
        )
    else ()
        add_executable(gcn_demo
                MultiLayer/GCN_Multi_Layer_Demo.cpp
                SingleLayer/GCN_Single_Layer_Demo_Utils.h
                GCN_Verify_With_PyG_Demo.cpp
        )

        target_link_libraries(gcn_demo
                fusion
                swbench::swbench
        )

    #    add_executable(gcn_layer_demo
    #            GCN_Layer_Demo.cpp
    #    )
    #
    #    target_link_libraries(gcn_layer_demo
    #            fusion
    #            swbench::swbench)
    endif()
