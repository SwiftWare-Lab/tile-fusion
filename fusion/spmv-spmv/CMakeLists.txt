include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

project(spmv)

#if(NOT TARGET MKL::MKL)
#    set(MKL_INTERFACE "lp64")
#    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
#endif()

if(TARGET MKL::MKL)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    add_executable(spmv_spmv_demo
            SpMV_SpMV_Demo.cpp
    )
    target_include_directories(spmv_spmv_demo PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(spmv_spmv_demo PRIVATE
            fusion
            ${BLAS_LINK_FLAGS}
            swbench::swbench
            OpenMP::OpenMP_CXX
    )
endif()