
project(jacobi_demo)


include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
        ${SYM_FUS}/gcn
)

#if(NOT TARGET MKL::MKL)
#    set(MKL_INTERFACE "lp64")
#    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
#endif()

if(TARGET MKL::MKL)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    add_executable(jacobi_demo
            jacobi_demo.cpp
    )
    target_include_directories(jacobi_demo PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(jacobi_demo
            fusion
            swbench::swbench
            ${BLAS_LINK_FLAGS}
            OpenMP::OpenMP_CXX
    )

else()
    add_executable(jacobi_demo
            jacobi_demo.cpp
    )

    target_link_libraries(jacobi_demo
            fusion
            swbench::swbench
            OpenMP::OpenMP_CXX
    )
endif()