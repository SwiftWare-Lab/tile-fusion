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
endif()