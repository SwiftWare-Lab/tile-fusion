
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")




if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
endif()

if(TARGET MKL::MKL)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    add_executable(example-app torchTest.cpp
            Torch_GCN_Layer_Utils.h
            Backend_Utils.h)
    target_link_libraries(example-app
            "${TORCH_LIBRARIES}"
            OpenMP::OpenMP_CXX
            $<LINK_ONLY:MKL::MKL>)
    target_include_directories(example-app PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )
    set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

endif ()
## The following code block is suggested to be used on Windows.
## According to https://github.com/pytorch/pytorch/issues/25457,
## the DLLs need to be copied to avoid memory errors.
#if (MSVC)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#    add_custom_command(TARGET example-app
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            ${TORCH_DLLS}
#            $<TARGET_FILE_DIR:example-app>)
#endif (MSVC)