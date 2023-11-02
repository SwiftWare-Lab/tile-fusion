
include_directories(
        ${SYM_AGG}/include
        ${SYM_FUS}/include
)

project(gcn)

if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
endif()

if(TARGET MKL::MKL)
    add_definitions(-DMKL)
    message(STATUS "${MKL_IMPORTED_TARGETS}")

    add_executable(gcn_demo
            MultiLayer/GCN_Multi_Layer_Demo.cpp
    )

    target_include_directories(gcn_demo PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )

    target_link_libraries(gcn_demo
            fusion
            swbench::swbench
            OpenMP::OpenMP_CXX
            $<LINK_ONLY:MKL::MKL>
    )

    add_executable(gcn_layer_demo
            SingleLayer/GCN_Layer_Demo.cpp
    )

    target_include_directories(gcn_layer_demo PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
    )

    target_link_libraries(gcn_layer_demo
            fusion
            swbench::swbench
            OpenMP::OpenMP_CXX
            $<LINK_ONLY:MKL::MKL>
    )
else ()
    add_executable(gcn_demo
            MultiLayer/GCN_Multi_Layer_Demo.cpp
            SingleLayer/GCN_Single_Layer_Demo_Utils.h
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