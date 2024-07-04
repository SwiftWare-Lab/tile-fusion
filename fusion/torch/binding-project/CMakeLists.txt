cmake_minimum_required( VERSION 3.20)


project(gcn_fusion_binding CXX C )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++17  -march=native")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib/)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
find_package( OpenMP REQUIRED)
if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
endif()

set(SYM_SW_GCN ${CMAKE_CURRENT_LIST_DIR})
file(GLOB SYM_SW_GCN_SRC
        "${SYM_SW_GCN}/src/*.cpp"
)


add_library(sw_gcn SHARED
        main.cpp
)
target_sources(sw_gcn PRIVATE
        ${SYM_SW_GCN_SRC}
)
target_include_directories(sw_gcn PRIVATE
        ${SYM_SW_GCN}/include/
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
        ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(sw_gcn PRIVATE
        ${TORCH_LIBRARIES}
        Python::Python
        OpenMP::OpenMP_CXX
        $<LINK_ONLY:MKL::MKL>
)