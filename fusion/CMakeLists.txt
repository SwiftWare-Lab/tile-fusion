cmake_minimum_required( VERSION 3.20)




project(sparse_fusion CXX C )

add_subdirectory(aggregation)

add_subdirectory(benchmark)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib/)

set(SYM_AGG ${CMAKE_CURRENT_LIST_DIR}/aggregation/)
set(SYM_BENCH ${CMAKE_CURRENT_LIST_DIR}/benchmark/)
set(SYM_FUS ${CMAKE_CURRENT_LIST_DIR})

list(APPEND CMAKE_MODULE_PATH
        ${SYM_AGG}/cmake
        "${SYM_AGG}/cmake/third_party"
        )
find_package( OpenMP REQUIRED)


include(metis)
if(METIS_FOUND)
    add_definitions(-DMETIS)
    message(STATUS "METIS found")
endif()
add_definitions(-DMETIS)

include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

if(NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG OPTIONAL_COMPONENTS)
endif()

add_definitions(-DMKL)
message(STATUS "${MKL_IMPORTED_TARGETS}")

file(GLOB SYM_FUS_SRC
        "${SYM_FUS}/src/*.cpp"
        )
add_library(fusion
        ${SYM_FUS_SRC}

        )
target_include_directories(fusion PRIVATE
        ${SYM_AGG}/include/
        ${SYM_BENCH}/include/
        ${SYM_FUS}/include/
        $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
        )
target_link_libraries(fusion PRIVATE
        aggregation
        OpenMP::OpenMP_CXX
        argparse
        $<LINK_ONLY:MKL::MKL>
        )



add_subdirectory(example)