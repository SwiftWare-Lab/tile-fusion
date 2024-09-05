cmake_minimum_required( VERSION 3.20)


project(sparse_fusion CXX C)

if(PROFILING_WITH_PAPI)
    message(STATUS "Profiling with PAPI")
    add_definitions(-DPROF_WITH_PAPI)
endif()

add_subdirectory(aggregation)

add_subdirectory(benchmark)

if(PROFILING_WITH_PAPI)
    include_directories(
	${PAPI_INCLUDE_DIRS}
	)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(OpenMP_CXX_LIB_NAMES "libiomp5")
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
        )
target_link_libraries(fusion PRIVATE
        aggregation
        OpenMP::OpenMP_CXX
        argparse
        )

find_package(CUDA OPTIONAL_COMPONENTS)
find_package(CUDAToolkit OPTIONAL_COMPONENTS)


add_subdirectory(example)
add_subdirectory(gcn)
add_subdirectory(spmv-spmv)
add_subdirectory(gpu)