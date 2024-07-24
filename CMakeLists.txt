cmake_minimum_required(VERSION 3.5...3.20)
project(cHySST VERSION 0.1.0 LANGUAGES C CXX)

find_package(Boost REQUIRED
             COMPONENTS serialization filesystem system program_options
             HINTS "${Boost_ROOT}")
include_directories(${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(examples)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

project(cHySST CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Enable all warnings, treat warnings as errors, compile with highest optimizaiton
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")