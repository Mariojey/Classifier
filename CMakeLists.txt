cmake_minimum_required(VERSION 3.14)
project(classifier LANGUAGES CXX)

#Cpp standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#pybind11 config
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(extern/pybind11)

#Source
set(SOURCES src/knn.cpp src/bindings.cpp)

pybind11_add_module(_classifier ${SOURCES})

target_include_directories(_classifier PRIVATE include extern/eigen)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(_classifier PRIVATE -O3)