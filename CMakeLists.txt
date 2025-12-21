cmake_minimum_required(VERSION 3.14)
project(classifier LANGUAGES CXX)

#Cpp standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#pybind11 config
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(pybind11_DIR /usr/include/pybind11)
find_package(pybind11 REQUIRED)

#Eigen
find_package(Eigen3 REQUIRED)

#Source
set(SOURCES src/knn.cpp src/bindings.cpp)

pybind11_add_module(_classifier ${SOURCES})

target_include_directories(_classifier PRIVATE include)

#Other libraries
target_link_libraries(_classifier PRIVATE Eigen3::Eigen)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_options(_classifier PRIVATE -03)