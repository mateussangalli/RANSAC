cmake_minimum_required(VERSION 3.12)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

project(ransac
    VERSION 1.0
    DESCRIPTION "Simple implementation of the RANSAC algorithm for CPP based on the EIGEN library"
    LANGUAGES CXX
    )

include(vcpkg/scripts/buildsystems/vcpkg.cmake)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(EstimatorRANSAC lib/ransac.cpp)
add_library(Model lib/model.cpp)
add_executable(LinearExample examples/linear_example.cpp)
add_executable(TestLinearModel tests/test_linear_model.cpp)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(EstimatorRANSAC PRIVATE Eigen3::Eigen)
target_link_libraries(Model PRIVATE Eigen3::Eigen)
target_link_libraries(TestLinearModel Model Eigen3::Eigen)
