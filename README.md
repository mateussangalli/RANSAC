# RANSAC
Simple implementation of the RANSAC based on the Eigen library.

Dependencies: Eigen and matplot++

Used [vcpkg](https://github.com/microsoft/vcpkg) to manage the dependencies.
The dependencies can be installed with

`./vcpkg install Eigen`
and
`./vcpkg install matplotplusplus`

The project can be built with `cd build && cmake .. && make`.

To run the example with the linear model run ExampleLinear in the build directory.
