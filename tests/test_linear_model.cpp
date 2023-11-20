#include <iostream>
#include "model.hpp"
#include <Eigen/src/Core/Matrix.h>

using namespace std;

#define IS_TRUE(x) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; }


void test_fit() {
    Eigen::MatrixXf data(2, 2);
    data << 0, 1, 1, 2;

    LinearModel model(1, 1);
    
    model.fit(data);
    
    Eigen::MatrixXf test_data(3, 2);
    test_data << 0, 1, 1, 2, 3, 4;
    float sum_errors = model.errors(test_data).sum();
    
    IS_TRUE(sum_errors < 0.001f);
}

int main() {
    test_fit();
}
