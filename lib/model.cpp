#include "model.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

using Eigen::MatrixXf;

LinearModel::LinearModel(int dim_in, int dim_out) {
    this -> dim_in = dim_in;
    this -> dim_out = dim_out;

    coefficients = MatrixXf(dim_out, dim_in+1);
    coefficients.setOnes();
}

Eigen::VectorXf SolveLeastSquares(Eigen::MatrixXf input_data, const Eigen::VectorXf &output) {
    input_data.conservativeResize(input_data.rows(), input_data.cols() + 1);
    input_data.col(input_data.cols()-1).setOnes();

    return input_data.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(output);
}

float SquareError(Eigen::VectorXf vec1, Eigen::VectorXf vec2) {
    return (vec1 - vec2).squaredNorm();
}


void LinearModel::Fit(Eigen::MatrixXf &data) {
    for (int i=0; i < dim_out; i++) {
        Eigen::VectorXf row = SolveLeastSquares(data.block(0, 0, data.rows(), dim_in), data.col(dim_in + i));
        coefficients.row(i) << row.transpose();
    }
}

Eigen::VectorXf LinearModel::Errors(Eigen::MatrixXf &data) {
    Eigen::MatrixXf input_data = data.block(0, 0, data.rows(), dim_in); 
    Eigen::MatrixXf output_data = data.block(0, dim_in, data.rows(), dim_out); 


    input_data.conservativeResize(data.rows(), dim_in + 1);
    input_data.col(dim_in).setOnes();

    Eigen::MatrixXf predicted = input_data * coefficients.transpose();

    Eigen::VectorXf error(data.rows());
    error.setOnes();

    for (int i=0; i < data.rows(); i++) {
        error.row(i) << SquareError(predicted.row(i), output_data.row(i));
    }

    return error;
}


std::unique_ptr<ModelInterface> LinearModel::Clone() const {
    return std::make_unique<LinearModel>(dim_in, dim_out);
}
