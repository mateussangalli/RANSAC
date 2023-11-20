#include "ransac.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <random>

// Samples num_samples rows from the data.
Eigen::MatrixXf SampleRows(const Eigen::MatrixXf & data, int num_samples) {
    std::vector<int> row_indices(data.rows());
    std::iota(row_indices.begin(), row_indices.end(), 0); // Fill with 0, 1, 2, ..., numRows-1
    std::shuffle(row_indices.begin(), row_indices.end(), std::mt19937(std::random_device()()));


    Eigen::MatrixXf out(num_samples, data.cols());
    for (int i = 0; i < num_samples; i++) {
        out.row(i) << data.row(row_indices[i]);
    }

    return out;
}

// Returns a new matrix by concatenating the rows at the given indices.
Eigen::MatrixXf gatherRows(const Eigen::MatrixXf & data, std::vector<int> indices) {
    Eigen::MatrixXf out(indices.size(), data.cols());
    for (int i=0; i<indices.size(); i++) {
        out.row(i) << data.row(indices[i]);
    }

    return out;
}

EstimatorRANSAC::EstimatorRANSAC(std::unique_ptr<ModelInterface> & model,
                                 std::unique_ptr<Eigen::MatrixXf> & data,
                                 int num_iter,
                                 int num_samples,
                                 float tolerance,
                                 int min_inliers) {
    this->model.swap(model);
    this->data.swap(data);
    this->num_iter = num_iter;
    this->tolerance = tolerance;
    this->min_inliers = min_inliers;
    this->num_samples = num_samples;

    best_error = 99999.f;
}

void EstimatorRANSAC::fit() {
    for (int i=0; i < num_iter; i++) {

        auto maybe_inliers = SampleRows(*data, num_samples);
        model->fit(maybe_inliers);

        auto errors = model->errors(*data);


        int current_index = 0;
        std::vector<int> inlier_indices, outlier_indices;
        for (auto error_value: errors.rowwise()) {
            float value = error_value.data()[0];

            if(value < tolerance) {
                inlier_indices.push_back(current_index);
            } else {
                outlier_indices.push_back(current_index);
            }
            current_index += 1;
        }


        if(inlier_indices.size() < min_inliers) { continue; }

        auto inliers = gatherRows(*data, inlier_indices);

        model->fit(inliers);

        float mean_error = model->errors(inliers).sum();

        if(mean_error < best_error) {
            best_error = mean_error;
            best_model.swap(model);
            model = best_model->clone();
            best_inlier_indices = inlier_indices;
            best_outlier_indices = outlier_indices;
        }
    }
}

Eigen::MatrixXf EstimatorRANSAC::getInliers() {
    return gatherRows(*data, best_inlier_indices);
}

Eigen::MatrixXf EstimatorRANSAC::getOutliers() {
    return gatherRows(*data, best_outlier_indices);
}
