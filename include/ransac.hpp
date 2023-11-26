#ifndef RANSAC
#define RANSAC

#include "model.hpp"

/*
 * The RANSAC class. Contains a pointer to the model and to the data.
 * It performs a fixed number of iterations, each iteration it trains a model
 * with num_samples samples. For a row to be considered an inlier, its error
 * must be smaller than tolerance. For a model to be valid, it must have at
 * least min_inliers inliers.
 */
class EstimatorRANSAC {
  public:
    explicit EstimatorRANSAC(std::unique_ptr<ModelInterface> model,
                    std::unique_ptr<Eigen::MatrixXf> data, int num_iter,
                    int num_samples, float tolerance, int min_inliers);

    ~EstimatorRANSAC() = default;

    // fits the best model using the paramters passed in the introduction.
    void fit();
    // returns the inliers of the data
    Eigen::MatrixXf getInliers();

    // returns the outliers of the data
    Eigen::MatrixXf getOutliers();

  private:
    std::unique_ptr<ModelInterface> model;
    std::unique_ptr<Eigen::MatrixXf> data;
    int num_iter;
    int num_samples;
    float tolerance;
    float min_inliers;

    float best_error;
    std::vector<int> best_inlier_indices, best_outlier_indices;
    std::unique_ptr<ModelInterface> best_model;
};

#endif
