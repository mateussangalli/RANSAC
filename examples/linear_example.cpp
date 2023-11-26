#include "model.hpp"
#include "ransac.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <matplot/core/figure_registry.h>
#include <matplot/matplot.h>
#include <random>

using namespace std;

static const int NUM_POINTS = 100;
static const float OUTLIER_PROBABILITY = 0.1;
static const float OUTLIER_DISTANCE = 1.f;

static const int NUM_ITER = 100;
static const int NUM_SAMPLES = 5;
static const int MIN_INLIER = 50;
static const float TOLERANCE = 0.001;

// Creates a line dataset with coefficient 2 and with some outliers on parallel lines at a distance 1.
// Will plot the inliers in blue and the outliers in red.
std::unique_ptr<Eigen::MatrixXf> makeLine() {
    std::mt19937 rng;
    auto data = make_unique<Eigen::MatrixXf>(100, 2);
    data->col(0).setRandom();
    data->col(1) << (data->col(0) * 2);

    std::uniform_real_distribution<float> unif(0, 1);
    for (int i=0; i<data->rows(); i++) {
        float r = unif(rng);
        if(r < OUTLIER_PROBABILITY / 2) {
            data->row(i).col(1) << data->row(i).col(1).value() + OUTLIER_DISTANCE;
        } else if (r < OUTLIER_PROBABILITY) {
            data->row(i).col(1) << data->row(i).col(1).value() - OUTLIER_DISTANCE;
        }
    }

    return data;

}

// Creates a vector<double> from a column matrix.
vector<double> toDoubleVector(Eigen::MatrixXf data) {
    vector<double> out;
    for (int i=0; i<data.rows(); i++) {
        double tmp = (double)data.row(i).value();
        out.push_back(tmp);
    }
    return out;
}

// Displays the inliers and outliers
void plotResults(Eigen::MatrixXf inliers, Eigen::MatrixXf outliers) {
    using namespace matplot;
    auto f = figure(false);

    auto points_x = toDoubleVector(inliers.col(0));
    auto points_y = toDoubleVector(inliers.col(1));

    
    auto outliers_x = toDoubleVector(outliers.col(0));
    auto outliers_y = toDoubleVector(outliers.col(1));

    auto ax = scatter(points_x, points_y);
    hold(on);
    scatter(outliers_x, outliers_y);

    matplot::legend({"Inliers", "Outliers"});

    show();

}

int main() {
    // define the data points
    auto data = makeLine();

    // define the base linear model
    std::unique_ptr<ModelInterface> model = make_unique<LinearModel>(1, 1);

    // define the RANSAC model
    EstimatorRANSAC estimator_ransac(std::move(model), std::move(data), NUM_ITER, NUM_SAMPLES, TOLERANCE, MIN_INLIER);
    estimator_ransac.fit();

    // get the inliers and outliers for displaying
    auto inliers = estimator_ransac.getInliers();
    auto outliers = estimator_ransac.getOutliers();

    plotResults(inliers, outliers);
}
