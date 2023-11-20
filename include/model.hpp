#ifndef MODEL_INTERFACE
#define MODEL_INTERFACE

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <memory>

/* 
 * Interface / abstract class for a generic machine learning model.
 * Defines the fit, errors and clone methods.
 */
class ModelInterface {
public:
    // fits the model to the data
    virtual void fit(const Eigen::MatrixXf & data) {};

    // computes the error for every row of the data and returns the results as a
    // vector where entry i corresponds to the error of the prediction of row i
    virtual Eigen::VectorXf errors(Eigen::MatrixXf const& data) = 0;

    // creates an unique pointer to another instance of this object
    // it is not supposed to copy learned parameters
    virtual std::unique_ptr<ModelInterface> clone() const = 0;
    virtual ~ModelInterface() = default;
};

/*
 * Class implementing a linear model based on the least squares method.
 */
class LinearModel : public ModelInterface {
public:
    LinearModel(int dim_in, int dim_out);
    std::unique_ptr<ModelInterface> clone() const override;
    ~LinearModel() = default;
    
    void fit(const Eigen::MatrixXf & data) override;
    Eigen::VectorXf errors(Eigen::MatrixXf const& data) override;

private:
    int dim_in, dim_out;
    Eigen::MatrixXf coefficients;
};

#endif
