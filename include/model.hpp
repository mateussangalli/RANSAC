#ifndef MODEL_INTERFACE
#define MODEL_INTERFACE

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <memory>

class ModelInterface {
public:
    virtual void Fit(Eigen::MatrixXf &data) {};
    /* virtual Eigen::VectorXf Errors(Eigen::MatrixXf &data); */

    virtual std::unique_ptr<ModelInterface> Clone() const = 0;
    virtual ~ModelInterface() = default;
};

class LinearModel : public ModelInterface {
public:
    LinearModel(int dim_in, int dim_out);
    std::unique_ptr<ModelInterface> Clone() const override;
    ~LinearModel() {};

    void Fit(Eigen::MatrixXf &data) override;
    Eigen::VectorXf Errors(Eigen::MatrixXf &data);

private:
    int dim_in, dim_out;
    Eigen::MatrixXf coefficients;
};

#endif
