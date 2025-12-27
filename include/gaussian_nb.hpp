#pragma once

#include "classifier.hpp"

#include <Eigen/Dense>
#include <vector>

namespace classifier_std {

class GaussianNB : public Classifier {

private:

    int nClasses_;
    int nFeatures_;

    Eigen::VectorXd classPriors_;
    Eigen::MatrixXd means_;
    Eigen::MatrixXd variances_;

    static constexpr double eps = 1e-9;

public:
    GaussianNB();

    void fit(
        const Eigen::MatrixXd &X,
        const Eigen::VectorXi &y
    ) override;

    int predict_one(
        const Eigen::VectorXd &x
    ) const override;

};

}