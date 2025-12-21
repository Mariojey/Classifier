#pragma once

#include "types.hpp"

namespace classifier_std{

class Classifier {

public:

    virtual ~Classifier() = default;

    virtual void fit(const Matrix& X, const Vector& y) = 0;

    virtual Vector predict(const Matrix& X) const = 0;

    virtual int predict_one(const Eigen::VectorXd& x) const = 0;

    virtual bool is_fitted() const = 0;
}

}