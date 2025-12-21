#pragma once

#include "classifier.hpp"

namespace classifier_std {


class KNNClassifier : public Classifier {    
private:
    int k_;
    Matrix X_train_;
    Vector y_train_;
    bool fitted_;

    double euclideanDistance(
        const Eigen::VectorXd& a,
        const Eigen::VectorXd& b
    )

public:

    explicit KNNClassifier(int k = 5);

    void fit(const Matrix& X, const Vector& y) override;

    Vector predict(const Matrix& X) const override;

    int predict_one(const Eigen::VectorXd& x) const override;

    bool is_fitted() const override;
};

}