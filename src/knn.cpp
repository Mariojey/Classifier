#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>

#include "knn.hpp"

using namespace std;

namespace classifier_std{

KNNClassifier::KNNClassifier(int k) : k_(k), fitted_(false)
{
    if(k_ <= 0){
        throw invalid_argument("K must be greather than 0");
    }
}

void KNNClassifier::fit(const Matrix& X, const Vector& y)
{
    if(X.rows() != y.size())
    {
        throw invalid_argument("X size and y size must be the same");
    }

    X_train_ = X;
    y_train_ = y;
    fitted_ = true;
}

Vector KNNClassifier::predict(const Matrix& X) const
{
    if (!fitted_)
    {
        throw runtime_error("Model not fitted");
    }

    Vector predictions(X.rows());

    for (int i = 0; i < X.rows(); ++i){

        predictions(i) = predict_one(X.row(i));
    
    }

    return predictions;
    
}

int KNNClassifier::predict_one(const Eigen::VectorXd& x) const
{
    if (!fitted_)
    {
        throw runtime_error("Model not fitted");
    }

    if (x.size() != X_train.cols())
    {
        throw invalid_argument("Incorrect input dimension");
    }

    vector<pair<double, int>> distances;
    distances.reserve(X_train.rows());

    for (int i = 0; i < X_train.rows(); ++i){

        double distance = euclidean_distance(x, X_train_.row(i));
        distances.emplace_back(distance, y_train_(i));
    }

    partial_sort(
        distances.begin(),
        distances.begin() + k_,
        distances.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        }
    )

    map<int, int> votes;

    for(int i = 0; i < k_; ++i){
        votes[distance[i].second]++;
    }

    int bestLabel = -1;
    int maxVotes = -1;

    for (const auto& [label, count]: votes) {
        if(count > maxVotes){
            maxVotes = count;
            bestLabel = label;
        }
    }

    return bestLabel;

}

bool KNNClassifier::is_fitted() const 
{
    return fitted_;
}

double KNNClassifier::euclidean_distance(
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b
) const{
    return (a - b).norm();
}

}