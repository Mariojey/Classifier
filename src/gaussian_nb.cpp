#include <iostream>
#include <cmath>
#include <unordered_map>

#include "gaussian_nb.hpp"

using namespace std;

namespace classifier_std {

GaussianNG::GaussianNB()
    : nClasses_(0), nFeatures_(0) {}

void GaussianNB::fit(

    const Eigen::MatrixXd &X,
    const Eigen::VectorXi &y
) {
    const int nSamples = X.rows();

    nFeatures_ = X.cols();

    vector<int> classes;
    unordered_map<int, int> classToIndex;

    for(int i = 0; i < y.size(); ++i){

        if(classToIndex.find(y[i]) == classToIndex.end()) {

            int index = classes.size();

            classes.push_back(y[i]);

            classToIndex[y[i]] = index;

        }

    }

    nClasses_ = classes.size();

    means_ = Eigen::MatrixXd::Zero(nClasses_, nFeatures_);
    variances_ = Eigen::MatrixXd::Zero(nClasses_, nFeatures_);
    classPriors_ = Eigen::VectorXd::Zero(nClasses_);

    vector<int> quantityOfSample(nClasses_, 0);

    for(int i = 0; i < nSamples; ++i){
        
        int class__ = classToIndex[y[i]];
        means_.row(class__) += X.row(i);
        quantityOfSample[class__]++;

    }

    for (int cl = 0; cl < nClasses_; ++cl){
        means_.row(cl) /= quantityOfSample[cl];
        classPriors_[cl] = static_cast<double>(quantityOfSample[cl]) / nSamples;
    }

    for(int idx = 0; idx < nSamples; ++idx)
    {
        int cl = classToIndex[y[i]];

        Eigen::VectorXd diff = X.row(i).transpose() - means_.row(cl).transpose();

        variances_.row(cl) += diff.array().square().matrix().transpose();
    }

    for (int cl = 0; cl < nClasses_; ++cl) {
        variances_.row(cl) /= counts[cl];
        variances_.row(cl).array() += eps;
    }

}

}