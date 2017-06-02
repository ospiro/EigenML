//
// Created by Oliver Spiro on 5/24/17.
//
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#ifndef PCA_PCAREDUCER_H
#define PCA_PCAREDUCER_H


class PCAReducer {
private:
//public://remove later
    Eigen::MatrixXf X_train;
    Eigen::VectorXf pca_mean;
    unsigned int n_components;
    Eigen::MatrixXf proj_matrix;
    Eigen::MatrixXf projection;
    bool isFit = false;
public:
    PCAReducer(unsigned int n_components);
    Eigen::MatrixXf getMean();
    void setMean(Eigen::MatrixXf);
    void fit(Eigen::MatrixXf X);
    Eigen::MatrixXf transform(Eigen::MatrixXf X_te);

};


#endif //PCA_PCAREDUCER_H
