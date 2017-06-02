//
// Created by Oliver Spiro on 5/24/17.
//
#include <iostream>
#include <vector>
#include "PCAReducer.h"
#include <numeric>
#include<String>

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &vec){
//  source: https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    std::vector<size_t> idx(vec.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(),idx.end(), [&vec](size_t i_1, size_t i_2){return vec[i_1] > vec[i_2];});
    return idx;
}
PCAReducer::PCAReducer(unsigned int n_components){
   this->n_components = n_components;
}
void PCAReducer::fit(Eigen::MatrixXf X){
// compute data mean
    X_train = X;
    pca_mean = X_train.colwise().mean();
    Eigen::MatrixXf centered_data=X_train.rowwise() - pca_mean.transpose();
//  compute covariance matrix
    int n_samples = X_train.rows();
    Eigen::MatrixXf covariance = 1.0/(n_samples-1) * (centered_data.transpose() * centered_data);
//  Compute eigenvalues and eigenvectors
    Eigen::EigenSolver<Eigen::MatrixXf> solver;
    solver.compute(covariance);
    Eigen::EigenSolver<Eigen::MatrixXf>::EigenvalueType evals = solver.eigenvalues();
    Eigen::EigenSolver<Eigen::MatrixXf>::EigenvectorsType evecs = solver.eigenvectors();
    std::vector<float> eigenvals;
//   Don't think there is an internal Eigen sort method, so converting to vector
    for(int i = 0; i<evals.rows();i++){
        eigenvals.push_back(evals(i).real());
    }
    std::vector<size_t> sorted_eval_idx = argsort(eigenvals);
    Eigen::EigenSolver<Eigen::MatrixXf>::EigenvectorsType sorted_evecs(covariance.rows(),n_components);
    for(int i = 0; i<n_components;i++){
        sorted_evecs.col(i) = evecs.col(sorted_eval_idx[i]);
    }
    proj_matrix = sorted_evecs.real();
    projection = centered_data*proj_matrix;
    isFit = true;
}
Eigen::MatrixXf PCAReducer::transform(Eigen::MatrixXf X_te) {
    if(X_te.rows() > X_te.cols()){
        std::cerr<<"Warning: More features than samples, be sure your matrix is in n x d format."<<std::endl;
    }
    if(!isFit){
        std::cerr<<"Error: Cannot transform before fitting.";
        exit(1);
    }
    return (X_te.rowwise() - pca_mean.transpose())*proj_matrix;
}
Eigen::MatrixXf PCAReducer::getMean() {
    return pca_mean;
}




