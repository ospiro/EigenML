#include <iostream>
#include <Eigen/Dense>
#include "PCAReducer.h"
int main() {
    Eigen::MatrixXf test_m = Eigen::MatrixXf::Random(100,3);
//    test_m << 1,2.5,3.7,5,50,8.51;
    PCAReducer test_red(2);
    test_red.fit(test_m);
    std::cout<< test_red.transform(test_m);
//    std::cout<<test_red.getMean();
    return 0;
}