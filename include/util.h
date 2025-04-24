#ifndef UTIL_H
#define UTIL_H

#include <npy.hpp>
#include <Eigen/Dense>

void replace_zeros(Eigen::VectorXd& vec);

void get_toy(const npy::npy_data<double>& reco, const size_t iToy, 
             std::vector<double>& result);

int validate(const npy::npy_data<double>& reco,
             const npy::npy_data<double>& gen,
             const npy::npy_data<double>& transfer);

#endif
