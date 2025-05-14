#ifndef SETUP_TRANSFER_H
#define SETUP_TRANSFER_H

#include <npy.hpp>
#include <Eigen/Dense>
#include <string>

void setup_transfer(
        const std::string& inputdir,
        Eigen::MatrixXd& transferMat,
        Eigen::HouseholderQR<Eigen::MatrixXd>& solver,
        Eigen::VectorXd& recoErr,
        const std::string& output_path);

#endif
