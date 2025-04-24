#ifndef SETUP_TRANSFER_H
#define SETUP_TRANSFER_H

#include <npy.hpp>
#include <Eigen/Dense>
#include <string>

void setup_transfer(
        const std::string& inputdir,
        Eigen::HouseholderQR<Eigen::MatrixXd>& solver,
        Eigen::VectorXd& recoErr);

#endif
