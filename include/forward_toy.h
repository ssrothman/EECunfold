#ifndef FORWARD_TOY_H
#define FORWARD_TOY_H

#include <npy.hpp>
#include <Eigen/Dense>

#include "util.h"

inline void forward_toy(
        const Eigen::MatrixXd& transferMat,
        const npy::npy_data<double>& gen_npy,
        const npy::npy_data<double>& unmatchedGen_npy,
        const npy::npy_data<double>& untransferedGen_npy,
        const size_t iToy,
        npy::npy_data<double>& forward_npy){

    std::vector<double> gen_v;
    get_toy(gen_npy, iToy, gen_v);
    Eigen::Map<Eigen::VectorXd> gen(gen_v.data(), gen_v.size());

    std::vector<double> unmatchedGen_v;
    get_toy(unmatchedGen_npy, iToy, unmatchedGen_v);
    Eigen::Map<Eigen::VectorXd> unmatchedGen(unmatchedGen_v.data(), unmatchedGen_v.size());

    std::vector<double> untransferedGen_v;
    get_toy(untransferedGen_npy, iToy, untransferedGen_v);
    Eigen::Map<Eigen::VectorXd> untransferedGen(untransferedGen_v.data(), untransferedGen_v.size());

    Eigen::VectorXd genPure = gen - unmatchedGen - untransferedGen;

    Eigen::VectorXd forward = transferMat * genPure;

    forward_npy.data.insert(
            forward_npy.data.end(),
            forward.data(),
            forward.data() + forward.size());
    ++forward_npy.shape[0];
}

#endif
