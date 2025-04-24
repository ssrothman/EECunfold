#ifndef UNFOLD_TOY_H
#define UNFOLD_TOY_H

#include <npy.hpp>
#include <Eigen/Dense>

#include "util.h"

template <typename DECOMPOSITION>
void unfold_toy(
        const DECOMPOSITION& solver,
        const npy::npy_data<double>& reco_npy,
        const npy::npy_data<double>& unmatchedReco_npy,
        const npy::npy_data<double>& untransferedReco_npy,
        const size_t iToy,
        npy::npy_data<double>& unfolded_npy){

    std::vector<double> reco_v;
    get_toy(reco_npy, iToy, reco_v);
    Eigen::Map<Eigen::VectorXd> reco(reco_v.data(), reco_v.size());

    std::vector<double> unmatchedReco_v;
    get_toy(unmatchedReco_npy, iToy, unmatchedReco_v);
    Eigen::Map<Eigen::VectorXd> unmatchedReco(unmatchedReco_v.data(), unmatchedReco_v.size());

    std::vector<double> untransferedReco_v;
    get_toy(untransferedReco_npy, iToy, untransferedReco_v);
    Eigen::Map<Eigen::VectorXd> untransferedReco(untransferedReco_v.data(), untransferedReco_v.size());

    Eigen::VectorXd recoPure = reco - unmatchedReco - untransferedReco;

    Eigen::VectorXd unfolded = solver.solve(recoPure);

    unfolded_npy.data.insert(
            unfolded_npy.data.end(),
            unfolded.data(),
            unfolded.data() + unfolded.size());
    ++unfolded_npy.shape[0];
}

#endif
