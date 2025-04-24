#include <stdio.h>
#include "npy.hpp"
#include <numeric>
#include <Eigen/Dense>

#include "unfold_toy.h"
#include "setup_transfer.h"

int main(){
    npy::npy_data<double> reco = npy::read_npy<double>("pythia_data/reco.npy");
    npy::npy_data<double> unmatchedReco = npy::read_npy<double>("pythia_data/unmatchedReco.npy");
    npy::npy_data<double> untransferedReco = npy::read_npy<double>("pythia_data/untransferedReco.npy");

    Eigen::VectorXd recoErr;
    Eigen::HouseholderQR<Eigen::MatrixXd> solver;
    setup_transfer("herwig_data", solver, recoErr);

    printf("RUNNING TOYS\n");
    npy::npy_data<double> unfolded_npy;
    unfolded_npy.shape = {0, reco.shape[1], reco.shape[2], 
                             reco.shape[3], reco.shape[4]};
    unfolded_npy.fortran_order = false;

    for (size_t iToy = 0; iToy < reco.shape[0]; ++iToy){
        printf("iToy = %zu\n", iToy);
        unfold_toy(solver, reco, unmatchedReco, untransferedReco,
                   iToy, unfolded_npy);
    }
    printf("unfolded_npy.shape = ");
    size_t product=1;
    for (size_t i = 0; i < unfolded_npy.shape.size(); ++i) {
        printf("%zu ", unfolded_npy.shape[i]);
        product *= unfolded_npy.shape[i];
    }
    printf("\n");
    printf("Total entries: %zu\n", unfolded_npy.data.size());
    printf("\t(dimension products: %zu)\n", product);
    printf("\n");
    npy::write_npy("data/pythia_unfolded_with_herwig.npy", unfolded_npy);
}
