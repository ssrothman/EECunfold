#include <stdio.h>
#include <npy.hpp>
#include <numeric>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "unfold_toy.h"
#include "setup_transfer.h"

using json = nlohmann::json;

int main(){
    std::ifstream json_file("config.json");
    json config = json::parse(json_file);

    std::string reco_path = config["reco_path"];
    std::string transfer_path = config["transfer_path"];
    std::string output_path = config["output_path"];

    npy::npy_data<double> reco = npy::read_npy<double>(reco_path+"reco.npy");
    npy::npy_data<double> unmatchedReco = npy::read_npy<double>(reco_path+"unmatchedReco.npy");
    npy::npy_data<double> untransferedReco = npy::read_npy<double>(reco_path+"untransferedReco.npy");

    Eigen::VectorXd recoErr;
    Eigen::HouseholderQR<Eigen::MatrixXd> solver;
    setup_transfer(transfer_path, solver, recoErr);

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
    npy::write_npy(output_path + "unfolded.npy", unfolded_npy);
}
