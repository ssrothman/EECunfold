#include <stdio.h>
#include <npy.hpp>
#include <numeric>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "unfold_toy.h"
#include "forward_toy.h"
#include "setup_transfer.h"

using json = nlohmann::json;

int main(){
    std::ifstream json_file("config.json");
    json config = json::parse(json_file);

    std::string data_path = config["data_path"];
    std::string transfer_path = config["transfer_path"];
    std::string output_path = config["output_path"];

    Eigen::VectorXd recoErr;
    Eigen::MatrixXd transferMat;
    Eigen::HouseholderQR<Eigen::MatrixXd> solver;
    setup_transfer(transfer_path, transferMat, solver, recoErr, 
                   output_path);

    if (config["unfold"]){
        npy::npy_data<double> reco = npy::read_npy<double>(data_path+"reco.npy");
        npy::npy_data<double> unmatchedReco = npy::read_npy<double>(data_path+"unmatchedReco.npy");
        npy::npy_data<double> untransferedReco = npy::read_npy<double>(data_path+"untransferedReco.npy");

        printf("UNFOLDING TOYS\n");
        npy::npy_data<double> unfolded_npy;
        unfolded_npy.shape = {0, reco.shape[1], reco.shape[2], 
                                 reco.shape[3], reco.shape[4]};
        unfolded_npy.fortran_order = false;

        for (size_t iToy = 0; iToy < reco.shape[0]; ++iToy){
            printf("iToy = %zu/%zu (%0.2f%%)\r", iToy+1, reco.shape[0],
                   100.0*(iToy+1)/reco.shape[0]);
            if (iToy % 10 == 9){
                fflush(stdout);
            }
            unfold_toy(solver, reco, unmatchedReco, untransferedReco,
                       iToy, unfolded_npy);
        }
        printf("\n");
        npy::write_npy(output_path + "unfolded.npy", unfolded_npy);
    } 
    if (config["forwardfold"]){
        npy::npy_data<double> gen = npy::read_npy<double>(data_path+"gen.npy");
        npy::npy_data<double> unmatchedGen = npy::read_npy<double>(data_path+"unmatchedGen.npy");
        npy::npy_data<double> untransferedGen = npy::read_npy<double>(data_path+"untransferedGen.npy");

        printf("FORWARD-FOLDING TOYS\n");
        npy::npy_data<double> forward_npy;
        forward_npy.shape = {0, gen.shape[1], gen.shape[2], 
                                gen.shape[3], gen.shape[4]};
        forward_npy.fortran_order = false;

        for (size_t iToy = 0; iToy < gen.shape[0]; ++iToy){
            printf("iToy = %zu/%zu (%0.2f%%)\r", iToy+1, gen.shape[0],
                   100.0*(iToy+1)/gen.shape[0]);
            if (iToy % 10 == 9){
                fflush(stdout);
            }
            forward_toy(transferMat, gen, unmatchedGen, untransferedGen,
                        iToy, forward_npy);
        }
        printf("\n");
        npy::write_npy(output_path + "forward.npy", forward_npy);
    } 
}
