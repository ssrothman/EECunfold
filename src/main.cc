#include <stdio.h>
#include "npy.hpp"
#include <numeric>
#include <Eigen/Dense>

void get_toy(const npy::npy_data<double>& reco, const size_t iToy, 
             std::vector<double>& result){

    result.clear();
    result.reserve(reco.shape[0]*reco.shape[1]*reco.shape[2]*reco.shape[3]);

    size_t index = 0;
    for (size_t pt = 0; pt < reco.shape[0]; ++pt){
        for (size_t R=0; R < reco.shape[1]; ++R){
            for(size_t r=0; r<reco.shape[2]; ++r){
                for(size_t c=0; c<reco.shape[3]; ++c){
                    for(size_t toy=0; toy<reco.shape[4]; ++toy){
                        if(toy == iToy){
                            result.push_back(reco.data[index]);
                        }
                        ++index;
                    }
                }
            }
        }
    }
}

int main(){
    npy::npy_data<double> reco = npy::read_npy<double>("data/recoPure.npy");
    npy::npy_data<double> transfer = npy::read_npy<double>("data/transfer.npy");

    printf("Reco shape: ");
    for (size_t i = 0; i < reco.shape.size(); ++i) {
        printf("%zu ", reco.shape[i]);
    }
    printf("\n");
    printf("Transfer shape: ");
    for (size_t i = 0; i < transfer.shape.size(); ++i) {
        printf("%zu ", transfer.shape[i]);
    }
    printf("\n");

    std::vector<double> nominal;
    get_toy(reco, 0, nominal);
    Eigen::Map<Eigen::VectorXd> recoNominal(nominal.data(), nominal.size());

    Eigen::VectorXd recoErr(recoNominal.size());
    recoErr.setZero();
    for (size_t iToy = 1; iToy < reco.shape[4]; ++iToy){
        std::vector<double> toy;
        get_toy(reco, iToy, toy);
        Eigen::Map<Eigen::VectorXd> recoToy(toy.data(), toy.size());
        recoErr.array() += (recoToy - recoNominal).array().square();
    }
    recoErr /= (reco.shape[4] - 1);
    recoErr = recoErr.array().sqrt();

    size_t nBinsReco = reco.shape[0]*reco.shape[1]*reco.shape[2]*reco.shape[3];
    assert(nBinsReco == recoErr.size());
    assert(nBinsReco == recoNominal.size());

    size_t nBinsGen = transfer.shape[4]*transfer.shape[5]*transfer.shape[6]*transfer.shape[7];

    printf("nBinsReco = %zu\n", nBinsReco);
    printf("nBinsGen = %zu\n", nBinsGen);
    printf("nBinsTransfer = %zu\n", nBinsReco*nBinsGen);
    assert(nBinsReco*nBinsGen == transfer.data.size());

    Eigen::Map<Eigen::MatrixXd> transferMat(transfer.data.data(), 
                                           nBinsReco,      
                                           nBinsGen);

    recoErr = (recoErr.array() == 0).select(recoErr.array(), 1.0);
    recoNominal.array() /= recoErr.array();
    transferMat.array().colwise() /= recoErr.array();

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(transferMat);
    Eigen::VectorXd unfolded = cod.solve(recoNominal);

    npy::npy_data<double> unfolded_npy;
    unfolded_npy.shape = {transfer.shape[4], transfer.shape[5], transfer.shape[6], transfer.shape[7]};
    unfolded_npy.data.resize(unfolded.size());
    std::copy(unfolded.data(), unfolded.data() + unfolded.size(), unfolded_npy.data.begin());
    npy::write_npy("data/unfolded.npy", unfolded_npy);

}
