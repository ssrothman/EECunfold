#include "util.h"

void replace_zeros(Eigen::VectorXd& vec){
    for (long i = 0; i < vec.size(); ++i){
        if (vec[i] == 0){
            vec[i] = 1;
        }
    }
}

void get_toy(const npy::npy_data<double>& reco, const size_t iToy, 
             std::vector<double>& result){

    result.clear();
    result.reserve(reco.shape[0]*reco.shape[1]*reco.shape[2]*reco.shape[3]);

    size_t index = 0;
    for(size_t toy=0; toy<reco.shape[0]; ++toy){
        for (size_t pt = 0; pt < reco.shape[1]; ++pt){
            for (size_t R=0; R < reco.shape[2]; ++R){
                for(size_t r=0; r<reco.shape[3]; ++r){
                    for(size_t c=0; c<reco.shape[4]; ++c){
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

int validate(const npy::npy_data<double>& reco,
             const npy::npy_data<double>& gen,
             const npy::npy_data<double>& transfer){

    assert(reco.fortran_order == false);
    assert(gen.fortran_order == false);
    assert(transfer.fortran_order == false);

    size_t nBinsReco = reco.shape[1]*reco.shape[2]*reco.shape[3]*reco.shape[4];
    size_t nBinsGen = gen.shape[1]*gen.shape[2]*gen.shape[3]*gen.shape[4];
    assert(nBinsReco*nBinsGen == transfer.data.size());

    assert(transfer.shape[0] == reco.shape[1]);
    assert(transfer.shape[1] == reco.shape[2]);
    assert(transfer.shape[2] == reco.shape[3]);
    assert(transfer.shape[3] == reco.shape[4]);
    assert(transfer.shape[4] == gen.shape[1]);
    assert(transfer.shape[5] == gen.shape[2]);
    assert(transfer.shape[6] == gen.shape[3]);
    assert(transfer.shape[7] == gen.shape[4]);

    return 0;
}
