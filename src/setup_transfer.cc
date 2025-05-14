#include "setup_transfer.h"
#include "util.h"

#include <stdio.h>

void setup_transfer(
        const std::string& inputdir,
        Eigen::MatrixXd& transferMat,
        Eigen::HouseholderQR<Eigen::MatrixXd>& solver,
        Eigen::VectorXd& recoErr,
        const std::string& output_path){

    auto reco_npy = npy::read_npy<double>(inputdir + "/reco.npy");
    auto unmatchedReco_npy = npy::read_npy<double>(inputdir + "/unmatchedReco.npy");
    auto untransferedReco_npy = npy::read_npy<double>(inputdir + "/untransferedReco.npy");
    auto gen_npy = npy::read_npy<double>(inputdir + "/gen.npy");
    auto unmatchedGen_npy = npy::read_npy<double>(inputdir + "/unmatchedGen.npy");
    auto untransferedGen_npy = npy::read_npy<double>(inputdir + "/untransferedGen.npy");

    auto transfer_npy = npy::read_npy<double>(inputdir + "/transfer.npy");

    std::vector<double> reco_v;
    get_toy(reco_npy, 0, reco_v);
    Eigen::Map<Eigen::VectorXd> reco(reco_v.data(), reco_v.size());

    std::vector<double> unmatchedReco_v;
    get_toy(unmatchedReco_npy, 0, unmatchedReco_v);
    Eigen::Map<Eigen::VectorXd> unmatchedReco(unmatchedReco_v.data(), unmatchedReco_v.size());

    std::vector<double> untransferedReco_v;
    get_toy(untransferedReco_npy, 0, untransferedReco_v);
    Eigen::Map<Eigen::VectorXd> untransferedReco(untransferedReco_v.data(), untransferedReco_v.size());

    std::vector<double> gen_v;
    get_toy(gen_npy, 0, gen_v);
    Eigen::Map<Eigen::VectorXd> gen(gen_v.data(), gen_v.size());

    std::vector<double> unmatchedGen_v;
    get_toy(unmatchedGen_npy, 0, unmatchedGen_v);
    Eigen::Map<Eigen::VectorXd> unmatchedGen(unmatchedGen_v.data(), unmatchedGen_v.size());

    std::vector<double> untransferedGen_v;
    get_toy(untransferedGen_npy, 0, untransferedGen_v);
    Eigen::Map<Eigen::VectorXd> untransferedGen(untransferedGen_v.data(), untransferedGen_v.size());

    Eigen::VectorXd recoPure = reco - unmatchedReco - untransferedReco;
    Eigen::VectorXd genPure = gen - unmatchedGen - untransferedGen;

    //recoErr.resize(recoPure.size());
    //recoErr.setZero();

    //for (size_t iToy = 1; iToy < reco.shape[0]; ++iToy){
    //    std::vector<double> toy;
    //    get_toy(reco, iToy, toy);
    //    Eigen::Map<Eigen::VectorXd> recoToy(toy.data(), toy.size());
    //    recoErr.array() += (recoToy - recoPure).array().square();
    //}
    //recoErr /= (reco.shape[0] - 1);
    //recoErr = recoErr.array().sqrt();

    size_t nBinsReco = reco_npy.shape[1]*reco_npy.shape[2]*reco_npy.shape[3]*reco_npy.shape[4];
    size_t nBinsGen = gen_npy.shape[1]*gen_npy.shape[2]*gen_npy.shape[3]*gen_npy.shape[4];

    transferMat = Eigen::Map<Eigen::MatrixXd>(transfer_npy.data.data(), 
                                              nBinsGen,      
                                              nBinsReco);
    transferMat.transposeInPlace();

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(nBinsGen);
    Eigen::VectorXd sumTransfer = transferMat * ones;
    printf("sumTransfer: %g\n", sumTransfer.sum());
    printf("reco: %g\n", recoPure.sum());
    printf("max(abs(reco - sumTransfer)) = %g\n", (recoPure - sumTransfer).array().abs().maxCoeff());
    printf("\n");
    fflush(stdout);

    printf("min before replace: %g\n", genPure.minCoeff());
    Eigen::VectorXd genDenom = genPure;
    replace_zeros(genDenom);
    printf("min genDenom: %g\n", genDenom.minCoeff());
    printf("min genPure: %g\n", genPure.minCoeff());
    transferMat.array().rowwise() /= genDenom.transpose().array();
    printf("renormed transfer\n");
    fflush(stdout);

    //recoErr = (recoErr.array() == 0).select(recoErr.array(), 1.0);
    //recoPure.array() /= recoErr.array();
    //transfer.array().colwise() /= recoErr.array();
    
    Eigen::VectorXd forward = transferMat * genPure;
    printf("recoPure: %g\n", recoPure.sum());
    printf("forward: %g\n", forward.sum());
    printf("max(abs(recoPure - forward)) = %g\n", (recoPure - forward).array().abs().maxCoeff());
    printf("\n");

    solver.compute(transferMat);
    Eigen::VectorXd unfolded = solver.solve(recoPure);

    printf("unfolded: %g\n", unfolded.sum());
    printf("genPure: %g\n", genPure.sum());
    printf("max(abs(unfolded - genPure)) = %g\n", (unfolded - genPure).array().abs().maxCoeff());

    Eigen::VectorXd unfolded_forward = transferMat * unfolded; 
    printf("error = %g\n", (unfolded_forward - recoPure).norm()/recoPure.norm());
    printf("\n");

    npy::npy_data<double> transferMat_npy;
    transferMat_npy.shape = transfer_npy.shape;
    transferMat_npy.fortran_order = false;
    transferMat_npy.data.insert(
            transferMat_npy.data.end(),
            transferMat.data(),
            transferMat.data() + transferMat.size());
    npy::write_npy(output_path + "transfer.npy", transferMat_npy);
}
