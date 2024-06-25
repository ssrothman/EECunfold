#include "npy.hpp"
#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include "simon_util_cpp/util.h"
#include "simon_util_cpp/recursive_reduce.h"
#include <chrono>
using namespace std::chrono;

#define EIGEN_USE_BLAS 
#define EIGEN_USE_LAPACKE

#include <Eigen/Dense>
#include <Eigen/Cholesky>

size_t lookup_index_transfer(
        const npy::npy_data<double>& transfer,
        size_t order, 
        size_t btag_reco, size_t pt_reco, size_t dR_reco,
        size_t btag_gen, size_t pt_gen, size_t dR_gen){
    const size_t order_multiplier = transfer.shape[1] * transfer.shape[2] * transfer.shape[3] * transfer.shape[4] * transfer.shape[5] * transfer.shape[6];
    const size_t btag_reco_multiplier = transfer.shape[2] * transfer.shape[3] * transfer.shape[4] * transfer.shape[5] * transfer.shape[6];
    const size_t pt_reco_multiplier = transfer.shape[3] * transfer.shape[4] * transfer.shape[5] * transfer.shape[6];
    const size_t dR_reco_multiplier = transfer.shape[4] * transfer.shape[5] * transfer.shape[6];
    const size_t btag_gen_multiplier = transfer.shape[5] * transfer.shape[6];
    const size_t pt_gen_multiplier = transfer.shape[6];
    const size_t dR_gen_multiplier = 1;

    return order * order_multiplier +
        btag_reco * btag_reco_multiplier +
        pt_reco * pt_reco_multiplier +
        dR_reco * dR_reco_multiplier +
        btag_gen * btag_gen_multiplier +
        pt_gen * pt_gen_multiplier +
        dR_gen * dR_gen_multiplier;
}

size_t lookup_index_vec(
        const npy::npy_data<double>& vec,
        size_t order,
        size_t btag, size_t pt, size_t dR){
    const size_t order_multiplier = vec.shape[1] * vec.shape[2] * vec.shape[3];
    const size_t btag_multiplier = vec.shape[2] * vec.shape[3];
    const size_t pt_multiplier = vec.shape[3];
    const size_t dR_multiplier = 1;

    return order * order_multiplier +
        btag * btag_multiplier +
        pt * pt_multiplier +
        dR * dR_multiplier;
}

template <typename S>
Eigen::VectorXd solve(const Eigen::MatrixXd& transfer,
                      const Eigen::VectorXd& reco,
                      const Eigen::VectorXd& gen){
    std::cout << typeid(S).name() << ":" << std::endl;

    auto start = high_resolution_clock::now();

    S solver;

    if constexpr (std::is_same<S, Eigen::BDCSVD<Eigen::MatrixXd>>::value ||
            std::is_same<S, Eigen::JacobiSVD<Eigen::MatrixXd>>::value){
        solver.compute(transfer, Eigen::ComputeThinU | Eigen::ComputeThinV);
    } else {
        solver.compute(transfer);
    }
    Eigen::VectorXd unfolded = solver.solve(reco);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);

    printf("\tdiff: %g\n",(unfolded - gen).norm());
    printf("\ttime: %lu\n", duration.count());
    return unfolded;
}

int main(){
    auto reco =     npy::read_npy<double>("test/recopure.npy");
    auto gen =      npy::read_npy<double>("test/genpure.npy");
    auto covreco =  npy::read_npy<double>("test/covreco.npy");
    auto covgen =   npy::read_npy<double>("test/covgen.npy");
    auto transfer = npy::read_npy<double>("test/transfer.npy");

    printf("reco shape: ");
    printVec(reco.shape);
    printf("\tnum_elements: %lu\n", reco.data.size());
    printf("\tsum: %f\n", recursive_reduce(reco.data, 0.));
    printf("covreco shape: ");
    printVec(covreco.shape);
    printf("\tnum_elements: %lu\n", covreco.data.size());
    printf("\tsum: %f\n", recursive_reduce(covreco.data, 0.));
    printf("gen shape: ");
    printVec(gen.shape);
    printf("\tnum_elements: %lu\n", gen.data.size());
    printf("\tsum: %f\n", recursive_reduce(gen.data, 0.));
    printf("covgen shape: ");
    printVec(covgen.shape);
    printf("\tnum_elements: %lu\n", covgen.data.size());
    printf("\tsum: %f\n", recursive_reduce(covgen.data, 0.));
    printf("transfer shape: ");
    printVec(transfer.shape);
    printf("\tnum_elements: %lu\n", transfer.data.size());
    printf("\tsum: %f\n", recursive_reduce(transfer.data, 0.));
    printf("\n");

    //Now in order to do the linear algebra 
    //we need to make everything into 2D arrays
    //or 1D vectors

    //reco and gen can just be reiterpreted as 1D
    size_t reco_size = reco.data.size();
    size_t gen_size = gen.data.size();

    //covreco and covgen can just be reinterpreted as 2D
    std::pair<size_t, size_t> covreco_shape = {reco_size, reco_size};
    std::pair<size_t, size_t> covgen_shape = {gen_size, gen_size};

    //transfer is more complicated
    //because it's block-diagonal in the order (first) dimension
    //and we haven't saved the full dimensionality
    // -> we need to expand it back out

    unsigned itransfer=0;
    std::vector<double> transfer_expanded;
    transfer_expanded.reserve(reco_size * gen_size);
    for(unsigned order_gen=0; order_gen < gen.shape[0]; ++order_gen){
        for(unsigned btag_gen=0; btag_gen < gen.shape[1]; ++btag_gen){
            for(unsigned pt_gen=0; pt_gen < gen.shape[2]; ++pt_gen){
                for(unsigned dR_gen=0; dR_gen < gen.shape[3]; ++dR_gen){
                    for(unsigned order_reco=0; order_reco < reco.shape[0]; ++order_reco){
                        for(unsigned btag_reco=0; btag_reco < reco.shape[1]; ++btag_reco){
                            for(unsigned pt_reco=0; pt_reco < reco.shape[2]; ++pt_reco){
                                for(unsigned dR_reco=0; dR_reco < reco.shape[3]; ++dR_reco){
                                    if(order_reco == order_gen){
                                        transfer_expanded.push_back(transfer.data[itransfer++]);
                                    } else {
                                        transfer_expanded.push_back(0.);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    printf("transfer_expanded size: %lu\n\n", transfer_expanded.size());

    //now we can plug these into eigen

    Eigen::Map<Eigen::VectorXd> reco_eigen(reco.data.data(), reco_size);
    Eigen::Map<Eigen::MatrixXd> covreco_eigen(covreco.data.data(), covreco_shape.first, covreco_shape.second);
    Eigen::Map<Eigen::VectorXd> gen_eigen(gen.data.data(), gen_size);
    Eigen::Map<Eigen::MatrixXd> covgen_eigen(covgen.data.data(), covgen_shape.first, covgen_shape.second);
    Eigen::Map<Eigen::MatrixXd> transfer_eigen(transfer_expanded.data(), reco_size, gen_size);

    //oops its backwards
    transfer_eigen.transposeInPlace();

    double sum_reco = reco_eigen.sum();
    double sum_gen = gen_eigen.sum();
    double sum_transfer = transfer_eigen.sum();
    double sum_covreco = covreco_eigen.sum();
    double sum_covgen = covgen_eigen.sum();

    printf("after conversion to Eigen\n");
    printf("sum reco: %f\n", sum_reco);
    printf("sum covreco: %f\n", sum_covreco);
    printf("sum gen: %f\n", sum_gen);
    printf("sum covgen: %f\n", sum_covgen);
    printf("sum transfer: %f\n\n", sum_transfer);

    Eigen::VectorXd sum_transfer_row = transfer_eigen.rowwise().sum();
    Eigen::VectorXd sum_transfer_col = transfer_eigen.colwise().sum();

    Eigen::VectorXd recoLessRowSum = reco_eigen - sum_transfer_row;
    Eigen::VectorXd recoLessColSum = reco_eigen - sum_transfer_col;
    printf("recoLessRowSum: %f\n", recoLessRowSum.norm());
    printf("recoLessColSum: %f\n", recoLessColSum.norm());

    Eigen::VectorXd ones = Eigen::VectorXd::Ones(gen_size);
    Eigen::VectorXd transfer_times_ones = transfer_eigen * ones;
    Eigen::VectorXd onemult_diff = transfer_times_ones - reco_eigen;
    printf("onemult_diff: %f\n", onemult_diff.norm());

    Eigen::MatrixXd transfer_over_gen(reco_size, gen_size);

    for(unsigned iReco=0; iReco<reco_size; ++iReco){
        for(unsigned iGen=0; iGen<gen_size; ++iGen){
            if(gen_eigen(iGen) > 0){
                transfer_over_gen(iReco, iGen) = transfer_eigen(iReco, iGen) /= gen_eigen(iGen);
            } else {
                transfer_over_gen(iReco, iGen) = 0;
            }
        }
    }
    Eigen::VectorXd forward = transfer_over_gen * gen_eigen;
    Eigen::VectorXd diff = reco_eigen - forward;
    printf("forward diff: %f\n\n", diff.norm());




    Eigen::MatrixXd thecov = covreco_eigen;
    //Eigen::MatrixXd thecov = Eigen::MatrixXd::Identity(reco_size, reco_size);

    Eigen::LDLT<Eigen::MatrixXd> llt(thecov);
    Eigen::MatrixXd L = Eigen::MatrixXd::Identity(reco_size, reco_size) 
        * llt.transpositionsP() * llt.matrixL();
    Eigen::VectorXd D = llt.vectorD().real();
    Eigen::MatrixXd check = L * D.asDiagonal() * L.transpose();

    std::cout << (covreco_eigen - check).norm() << std::endl;

    Eigen::VectorXd Ds = D.cwiseSqrt();
    Eigen::VectorXd Di = D.cwiseSqrt().cwiseInverse();
    
    constexpr double threshold = 1e10;
    Eigen::VectorXd Di0 = Di;
    for (int i=0; i<Di.size(); ++i){
        if (!std::isfinite(Di(i)) || Di(i) > threshold){
            Di0(i) = 0;
        }
    }

    Eigen::MatrixXd printmat(D.size(), 4);
    printmat << D, Ds, Di, Di0;

    Eigen::MatrixXd G = L * Di0.asDiagonal();

    std::cout << printmat << std::endl;

    Eigen::VectorXd Greco = G*reco_eigen;
    Eigen::MatrixXd Gtransfer = G*transfer_over_gen;

    /*solve<Eigen::PartialPivLU<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);*/

    solve<Eigen::FullPivLU<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);

    /*solve<Eigen::HouseholderQR<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);*/

    solve<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);

    solve<Eigen::FullPivHouseholderQR<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);

    Eigen::VectorXd unf = solve<Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);

    /*solve<Eigen::LLT<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);*/

    /*solve<Eigen::LDLT<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);*/

    solve<Eigen::BDCSVD<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);

    solve<Eigen::JacobiSVD<Eigen::MatrixXd>>(
            Gtransfer, Greco, gen_eigen);

    //std::cout << unfold-gen_eigen << std::endl;
    //
    std::vector<double> unfvec(unf.reshaped().begin(), unf.reshaped().end());

    npy::npy_data<double> unfnpy; 
    unfnpy.data = unfvec;
    unfnpy.shape = gen.shape;
    unfnpy.fortran_order = false;

    npy::write_npy("unfolded.npy", unfnpy);

    return 0;
}
