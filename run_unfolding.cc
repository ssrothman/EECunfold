#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

#include "npy.hpp"

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#include <Eigen/Dense>

template <typename T>
void dump_npy(const std::string fname,
              const T& data,
              const std::vector<size_t>& shape){
    printf("Dumping to %s\n", fname.c_str());
    npy::npy_data<double> npydata;
    npydata.data = std::vector<double>(data.reshaped().begin(),
                                       data.reshaped().end());
    npydata.shape = shape;
    npydata.fortran_order = false;

    npy::write_npy(fname, npydata);
}

int main(){
    std::string Afile = "A.npy";
    std::string Cfile = "C.npy";
    std::string rfile = "r.npy";
    std::string gfile = "g.npy";

    auto A_npy = npy::read_npy<double>(Afile);
    auto C_npy = npy::read_npy<double>(Cfile);
    auto r_npy = npy::read_npy<double>(rfile);
    auto g_npy = npy::read_npy<double>(gfile);

    Eigen::Map<Eigen::MatrixXd> A(A_npy.data.data(), A_npy.shape[1], A_npy.shape[0]);
    Eigen::Map<Eigen::MatrixXd> C(C_npy.data.data(), C_npy.shape[0], C_npy.shape[1]);
    Eigen::Map<Eigen::VectorXd> r(r_npy.data.data(), r_npy.shape[0]);
    Eigen::Map<Eigen::VectorXd> g(g_npy.data.data(), g_npy.shape[0]);

    printf("A shape: %d x %d\n", A.rows(), A.cols());
    printf("C shape: %d x %d\n", C.rows(), C.cols());
    printf("r shape: %d\n", r.size());
    printf("g shape: %d\n", g.size());

    Eigen::MatrixXd A_t = A.transpose();

    Eigen::VectorXd r_forward = A_t * g;
    printf("DEBUG: (r_forward - r).norm(): %f\n", (r_forward - r).norm());

    //force C to be positive-definite
    for (unsigned i=0; i<C.rows(); ++i){
        C(i, i) += 1e-6;
    }

    //check that C is self-adjoint
    if ((C - C.transpose()).norm() > 1e-6){
        printf("FATAL: C is not self-adjoint\n");
        return 1;
    }else {
        printf("DEBUG: C is self-adjoint\n");
    }

    Eigen::LLT<Eigen::MatrixXd> llt_of_C(C);
    if (llt_of_C.info() != Eigen::Success){
        printf("FATAL: LLT failed; C is probably not positive-definite\n");
    } else {
        printf("DEBUG: LLT succeeded\n");
    }
    Eigen::MatrixXd L = llt_of_C.matrixL();
    //printf("GOT L2\n");
    Eigen::MatrixXd C_reconstructed = L * L.transpose();

    printf("DEBUG: (C_reconstructed - C).norm(): %f\n", (C_reconstructed - C).norm());

    //force L to be non-singular
    for (unsigned i=0; i<L.rows(); ++i){
        if (L(i, i) < 1e-6){
            L(i, i) = 1;
        }
    }

    Eigen::MatrixXd A_tilde = A_t;
    L.triangularView<Eigen::Lower>().solveInPlace(A_tilde);
    Eigen::VectorXd r_tilde = r;
    L.triangularView<Eigen::Lower>().solveInPlace(r_tilde);

    
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(A_tilde);
    if (cod.info() != Eigen::Success){
        printf("FATAL: COD failed\n");
        return 1;
    } else {
        printf("DEBUG: COD succeeded\n");
    }
    Eigen::VectorXd g_pred = cod.solve(r_tilde);
    Eigen::VectorXd r_pred_forward = A_t * g_pred;
    printf("DEBUG: (r_pred_forward - r).norm(): %f\n", (r_pred_forward - r).norm());

    Eigen::VectorXd g_diff = g - g_pred;
    printf("DEBUG: g_diff.norm(): %f\n", g_diff.norm());

    dump_npy("A_tilde.npy", A_tilde, {A_tilde.cols(), A_tilde.rows()});
    dump_npy("r_tilde.npy", r_tilde, {r_tilde.size()});
    dump_npy("g_pred.npy", g_pred, {g_pred.size()});

    Eigen::MatrixXd Hessian = A_tilde.transpose() * A_tilde;
    dump_npy("Hessian.npy", Hessian, {Hessian.cols(), Hessian.rows()});
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_Hessian(Hessian);
    if (cod_Hessian.info() != Eigen::Success){
        printf("FATAL: COD Hessian failed\n");
        return 1;
    } else {
        printf("DEBUG: COD Hessian succeeded\n");
    }
    Eigen::MatrixXd Hessian_inv = cod_Hessian.pseudoInverse();

    dump_npy("cov_g_pred.npy", Hessian_inv, {Hessian_inv.cols(), Hessian_inv.rows()});
    dump_npy("L.npy", L, {L.cols(), L.rows()});

    return 0;
}
