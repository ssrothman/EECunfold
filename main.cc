#include "npy.hpp"
#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include "simon_util_cpp/util.h"
#include "simon_util_cpp/recursive_reduce.h"
#include <chrono>
using namespace std::chrono;

#include "CLI11/include/CLI/CLI.hpp"

#define EIGEN_USE_BLAS 
#define EIGEN_USE_LAPACKE

#include <Eigen/Dense>
#include <Eigen/Cholesky>

void get_transfer_shape(const std::vector<size_t>& reco_shape,
                                   const std::vector<size_t>& gen_shape,
                                   std::vector<size_t>& transfer_shape){
    transfer_shape.clear();
    transfer_shape.insert(transfer_shape.end(), reco_shape.begin(), reco_shape.end());
    transfer_shape.insert(transfer_shape.end(), gen_shape.begin(), gen_shape.end());
}

void get_all_shapes(const std::string& folder,
        std::vector<size_t>& gen_shape,
        std::vector<size_t>& covgen_shape,
        std::vector<size_t>& reco_shape,
        std::vector<size_t>& covreco_shape,
        std::vector<size_t>& transfer_shape){
    auto gen_npy = npy::read_npy<double>(folder + "/gen.npy");
    auto covgen_npy = npy::read_npy<double>(folder + "/covgen.npy");
    auto reco_npy = npy::read_npy<double>(folder + "/reco.npy");
    auto covreco_npy = npy::read_npy<double>(folder + "/covreco.npy");

    gen_shape = gen_npy.shape;
    covgen_shape = covgen_npy.shape;
    reco_shape = reco_npy.shape;
    covreco_shape = covreco_npy.shape;
    get_transfer_shape(reco_shape, gen_shape, transfer_shape);
}

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

void forwardfold(const Eigen::VectorXd& gen,
                 const Eigen::MatrixXd& covgen,
                 const Eigen::MatrixXd& transfer,
                 Eigen::VectorXd& forwarded,
                 Eigen::MatrixXd& covforwarded){
    forwarded = transfer * gen;
    covforwarded = transfer * covgen * transfer.transpose();
}

void unfold(const Eigen::VectorXd& reco,
            const Eigen::MatrixXd& covreco,
            const Eigen::MatrixXd& transfer,
            Eigen::VectorXd& unfolded,
            Eigen::MatrixXd& covunfolded){
    Eigen::LLT<Eigen::MatrixXd> llt(covreco);
    Eigen::MatrixXd L = llt.matrixL();
    
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> codL(L);
    Eigen::VectorXd Lreco = codL.solve(reco);
    Eigen::MatrixXd Ltransfer = codL.solve(transfer);

    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> codLtransfer(Ltransfer);
    unfolded = codLtransfer.solve(Lreco);

    Eigen::MatrixXd LtransferLtransfer = Ltransfer.transpose() * Ltransfer;
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> codLtransferLtransfer(LtransferLtransfer);
    covunfolded = codLtransferLtransfer.pseudoInverse();
}

void getGen(const std::string& folder,
            Eigen::VectorXd& gen,
            Eigen::MatrixXd& covgen,
            size_t& gen_size,
            std::vector<size_t>& gen_shape,
            std::vector<size_t>& covgen_shape){
    auto gen_npy = npy::read_npy<double>(folder + "/gen.npy");
    auto covgen_npy = npy::read_npy<double>(folder + "/covgen.npy");

    gen_size = gen_npy.data.size();
    gen_shape = gen_npy.shape;
    covgen_shape = covgen_npy.shape;

    gen = Eigen::Map<Eigen::VectorXd>(gen_npy.data.data(), gen_size);
    covgen = Eigen::Map<Eigen::MatrixXd>(covgen_npy.data.data(), gen_size, gen_size);

    printf("Gen shape: %lu\n", gen_size);
    printf("covgen shape: %lu %lu\n", covgen.rows(), covgen.cols());
}

void getReco(const std::string& folder,
                Eigen::VectorXd& reco,
                Eigen::MatrixXd& covreco,
                size_t& reco_size,
                std::vector<size_t>& reco_shape,
                std::vector<size_t>& covreco_shape){
    auto reco_npy = npy::read_npy<double>(folder + "/reco.npy");
    auto covreco_npy = npy::read_npy<double>(folder + "/covreco.npy");

    reco_size = reco_npy.data.size();
    reco_shape = reco_npy.shape;
    covreco_shape = covreco_npy.shape;

    reco = Eigen::Map<Eigen::VectorXd>(reco_npy.data.data(), reco_size);
    covreco = Eigen::Map<Eigen::MatrixXd>(covreco_npy.data.data(), reco_size, reco_size);

    printf("Reco shape: %lu\n", reco_size);
    printf("covreco shape: %lu %lu\n", covreco.rows(), covreco.cols());
}

void getTransfer(const std::string& transfer_folder,
                 const std::string& reco_template_folder,
                 const std::string& gen_template_folder,
                 Eigen::MatrixXd& transfer,
                 Eigen::VectorXd& reco_template,
                 Eigen::VectorXd& gen_template){

    auto reco_npy = npy::read_npy<double>(reco_template_folder + "/reco.npy");
    auto recopure_npy = npy::read_npy<double>(reco_template_folder + "/recopure.npy");
    auto gen_npy = npy::read_npy<double>(gen_template_folder + "/gen.npy");
    auto genpure_npy = npy::read_npy<double>(gen_template_folder + "/genpure.npy");
    auto transfer_npy = npy::read_npy<double>(transfer_folder + "/transfer_goodshape.npy");

    const size_t reco_size = reco_npy.data.size();
    const size_t gen_size = gen_npy.data.size();

    Eigen::VectorXd reco = Eigen::Map<Eigen::VectorXd>(reco_npy.data.data(), reco_size);
    Eigen::VectorXd recopure = Eigen::Map<Eigen::VectorXd>(recopure_npy.data.data(), reco_size);

    Eigen::VectorXd gen = Eigen::Map<Eigen::VectorXd>(gen_npy.data.data(), gen_size);
    Eigen::VectorXd genpure = Eigen::Map<Eigen::VectorXd>(genpure_npy.data.data(), gen_size);

    Eigen::MatrixXd transfer_expanded = Eigen::Map<Eigen::MatrixXd>(transfer_npy.data.data(), reco_size, gen_size);

    //step 2: divide by gen
    Eigen::MatrixXd transfer_over_gen(reco_size, gen_size);
    for(size_t iReco=0; iReco<reco_size; ++iReco){
        for(size_t iGen=0; iGen<gen_size; ++iGen){
            if(genpure(iGen) > 0){
                transfer_over_gen(iReco, iGen) = transfer_expanded(iReco, iGen) /= genpure(iGen);
            } else {
                transfer_over_gen(iReco, iGen) = 0;
            }
        }
    }

    //check forward
    //printf("forward diff test: %g\n", (transfer_over_gen * genpure - recopure).norm());

    //step 3: compute reco template

    recopure = (recopure.array() == 0).select(1, recopure); //avoid division by zero
    reco_template = reco.array()/recopure.array();
    
    //step 4: compute gen template

    gen = (gen.array() == 0).select(1, gen); //avoid division by zero
    gen_template = genpure.array()/gen.array();

    //step 5: subsume templates into transfer matrix
    transfer = Eigen::MatrixXd(reco_size, gen_size);
    for(size_t iReco=0; iReco<reco_size; ++iReco){
        for(size_t iGen=0; iGen<gen_size; ++iGen){
            transfer(iReco, iGen) = transfer_over_gen(iReco, iGen) * reco_template(iReco) * gen_template(iGen);
        }
    }
    printf("transfer shape: %lu %lu\n", transfer.rows(), transfer.cols());
}

template <typename S, typename T>
T solve(const Eigen::MatrixXd& transfer,
                      const T& reco){
    std::cout << typeid(S).name() << ":" << std::endl;

    auto start = high_resolution_clock::now();

    S solver;

    if constexpr (std::is_same<S, Eigen::BDCSVD<Eigen::MatrixXd>>::value ||
            std::is_same<S, Eigen::JacobiSVD<Eigen::MatrixXd>>::value){
        solver.compute(transfer, Eigen::ComputeThinU | Eigen::ComputeThinV);
    } else {
        solver.compute(transfer);
    }
    T unfolded = solver.solve(reco);

    T forwarded = transfer * unfolded;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);

    printf("\tdiff: %g\n",(forwarded - reco).norm());
    printf("\ttime: %lu\n", duration.count());
    return unfolded;
}

void handle_forward(const std::string& gen_folder,
                    const std::string& transfer_folder,
                    const std::string& reco_template_folder,
                    const std::string& gen_template_folder,
                    const std::string& out_folder){
    Eigen::MatrixXd transfer;
    Eigen::VectorXd gen;
    Eigen::MatrixXd covgen;
    Eigen::VectorXd reco_template;
    Eigen::VectorXd gen_template;
    size_t gen_size;
    std::vector<size_t> gen_shape;
    std::vector<size_t> covgen_shape;
    std::vector<size_t> reco_shape;
    std::vector<size_t> covreco_shape;
    std::vector<size_t> transfer_shape;

    getGen(gen_folder, 
           gen, covgen, 
           gen_size, gen_shape, covgen_shape);
    getTransfer(transfer_folder, 
                reco_template_folder,
                gen_template_folder,
                transfer,
                reco_template,
                gen_template);

    Eigen::VectorXd forwarded;
    Eigen::MatrixXd covforwarded;

    forwardfold(gen, covgen, transfer,
            forwarded, covforwarded);

    get_all_shapes(gen_folder, gen_shape, covgen_shape, reco_shape, covreco_shape, transfer_shape);

    dump_npy(out_folder + "/forwarded.npy", forwarded, reco_shape);
    dump_npy(out_folder + "/covforwarded.npy", covforwarded, covreco_shape);
    dump_npy(out_folder + "/transfer_mat.npy", transfer, transfer_shape);
    dump_npy(out_folder + "/reco_template.npy", reco_template, reco_shape);
    dump_npy(out_folder + "/gen_template.npy", gen_template, gen_shape);
}

void handle_unfold(const std::string& reco_folder,
                   const std::string& transfer_folder,
                   const std::string& reco_template_folder,
                   const std::string& gen_template_folder,
                   const std::string& out_folder){
    Eigen::MatrixXd transfer;
    Eigen::VectorXd reco;
    Eigen::MatrixXd covreco;
    Eigen::VectorXd gen_template;
    Eigen::VectorXd reco_template;
    size_t reco_size;
    std::vector<size_t> reco_shape;
    std::vector<size_t> covreco_shape;
    std::vector<size_t> gen_shape;
    std::vector<size_t> covgen_shape;
    std::vector<size_t> transfer_shape;

    getReco(reco_folder, 
            reco, covreco, 
            reco_size, reco_shape, covreco_shape);
    getTransfer(transfer_folder,
                reco_template_folder,
                gen_template_folder,
                transfer,
                reco_template,
                gen_template);

    Eigen::VectorXd unfolded;
    Eigen::MatrixXd covunfolded;

    unfold(reco, covreco, transfer,
            unfolded, covunfolded);

    dump_npy(out_folder + "/unfolded.npy", unfolded, gen_shape);
    dump_npy(out_folder + "/covunfolded.npy", covunfolded, covgen_shape);
    dump_npy(out_folder + "/transfer_mat.npy", transfer, transfer_shape);
    dump_npy(out_folder + "/reco_template.npy", reco_template, reco_shape);
    dump_npy(out_folder + "/gen_template.npy", gen_template, gen_shape);
}

int main(int argc, char** argv){
    CLI::App program("forward-folding and unfolding utility");
    argv = program.ensure_utf8(argv);

    std::string data_folder;
    std::string transfer_folder;
    std::string reco_template_folder;
    std::string gen_template_folder;
    std::string out_folder;

    CLI::App* foldcmd = program.add_subcommand("fold", "Perform forward folding")->ignore_case();
    foldcmd->add_option("gen", data_folder, 
            "Folder containing gen, covgen matrices")
            ->required()
            ->option_text("PATH");
    foldcmd->add_option("transfer", transfer_folder,
            "Folder containing transfer matrix")
            ->required()
            ->option_text("PATH");
    foldcmd->add_option("reco_template", reco_template_folder,
            "Folder containing reco, recopure for template")
            ->required()
            ->option_text("PATH");
    foldcmd->add_option("gen_template", gen_template_folder,
            "Folder containing gen, genpure for template")
            ->required()
            ->option_text("PATH");
    foldcmd->add_option("out", out_folder,
            "Output folder")
            ->required()
            ->option_text("PATH");
    
    CLI::App* unfoldcmd = program.add_subcommand("unfold", "Perform unfolding")->ignore_case();
    unfoldcmd->add_option("reco", data_folder,
            "Folder containing reco, covreco matrices")
            ->required()
            ->option_text("PATH");
    unfoldcmd->add_option("transfer", transfer_folder,
            "Folder containing transfer matrix")
            ->required()
            ->option_text("PATH");
    unfoldcmd->add_option("reco_template", reco_template_folder,
            "Folder containing reco, recopure for template")
            ->required()
            ->option_text("PATH");
    unfoldcmd->add_option("gen_template", gen_template_folder,
            "Folder containing gen, genpure for template")
            ->required()
            ->option_text("PATH");
    unfoldcmd->add_option("out", out_folder,
            "Output folder")
            ->required()
            ->option_text("PATH");

    CLI::App* foldunfoldcmd = program.add_subcommand("foldunfold", "Perform forward folding and unfolding")->ignore_case();
    foldunfoldcmd->add_option("genAndReco", data_folder,
            "Folder containing gen, reco, ... matrices")
            ->required()
            ->option_text("PATH");
    foldunfoldcmd->add_option("transfer", transfer_folder,
            "Folder containing transfer matrix")
            ->required()
            ->option_text("PATH");
    foldunfoldcmd->add_option("reco_template", reco_template_folder,
            "Folder containing reco, recopure for template")
            ->required()
            ->option_text("PATH");
    foldunfoldcmd->add_option("gen_template", gen_template_folder,
            "Folder containing gen, genpure for template")
            ->required()
            ->option_text("PATH");
    foldunfoldcmd->add_option("out", out_folder,
            "Output folder")
            ->required()
            ->option_text("PATH");

    program.require_subcommand(1);

    CLI11_PARSE(program, argc, argv);

    if(foldcmd->parsed() || foldunfoldcmd->parsed()){
        handle_forward(data_folder, transfer_folder, reco_template_folder, gen_template_folder, out_folder);
    } 

    if(unfoldcmd->parsed() || foldunfoldcmd->parsed()){
        handle_unfold(data_folder, transfer_folder, reco_template_folder, gen_template_folder, out_folder);
    }

    return 0;
}
