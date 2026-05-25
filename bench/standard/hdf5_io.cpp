#include "hdf5_io.h"

#include <filesystem>
#include <format>
#include <grit/enums.h>
#include <h5pp/h5pp.h>
#include <stdexcept>
#include <string>

namespace bench_standard {
    namespace {
        constexpr std::string_view group_path   = "/grit/standard";
        constexpr std::string_view dataset_path = "/grit/standard/eigvecs";
    }

    DenseMatrix load_initial_guess_hdf5(const std::filesystem::path &path) {
        auto file = h5pp::File(path.string(), h5pp::FileAccess::READONLY);
        if(!file.linkExists(std::string(dataset_path))) {
            throw std::runtime_error(std::format("HDF5 initial guess is missing dataset {}", dataset_path));
        }
        return file.readDataset<DenseMatrix>(std::string(dataset_path));
    }

    void save_eigvecs_hdf5(const std::filesystem::path &path, const Options &opts, const SolveResult &result) {
        if(result.eigvecs.size() == 0) throw std::runtime_error("No eigenvectors are available to save");
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

        auto file_access = std::filesystem::exists(path) ? h5pp::FileAccess::READWRITE : h5pp::FileAccess::REPLACE;
        auto file        = h5pp::File(path.string(), file_access);
        file.writeDataset(result.eigvecs, std::string(dataset_path));

        file.writeAttribute(opts.matrix, std::string(group_path), "matrix");
        file.writeAttribute(opts.nev, std::string(group_path), "nev");
        file.writeAttribute(opts.ncv, std::string(group_path), "ncv");
        file.writeAttribute(opts.block_size, std::string(group_path), "block_size");
        file.writeAttribute(opts.max_basis_blocks, std::string(group_path), "max_basis_blocks");
        file.writeAttribute(std::string(grit::enum2sv(opts.ritz)), std::string(group_path), "ritz");
        file.writeAttribute(std::string(grit::form::base<Scalar>::ResidualCorrectionToString(opts.residual_correction)), std::string(group_path),
                            "residual_correction");
        file.writeAttribute(opts.refined_rayleigh_ritz, std::string(group_path), "refined_rayleigh_ritz");
        file.writeAttribute(opts.adaptive_inner_tolerance, std::string(group_path), "adaptive_inner_tolerance");
        file.writeAttribute(opts.tol, std::string(group_path), "tol");
        file.writeAttribute(opts.tol_stall_evals, std::string(group_path), "tol_stall_evals");
        file.writeAttribute(opts.tol_stall_rnorm, std::string(group_path), "tol_stall_rnorm");
        file.writeAttribute(result.eigenvalue, std::string(group_path), "eigenvalue");
        file.writeAttribute(result.residual, std::string(group_path), "residual");
        file.writeAttribute(result.iterations, std::string(group_path), "iterations");
        file.writeAttribute(result.matvecs, std::string(group_path), "matvecs");
        file.writeAttribute(result.stop_reason, std::string(group_path), "stop_reason");
    }
}
