#pragma once

#include "enums.h"
#include <complex>
#include <Eigen/Core>
#include <limits>
#include <optional>
#include <spdlog/common.h>
#include <utility>

namespace grit {
    template<typename Scalar_>
    class gdplusk_config {
        public:
        using Scalar     = Scalar_;
        using RealScalar = decltype(std::real(std::declval<Scalar>()));
        using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        Eigen::Index              nev                               = 1;
        Eigen::Index              ncv                               = 8;
        Eigen::Index              b                                 = 2;
        OptRitz                   ritz                              = OptRitz::SR;
        Eigen::Index              max_iters                         = 100;
        Eigen::Index              max_matvecs                       = -1;
        RealScalar                abstol                            = std::numeric_limits<RealScalar>::epsilon() * 10000;
        RealScalar                reltol                            = 0;
        RealScalar                tol_stall_evals                  = RealScalar{0};
        RealScalar                tol_stall_rnorm                 = RealScalar{0};
        RealScalar                inner_tol                         = RealScalar{0.1f};
        Eigen::Index              inner_iters                       = 1000;
        spdlog::level::level_enum logLevel                          = spdlog::level::warn;

        Eigen::Index maxExtraRitzHistory    = 1;
        Eigen::Index maxRitzResidualHistory = 1;
        Eigen::Index maxBasisBlocks         = 8;
        Eigen::Index maxRetainBlocks        = 1;

        gdplusk_config() = default;

        gdplusk_config &set_initial_guess(const MatrixType &V);
        gdplusk_config &clear_initial_guess();

        [[nodiscard]] bool              has_initial_guess() const;
        [[nodiscard]] const MatrixType &initial_guess() const;

        private:
        const MatrixType *initial_guess_ptr = nullptr;
        MatrixType        empty_initial_guess;
    };
}
