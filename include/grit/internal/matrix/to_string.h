#pragma once

#include <Eigen/Core>
#include <sstream>
#include <string>

namespace grit::internal::matrix {
    template<typename Derived>
    std::string to_string(const Eigen::MatrixBase<Derived> &matrix, int = 8) {
        std::ostringstream oss;
        oss << matrix;
        return oss.str();
    }
}
