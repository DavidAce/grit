#pragma once

#include "aliases.h"
#include <filesystem>

namespace bench_standard {
    SparseMatrix read_matrix_market(const std::filesystem::path &path);
}
