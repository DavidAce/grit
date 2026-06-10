#pragma once

#include "aliases.h"
#include "options.h"
#include "solve_result.h"

namespace bench_generalized {
    SolveResult solve_once(const SparseMatrix &matrix_a, const SparseMatrix &matrix_b, Options opts, int rep);
}
