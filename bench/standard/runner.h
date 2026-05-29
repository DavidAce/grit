#pragma once

#include "aliases.h"
#include "options.h"
#include "solve_result.h"

namespace bench_standard {
    SolveResult                  solve_once(const SparseMatrix &matrix, Options opts, int rep);
}
