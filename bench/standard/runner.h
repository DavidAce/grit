#pragma once

#include "aliases.h"
#include "options.h"
#include "solve_result.h"
#include <grit/config.h>

namespace bench_standard {
    grit::gdplusk_config<Scalar> make_solver_config(const Options &opts);
    SolveResult                  solve_once(const SparseMatrix &matrix, Options opts, int rep);
}
