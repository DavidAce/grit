#include "hdf5_io.h"
#include "report.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <h5pp/h5pp.h>
#include <limits>
#include <map>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace bench_generalized {
    namespace {
        constexpr std::string_view legacy_group_path        = "/grit/generalized";
        constexpr std::string_view legacy_dataset_path      = "/grit/generalized/eigvecs";
        constexpr std::string_view legacy_result_table_path = "/grit/generalized/results";

        std::string group_path(Algo algo) { return std::format("/grit/generalized/{}", algo_name(algo)); }
        std::string dataset_path(Algo algo) { return std::format("{}/eigvecs", group_path(algo)); }
        std::string result_table_path(Algo algo) { return std::format("{}/results", group_path(algo)); }
        std::string snapshot_group_path(Algo algo) { return std::format("{}/snapshots", group_path(algo)); }
        std::string result_table_title(Algo algo) { return std::format("GRIT generalized {} results", algo_name(algo)); }
        std::string snapshot_table_title(Algo algo) { return std::format("GRIT generalized {} snapshots", algo_name(algo)); }

        template<typename Row, typename Field>
        std::size_t member_offset(Field Row::*member) {
            Row         row;
            const auto *base  = reinterpret_cast<const std::byte *>(&row);
            const auto *field = reinterpret_cast<const std::byte *>(&(row.*member));
            return static_cast<std::size_t>(field - base);
        }

        template<typename Row, typename Field>
        void insert_field(h5pp::hid::h5t &h5type, std::string_view name, Field Row::*member) {
            auto field_type = h5pp::type::getH5Type<Field>();
            if(H5Tinsert(h5type, std::string(name).c_str(), member_offset(member), field_type) < 0) {
                throw std::runtime_error(std::format("Failed to insert HDF5 solver snapshot table field '{}'", name));
            }
        }

        template<typename Row>
        void insert_common_fields(h5pp::hid::h5t &h5type) {
            insert_field(h5type, "case_id", &Row::case_id);
            insert_field(h5type, "rep", &Row::rep);
            insert_field(h5type, "seed", &Row::seed);
            insert_field(h5type, "rep_seed", &Row::rep_seed);
            insert_field(h5type, "matrix_a_path", &Row::matrix_a_path);
            insert_field(h5type, "matrix_b_path", &Row::matrix_b_path);
            insert_field(h5type, "initial_guess", &Row::initial_guess);
            insert_field(h5type, "algo", &Row::algo);
            insert_field(h5type, "nev", &Row::nev);
            insert_field(h5type, "ncv", &Row::ncv);
            insert_field(h5type, "block_size", &Row::block_size);
            insert_field(h5type, "max_iters", &Row::max_iters);
            insert_field(h5type, "max_matvecs", &Row::max_matvecs);
            insert_field(h5type, "abstol", &Row::abstol);
            insert_field(h5type, "reltol", &Row::reltol);
            insert_field(h5type, "ritz_saturation_tolerance", &Row::ritz_saturation_tolerance);
            insert_field(h5type, "quit_when_saturated", &Row::quit_when_saturated);
            insert_field(h5type, "ritz", &Row::ritz);
            insert_field(h5type, "use_refined_rayleigh_ritz", &Row::use_refined_rayleigh_ritz);
            insert_field(h5type, "use_b_inner_product", &Row::use_b_inner_product);
            insert_field(h5type, "use_rescaled_rnorm_tolerance", &Row::use_rescaled_rnorm_tolerance);
            insert_field(h5type, "stop_reason", &Row::stop_reason);
            insert_field(h5type, "eigenvalue", &Row::eigenvalue);
            insert_field(h5type, "rnorm_abs", &Row::rnorm_abs);
            insert_field(h5type, "rnorm_rel", &Row::rnorm_rel);
            insert_field(h5type, "outer_iterations", &Row::outer_iterations);
            insert_field(h5type, "matvecs", &Row::matvecs);
            insert_field(h5type, "outer_matvecs", &Row::outer_matvecs);
            insert_field(h5type, "inner_matvecs", &Row::inner_matvecs);
            insert_field(h5type, "inner_iterations", &Row::inner_iterations);
            insert_field(h5type, "operator_inner", &Row::operator_inner);
            insert_field(h5type, "precond", &Row::precond);
            insert_field(h5type, "precond_inner", &Row::precond_inner);
            insert_field(h5type, "precond_total", &Row::precond_total);
            insert_field(h5type, "saturation_count_eigval", &Row::saturation_count_eigval);
            insert_field(h5type, "saturation_count_rnorm", &Row::saturation_count_rnorm);
            insert_field(h5type, "saturation_count_max", &Row::saturation_count_max);
            insert_field(h5type, "op_norm_estimate", &Row::op_norm_estimate);
            insert_field(h5type, "condition_a", &Row::condition_a);
            insert_field(h5type, "condition_b", &Row::condition_b);
            insert_field(h5type, "gap", &Row::gap);
            insert_field(h5type, "residual_converged", &Row::residual_converged);
            insert_field(h5type, "residual_below_gap", &Row::residual_below_gap);
            insert_field(h5type, "time", &Row::time);
            insert_field(h5type, "vmrss_mib", &Row::vmrss_mib);
            insert_field(h5type, "vmhwm_mib", &Row::vmhwm_mib);
            insert_field(h5type, "vmpeak_mib", &Row::vmpeak_mib);
        }

        template<typename Row>
        h5pp::hid::h5t make_result_table_type() {
            auto h5type = h5pp::hid::h5t(H5Tcreate(H5T_COMPOUND, sizeof(Row)));
            insert_common_fields<Row>(h5type);
            if constexpr(std::is_same_v<Row, GdpluskSnapshot>) {
                insert_field(h5type, "max_basis_blocks", &Row::max_basis_blocks);
                insert_field(h5type, "inner_max_iters", &Row::inner_max_iters);
                insert_field(h5type, "inner_tol", &Row::inner_tol);
                insert_field(h5type, "auto_probe_length", &Row::auto_probe_length);
                insert_field(h5type, "auto_max_probes", &Row::auto_max_probes);
                insert_field(h5type, "residual_correction", &Row::residual_correction);
                insert_field(h5type, "use_jd_b_only", &Row::use_jd_b_only);
                insert_field(h5type, "use_adaptive_inner_tolerance", &Row::use_adaptive_inner_tolerance);
                insert_field(h5type, "inner_tol_last", &Row::inner_tol_last);
                insert_field(h5type, "inner_error_last", &Row::inner_error_last);
                insert_field(h5type, "matvecs_a_inner", &Row::matvecs_a_inner);
                insert_field(h5type, "matvecs_b_inner", &Row::matvecs_b_inner);
                insert_field(h5type, "matvecs_a_total", &Row::matvecs_a_total);
                insert_field(h5type, "matvecs_b_total", &Row::matvecs_b_total);
                insert_field(h5type, "preconditioner_updates_inner", &Row::preconditioner_updates_inner);
                insert_field(h5type, "preconditioner_updates_total", &Row::preconditioner_updates_total);
                insert_field(h5type, "preconditioner_apply_inner", &Row::preconditioner_apply_inner);
                insert_field(h5type, "preconditioner_apply_total", &Row::preconditioner_apply_total);
                insert_field(h5type, "time_solve_inner", &Row::time_solve_inner);
                insert_field(h5type, "time_operator_inner", &Row::time_operator_inner);
                insert_field(h5type, "time_matvecs_a_inner", &Row::time_matvecs_a_inner);
                insert_field(h5type, "time_matvecs_b_inner", &Row::time_matvecs_b_inner);
                insert_field(h5type, "time_preconditioner_inner", &Row::time_preconditioner_inner);
                insert_field(h5type, "time_preconditioner_update_inner", &Row::time_preconditioner_update_inner);
                insert_field(h5type, "time_preconditioner_update_total", &Row::time_preconditioner_update_total);
                insert_field(h5type, "time_preconditioner_apply_inner", &Row::time_preconditioner_apply_inner);
                insert_field(h5type, "time_preconditioner_apply_total", &Row::time_preconditioner_apply_total);
                insert_field(h5type, "time_project_left_inner", &Row::time_project_left_inner);
                insert_field(h5type, "time_project_right_inner", &Row::time_project_right_inner);
                insert_field(h5type, "time_residual_correction", &Row::time_residual_correction);
                insert_field(h5type, "time_build", &Row::time_build);
                insert_field(h5type, "time_status_update", &Row::time_status_update);
                insert_field(h5type, "first_cheap_olsen_to_jd_outer_iter", &Row::first_cheap_olsen_to_jd_outer_iter);
                insert_field(h5type, "residual_correction_active", &Row::residual_correction_active);
                insert_field(h5type, "residual_correction_iteration", &Row::residual_correction_iteration);
                insert_field(h5type, "auto_cheap_olsen_iters", &Row::auto_cheap_olsen_iters);
                insert_field(h5type, "auto_jd_outer_iters_since_probe", &Row::auto_jd_outer_iters_since_probe);
                insert_field(h5type, "num_cheap_olsen_to_jd_switches", &Row::num_cheap_olsen_to_jd_switches);
                insert_field(h5type, "num_jd_to_cheap_olsen_switches", &Row::num_jd_to_cheap_olsen_switches);
                insert_field(h5type, "cheap_olsen_to_jd_switch_outer_iters", &Row::cheap_olsen_to_jd_switch_outer_iters);
                insert_field(h5type, "jd_to_cheap_olsen_switch_outer_iters", &Row::jd_to_cheap_olsen_switch_outer_iters);
            } else if constexpr(std::is_same_v<Row, LanczosSnapshot>) {
                insert_field(h5type, "max_retain_blocks", &Row::max_retain_blocks);
                insert_field(h5type, "time_orthogonalize", &Row::time_orthogonalize);
                insert_field(h5type, "time_orthonormalize", &Row::time_orthonormalize);
                insert_field(h5type, "time_orth_project", &Row::time_orth_project);
                insert_field(h5type, "time_orth_factor", &Row::time_orth_factor);
                insert_field(h5type, "time_orth_update", &Row::time_orth_update);
                insert_field(h5type, "time_orth_refresh", &Row::time_orth_refresh);
                insert_field(h5type, "time_orth_mask", &Row::time_orth_mask);
                insert_field(h5type, "time_diagonalize", &Row::time_diagonalize);
                insert_field(h5type, "time_extract_ritz", &Row::time_extract_ritz);
                insert_field(h5type, "time_restart", &Row::time_restart);
            }
            return h5type;
        }

        std::string to_string(std::string_view text) { return std::string{text}; }
        std::string to_string(const h5pp::fstr_t<16> &text) { return to_string(static_cast<std::string_view>(text)); }
        std::string to_string(const h5pp::fstr_t<32> &text) { return to_string(static_cast<std::string_view>(text)); }
        std::string to_string(const h5pp::fstr_t<64> &text) { return to_string(static_cast<std::string_view>(text)); }
        std::string to_string(const h5pp::fstr_t<1024> &text) { return to_string(static_cast<std::string_view>(text)); }

        template<typename Snapshot>
        std::string snapshot_table_path(Algo algo, const Snapshot &snapshot) {
            return std::format("{}/case_{:03}/rep_{:03}/status", snapshot_group_path(algo), snapshot.case_id, snapshot.rep);
        }

        struct RunningStats {
            int64_t count  = 0;
            double  sum    = 0.0;
            double  sum_sq = 0.0;

            void add(double value) {
                if(!std::isfinite(value)) return;
                count++;
                sum    += value;
                sum_sq += value * value;
            }

            [[nodiscard]] double mean() const { return count > 0 ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN(); }
            [[nodiscard]] double stddev() const {
                if(count < 2) return 0.0;
                auto m   = mean();
                auto var = (sum_sq - static_cast<double>(count) * m * m) / static_cast<double>(count - 1);
                return std::sqrt(std::max(0.0, var));
            }
            [[nodiscard]] double stderr() const { return count > 1 ? stddev() / std::sqrt(static_cast<double>(count)) : 0.0; }
        };

        template<typename Row>
        struct SummaryGroup {
            Row          row;
            int64_t      count     = 0;
            int64_t      converged = 0;
            RunningStats eigval;
            RunningStats rnorm_abs;
            RunningStats rnorm_rel;
            RunningStats outer_iterations;
            RunningStats matvecs;
            RunningStats time;
            RunningStats vmhwm_mib;
            RunningStats first_jd_switch;
            RunningStats time_orthogonalize;
            RunningStats time_orthonormalize;
            RunningStats time_orth_project;
            RunningStats time_orth_factor;
            RunningStats time_orth_update;
            RunningStats time_orth_refresh;
            RunningStats time_orth_mask;
            RunningStats time_diagonalize;
            RunningStats time_extract_ritz;
            RunningStats time_restart;

            void add(const Row &value) {
                if(count == 0) row = value;
                count++;
                if(to_string(value.stop_reason).find("converged") != std::string::npos) converged++;
                eigval.add(value.eigenvalue);
                rnorm_abs.add(value.rnorm_abs);
                rnorm_rel.add(value.rnorm_rel);
                outer_iterations.add(static_cast<double>(value.outer_iterations));
                matvecs.add(static_cast<double>(value.matvecs));
                time.add(value.time);
                vmhwm_mib.add(value.vmhwm_mib);
                if constexpr(std::is_same_v<Row, GdpluskSnapshot>) {
                    if(value.first_cheap_olsen_to_jd_outer_iter >= 0) first_jd_switch.add(static_cast<double>(value.first_cheap_olsen_to_jd_outer_iter));
                } else if constexpr(std::is_same_v<Row, LanczosSnapshot>) {
                    time_orthogonalize.add(value.time_orthogonalize);
                    time_orthonormalize.add(value.time_orthonormalize);
                    time_orth_project.add(value.time_orth_project);
                    time_orth_factor.add(value.time_orth_factor);
                    time_orth_update.add(value.time_orth_update);
                    time_orth_refresh.add(value.time_orth_refresh);
                    time_orth_mask.add(value.time_orth_mask);
                    time_diagonalize.add(value.time_diagonalize);
                    time_extract_ritz.add(value.time_extract_ritz);
                    time_restart.add(value.time_restart);
                }
            }
        };

        std::string mean_stderr_text(const RunningStats &stats, std::string_view fmt = "{:.3f}") {
            if(stats.count == 0) return "n/a";
            auto mean = stats.mean();
            auto se   = stats.stderr();
            return std::format("{} ± {}", std::vformat(fmt, std::make_format_args(mean)), std::vformat(fmt, std::make_format_args(se)));
        }

        template<typename Result>
        consteval Algo            result_algo();
        template<> consteval Algo result_algo<GdpluskSolveResult>() { return Algo::gdplusk; }
        template<> consteval Algo result_algo<LanczosSolveResult>() { return Algo::lanczos; }
        template<> consteval Algo result_algo<LobpcgSolveResult>() { return Algo::lobpcg; }

        template<typename Result>
        void save_eigvecs_hdf5_impl(const std::filesystem::path &path, const Result &result) {
            if(result.eigvecs.size() == 0) throw std::runtime_error("No eigenvectors are available to save");
            if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

            constexpr auto algo        = result_algo<Result>();
            const auto    &snapshot    = result.final;
            auto           file_access = std::filesystem::exists(path) ? h5pp::FileAccess::READWRITE : h5pp::FileAccess::REPLACE;
            auto           file        = h5pp::File(path.string(), file_access);
            file.writeDataset(result.eigvecs, dataset_path(algo));

            file.writeAttribute(to_string(snapshot.matrix_a_path), group_path(algo), "matrix_a");
            file.writeAttribute(to_string(snapshot.matrix_b_path), group_path(algo), "matrix_b");
            file.writeAttribute(to_string(snapshot.algo), group_path(algo), "algo");
            file.writeAttribute(snapshot.nev, group_path(algo), "nev");
            file.writeAttribute(snapshot.ncv, group_path(algo), "ncv");
            file.writeAttribute(snapshot.block_size, group_path(algo), "block_size");
            file.writeAttribute(to_string(snapshot.ritz), group_path(algo), "ritz");
            file.writeAttribute(static_cast<bool>(snapshot.use_refined_rayleigh_ritz), group_path(algo), "use_refined_rayleigh_ritz");
            file.writeAttribute(static_cast<bool>(snapshot.use_b_inner_product), group_path(algo), "use_b_inner_product");
            file.writeAttribute(snapshot.abstol, group_path(algo), "abstol");
            file.writeAttribute(snapshot.ritz_saturation_tolerance, group_path(algo), "ritz_saturation_tolerance");
            file.writeAttribute(static_cast<bool>(snapshot.quit_when_saturated), group_path(algo), "quit_when_saturated");
            if constexpr(std::is_same_v<Result, GdpluskSolveResult>) {
                file.writeAttribute(snapshot.max_basis_blocks, group_path(algo), "max_basis_blocks");
                file.writeAttribute(to_string(snapshot.residual_correction), group_path(algo), "residual_correction");
                file.writeAttribute(static_cast<bool>(snapshot.use_jd_b_only), group_path(algo), "use_jd_b_only");
                file.writeAttribute(static_cast<bool>(snapshot.use_adaptive_inner_tolerance), group_path(algo), "use_adaptive_inner_tolerance");
                file.writeAttribute(snapshot.auto_probe_length, group_path(algo), "auto_probe_length");
                file.writeAttribute(snapshot.auto_max_probes, group_path(algo), "auto_max_probes");
            }
            file.writeAttribute(snapshot.eigenvalue, group_path(algo), "eigenvalue");
            file.writeAttribute(snapshot.rnorm_abs, group_path(algo), "rnorm_abs");
            file.writeAttribute(snapshot.rnorm_rel, group_path(algo), "rnorm_rel");
            file.writeAttribute(snapshot.outer_iterations, group_path(algo), "outer_iterations");
            file.writeAttribute(snapshot.matvecs, group_path(algo), "matvecs");
            file.writeAttribute(to_string(snapshot.stop_reason), group_path(algo), "stop_reason");
        }

        template<typename Result>
        void append_result_hdf5_impl(const std::filesystem::path &path, const Result &result) {
            constexpr auto algo = result_algo<Result>();
            auto           file = h5pp::File(path.string(), h5pp::FileAccess::READWRITE);
            file.appendTableRecords(result.final, result_table_path(algo));
            if(!result.snapshots.empty()) {
                using Row        = std::remove_cvref_t<decltype(result.final)>;
                auto h5type      = make_result_table_type<Row>();
                auto path_status = snapshot_table_path(algo, result.final);
                file.createTable(h5type, path_status, snapshot_table_title(algo));
                file.appendTableRecords(result.snapshots, path_status);
            }
        }

        template<typename Row>
        void print_results_summary_rows(const std::filesystem::path &path, const std::vector<Row> &rows) {
            if(rows.empty()) {
                std::println("summary: no benchmark result rows in {}", path.string());
                return;
            }

            std::map<int32_t, SummaryGroup<Row>> groups;
            for(const auto &row : rows) groups[row.case_id].add(row);

            std::println("");
            std::println("summary: {}", path.string());
            if constexpr(std::is_same_v<Row, GdpluskSnapshot>) {
                std::println("{:<5} {:>8} {:>5} {:>5} {:<17} {:>9} {:>9} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}", "case", "algo", "ncv", "blk",
                             "correction", "conv", "reps", "eigval mean", "rnorm_abs mean", "rnorm_rel mean", "outer_iter mean±se", "matvec mean±se",
                             "time mean±se", "jd switch ±se");
                for(const auto &[case_id, group] : groups) {
                    (void) case_id;
                    std::println("{:<5} {:>8} {:>5} {:>5} {:<17} {:>4}/{:<4} {:>9} {:>18.10e} {:>18.4e} {:>18.4e} {:>18} {:>18} {:>18} {:>18}",
                                 group.row.case_id, to_string(group.row.algo), group.row.ncv, group.row.block_size, to_string(group.row.residual_correction),
                                 group.converged, group.count, group.count, group.eigval.mean(), group.rnorm_abs.mean(), group.rnorm_rel.mean(),
                                 mean_stderr_text(group.outer_iterations, "{:.1f}"), mean_stderr_text(group.matvecs, "{:.1f}"),
                                 mean_stderr_text(group.time, "{:.4f}"), mean_stderr_text(group.first_jd_switch, "{:.1f}"));
                }
            } else if constexpr(std::is_same_v<Row, LanczosSnapshot>) {
                std::println("{:<5} {:>8} {:>5} {:>5} {:>6} {:>9} {:>9} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} "
                             "{:>18} {:>18} {:>18}",
                             "case", "algo", "ncv", "blk", "retain", "conv", "reps", "eigval mean", "rnorm_abs mean", "rnorm_rel mean", "outer_iter mean±se",
                             "matvec mean±se", "time mean±se", "orth mean±se", "orthn mean±se", "proj mean±se", "factor mean±se", "update mean±se",
                             "refresh mean±se", "mask mean±se", "diag mean±se", "ritz mean±se", "restart mean±se");
                for(const auto &[case_id, group] : groups) {
                    (void) case_id;
                    std::println("{:<5} {:>8} {:>5} {:>5} {:>6} {:>4}/{:<4} {:>9} {:>18.10e} {:>18.4e} {:>18.4e} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} "
                                 "{:>18} {:>18} {:>18} {:>18} {:>18} {:>18}",
                                 group.row.case_id, to_string(group.row.algo), group.row.ncv, group.row.block_size, group.row.max_retain_blocks,
                                 group.converged, group.count, group.count, group.eigval.mean(), group.rnorm_abs.mean(), group.rnorm_rel.mean(),
                                 mean_stderr_text(group.outer_iterations, "{:.1f}"), mean_stderr_text(group.matvecs, "{:.1f}"),
                                 mean_stderr_text(group.time, "{:.4f}"), mean_stderr_text(group.time_orthogonalize, "{:.4f}"),
                                 mean_stderr_text(group.time_orthonormalize, "{:.4f}"), mean_stderr_text(group.time_orth_project, "{:.4f}"),
                                 mean_stderr_text(group.time_orth_factor, "{:.4f}"), mean_stderr_text(group.time_orth_update, "{:.4f}"),
                                 mean_stderr_text(group.time_orth_refresh, "{:.4f}"), mean_stderr_text(group.time_orth_mask, "{:.4f}"),
                                 mean_stderr_text(group.time_diagonalize, "{:.4f}"), mean_stderr_text(group.time_extract_ritz, "{:.4f}"),
                                 mean_stderr_text(group.time_restart, "{:.4f}"));
                }
            } else {
                std::println("{:<5} {:>8} {:>5} {:>5} {:>9} {:>9} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}", "case", "algo", "ncv", "blk", "conv", "reps",
                             "eigval mean", "rnorm_abs mean", "rnorm_rel mean", "outer_iter mean±se", "matvec mean±se", "time mean±se");
                for(const auto &[case_id, group] : groups) {
                    (void) case_id;
                    std::println("{:<5} {:>8} {:>5} {:>5} {:>4}/{:<4} {:>9} {:>18.10e} {:>18.4e} {:>18.4e} {:>18} {:>18} {:>18}", group.row.case_id,
                                 to_string(group.row.algo), group.row.ncv, group.row.block_size, group.converged, group.count, group.count, group.eigval.mean(),
                                 group.rnorm_abs.mean(), group.rnorm_rel.mean(), mean_stderr_text(group.outer_iterations, "{:.1f}"),
                                 mean_stderr_text(group.matvecs, "{:.1f}"), mean_stderr_text(group.time, "{:.4f}"));
                }
            }
        }
    }

    DenseMatrix load_initial_guess_hdf5(const std::filesystem::path &path, Algo algo) {
        auto file         = h5pp::File(path.string(), h5pp::FileAccess::READONLY);
        auto algo_dataset = dataset_path(algo);
        if(file.linkExists(algo_dataset)) return file.readDataset<DenseMatrix>(algo_dataset);
        if(file.linkExists(std::string(legacy_dataset_path))) return file.readDataset<DenseMatrix>(std::string(legacy_dataset_path));
        throw std::runtime_error(std::format("HDF5 initial guess is missing datasets {} and {}", algo_dataset, legacy_dataset_path));
    }

    void save_eigvecs_hdf5(const std::filesystem::path &path, const SolveResult &result) {
        std::visit([&](const auto &typed_result) { save_eigvecs_hdf5_impl(path, typed_result); }, result);
    }

    void initialize_results_hdf5(const std::filesystem::path &path, Algo algo) {
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        auto file = h5pp::File(path.string(), h5pp::FileAccess::REPLACE);
        switch(algo) {
            case Algo::gdplusk: {
                auto h5type = make_result_table_type<GdpluskSnapshot>();
                file.createTable(h5type, result_table_path(algo), result_table_title(algo));
                break;
            }
            case Algo::lanczos: {
                auto h5type = make_result_table_type<LanczosSnapshot>();
                file.createTable(h5type, result_table_path(algo), result_table_title(algo));
                break;
            }
            case Algo::lobpcg: {
                auto h5type = make_result_table_type<LobpcgSnapshot>();
                file.createTable(h5type, result_table_path(algo), result_table_title(algo));
                break;
            }
        }
    }

    void append_result_hdf5(const std::filesystem::path &path, const SolveResult &result) {
        std::visit([&](const auto &typed_result) { append_result_hdf5_impl(path, typed_result); }, result);
    }

    void print_results_summary_hdf5(const std::filesystem::path &path, Algo algo) {
        auto file       = h5pp::File(path.string(), h5pp::FileAccess::READONLY);
        auto table_path = result_table_path(algo);
        if(!file.linkExists(table_path)) {
            if(algo == Algo::gdplusk && file.linkExists(std::string(legacy_result_table_path))) {
                auto rows = file.readTableRecords<std::vector<GdpluskSnapshot>>(std::string(legacy_result_table_path), h5pp::TableSelection::ALL);
                print_results_summary_rows(path, rows);
                return;
            }
            throw std::runtime_error(std::format("HDF5 results file is missing table {}", table_path));
        }

        switch(algo) {
            case Algo::gdplusk: {
                auto rows = file.readTableRecords<std::vector<GdpluskSnapshot>>(table_path, h5pp::TableSelection::ALL);
                print_results_summary_rows(path, rows);
                break;
            }
            case Algo::lanczos: {
                auto rows = file.readTableRecords<std::vector<LanczosSnapshot>>(table_path, h5pp::TableSelection::ALL);
                print_results_summary_rows(path, rows);
                break;
            }
            case Algo::lobpcg: {
                auto rows = file.readTableRecords<std::vector<LobpcgSnapshot>>(table_path, h5pp::TableSelection::ALL);
                print_results_summary_rows(path, rows);
                break;
            }
        }
    }
}
