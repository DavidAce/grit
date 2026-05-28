#include "hdf5_io.h"
#include "report.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <grit/enums.h>
#include <h5pp/h5pp.h>
#include <limits>
#include <map>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bench_standard {
    namespace {
        constexpr std::string_view group_path           = "/grit/standard";
        constexpr std::string_view dataset_path         = "/grit/standard/eigvecs";
        constexpr std::string_view result_table_path    = "/grit/standard/results";
        constexpr std::string_view result_table_title   = "GRIT standard solver snapshots";
        constexpr std::string_view snapshot_group_path  = "/grit/standard/snapshots";
        constexpr std::string_view snapshot_table_title = "GRIT standard solver snapshots";

        template<typename Field>
        std::size_t member_offset(Field SolverSnapshot::*member) {
            SolverSnapshot row;
            const auto    *base  = reinterpret_cast<const std::byte *>(&row);
            const auto    *field = reinterpret_cast<const std::byte *>(&(row.*member));
            return static_cast<std::size_t>(field - base);
        }

        template<typename Field>
        void insert_field(h5pp::hid::h5t &h5type, std::string_view name, Field SolverSnapshot::*member) {
            auto field_type = h5pp::type::getH5Type<Field>();
            if(H5Tinsert(h5type, std::string(name).c_str(), member_offset(member), field_type) < 0) {
                throw std::runtime_error(std::format("Failed to insert HDF5 solver snapshot table field '{}'", name));
            }
        }

        h5pp::hid::h5t make_result_table_type() {
            auto h5type = h5pp::hid::h5t(H5Tcreate(H5T_COMPOUND, sizeof(SolverSnapshot)));
            insert_field(h5type, "case_id", &SolverSnapshot::case_id);
            insert_field(h5type, "rep", &SolverSnapshot::rep);
            insert_field(h5type, "seed", &SolverSnapshot::seed);
            insert_field(h5type, "rep_seed", &SolverSnapshot::rep_seed);
            insert_field(h5type, "matrix_path", &SolverSnapshot::matrix_path);
            insert_field(h5type, "initial_guess", &SolverSnapshot::initial_guess);
            insert_field(h5type, "nev", &SolverSnapshot::nev);
            insert_field(h5type, "ncv", &SolverSnapshot::ncv);
            insert_field(h5type, "block_size", &SolverSnapshot::block_size);
            insert_field(h5type, "max_basis_blocks", &SolverSnapshot::max_basis_blocks);
            insert_field(h5type, "max_iters", &SolverSnapshot::max_iters);
            insert_field(h5type, "max_matvecs", &SolverSnapshot::max_matvecs);
            insert_field(h5type, "inner_max_iters", &SolverSnapshot::inner_max_iters);
            insert_field(h5type, "tol", &SolverSnapshot::tol);
            insert_field(h5type, "tol_rnorm_relative", &SolverSnapshot::tol_rnorm_relative);
            insert_field(h5type, "sat_eigval_threshold", &SolverSnapshot::sat_eigval_threshold);
            insert_field(h5type, "sat_rnorm_threshold", &SolverSnapshot::sat_rnorm_threshold);
            insert_field(h5type, "inner_tol", &SolverSnapshot::inner_tol);
            insert_field(h5type, "auto_min_dwell_iters", &SolverSnapshot::auto_min_dwell_iters);
            insert_field(h5type, "auto_sat_eigval_threshold", &SolverSnapshot::auto_sat_eigval_threshold);
            insert_field(h5type, "auto_sat_rnorm_threshold", &SolverSnapshot::auto_sat_rnorm_threshold);
            insert_field(h5type, "auto_jd_start_rnorm_threshold", &SolverSnapshot::auto_jd_start_rnorm_threshold);
            insert_field(h5type, "auto_cheap_probe_interval", &SolverSnapshot::auto_cheap_probe_interval);
            insert_field(h5type, "auto_cheap_probe_factor", &SolverSnapshot::auto_cheap_probe_factor);
            insert_field(h5type, "ritz", &SolverSnapshot::ritz);
            insert_field(h5type, "residual_correction", &SolverSnapshot::residual_correction);
            insert_field(h5type, "use_refined_rayleigh_ritz", &SolverSnapshot::use_refined_rayleigh_ritz);
            insert_field(h5type, "use_relative_rnorm_tolerance", &SolverSnapshot::use_relative_rnorm_tolerance);
            insert_field(h5type, "use_adaptive_inner_tolerance", &SolverSnapshot::use_adaptive_inner_tolerance);
            insert_field(h5type, "stop_reason", &SolverSnapshot::stop_reason);
            insert_field(h5type, "eigenvalue", &SolverSnapshot::eigenvalue);
            insert_field(h5type, "rnorm", &SolverSnapshot::rnorm);
            insert_field(h5type, "rrnorm", &SolverSnapshot::rrnorm);
            insert_field(h5type, "iterations", &SolverSnapshot::iterations);
            insert_field(h5type, "matvecs", &SolverSnapshot::matvecs);
            insert_field(h5type, "outer_matvecs", &SolverSnapshot::outer_matvecs);
            insert_field(h5type, "inner_matvecs", &SolverSnapshot::inner_matvecs);
            insert_field(h5type, "inner_iterations", &SolverSnapshot::inner_iterations);
            insert_field(h5type, "jdops_inner", &SolverSnapshot::jdops_inner);
            insert_field(h5type, "precond", &SolverSnapshot::precond);
            insert_field(h5type, "precond_inner", &SolverSnapshot::precond_inner);
            insert_field(h5type, "precond_total", &SolverSnapshot::precond_total);
            insert_field(h5type, "inner_tol_last", &SolverSnapshot::inner_tol_last);
            insert_field(h5type, "inner_error_last", &SolverSnapshot::inner_error_last);
            insert_field(h5type, "first_cheap_to_jd_iter", &SolverSnapshot::first_cheap_to_jd_iter);
            insert_field(h5type, "residual_correction_active", &SolverSnapshot::residual_correction_active);
            insert_field(h5type, "residual_correction_step", &SolverSnapshot::residual_correction_step);
            insert_field(h5type, "auto_dwell", &SolverSnapshot::auto_dwell);
            insert_field(h5type, "auto_jd_steps_since_probe", &SolverSnapshot::auto_jd_steps_since_probe);
            insert_field(h5type, "saturation_count_eigval", &SolverSnapshot::saturation_count_eigval);
            insert_field(h5type, "saturation_count_rnorm", &SolverSnapshot::saturation_count_rnorm);
            insert_field(h5type, "saturation_count_max", &SolverSnapshot::saturation_count_max);
            insert_field(h5type, "op_norm_estimate", &SolverSnapshot::op_norm_estimate);
            insert_field(h5type, "condition", &SolverSnapshot::condition);
            insert_field(h5type, "sensitivity", &SolverSnapshot::sensitivity);
            insert_field(h5type, "gap", &SolverSnapshot::gap);
            insert_field(h5type, "rnorm_below_tol", &SolverSnapshot::rnorm_below_tol);
            insert_field(h5type, "rnorm_below_gap", &SolverSnapshot::rnorm_below_gap);
            insert_field(h5type, "num_cheap_to_jd_switches", &SolverSnapshot::num_cheap_to_jd_switches);
            insert_field(h5type, "num_jd_to_cheap_switches", &SolverSnapshot::num_jd_to_cheap_switches);
            insert_field(h5type, "cheap_to_jd_switch_iters", &SolverSnapshot::cheap_to_jd_switch_iters);
            insert_field(h5type, "jd_to_cheap_switch_iters", &SolverSnapshot::jd_to_cheap_switch_iters);
            insert_field(h5type, "seconds", &SolverSnapshot::seconds);
            insert_field(h5type, "vmrss_mib", &SolverSnapshot::vmrss_mib);
            insert_field(h5type, "vmhwm_mib", &SolverSnapshot::vmhwm_mib);
            insert_field(h5type, "vmpeak_mib", &SolverSnapshot::vmpeak_mib);
            return h5type;
        }

        std::string to_string(std::string_view text) { return std::string{text}; }

        std::string to_string(const h5pp::fstr_t<16> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string to_string(const h5pp::fstr_t<32> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string to_string(const h5pp::fstr_t<64> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string to_string(const h5pp::fstr_t<1024> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string snapshot_table_path(const SolverSnapshot &snapshot) {
            return std::format("{}/case_{:03}/rep_{:03}/status", snapshot_group_path, snapshot.case_id, snapshot.rep);
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

        struct SummaryGroup {
            SolverSnapshot row;
            int64_t        count     = 0;
            int64_t        converged = 0;
            RunningStats   eigval;
            RunningStats   rnorm;
            RunningStats   rrnorm;
            RunningStats   iterations;
            RunningStats   matvecs;
            RunningStats   seconds;
            RunningStats   vmhwm_mib;
            RunningStats   first_jd_switch;

            void add(const SolverSnapshot &value) {
                if(count == 0) row = value;
                count++;
                if(to_string(value.stop_reason).find("converged") != std::string::npos) converged++;
                eigval.add(value.eigenvalue);
                rnorm.add(value.rnorm);
                rrnorm.add(value.rrnorm);
                iterations.add(static_cast<double>(value.iterations));
                matvecs.add(static_cast<double>(value.matvecs));
                seconds.add(value.seconds);
                vmhwm_mib.add(value.vmhwm_mib);
                if(value.first_cheap_to_jd_iter >= 0) first_jd_switch.add(static_cast<double>(value.first_cheap_to_jd_iter));
            }
        };

        std::string mean_stderr_text(const RunningStats &stats, std::string_view fmt = "{:.3f}") {
            if(stats.count == 0) return "n/a";
            auto mean = stats.mean();
            auto se   = stats.stderr();
            return std::format("{} ± {}", std::vformat(fmt, std::make_format_args(mean)), std::vformat(fmt, std::make_format_args(se)));
        }
    }

    DenseMatrix load_initial_guess_hdf5(const std::filesystem::path &path) {
        auto file = h5pp::File(path.string(), h5pp::FileAccess::READONLY);
        if(!file.linkExists(dataset_path)) { throw std::runtime_error(std::format("HDF5 initial guess is missing dataset {}", dataset_path)); }
        return file.readDataset<DenseMatrix>(dataset_path);
    }

    void save_eigvecs_hdf5(const std::filesystem::path &path, const SolveResult &result) {
        if(result.eigvecs.size() == 0) throw std::runtime_error("No eigenvectors are available to save");
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

        const auto &snapshot    = result.final;
        auto        file_access = std::filesystem::exists(path) ? h5pp::FileAccess::READWRITE : h5pp::FileAccess::REPLACE;
        auto        file        = h5pp::File(path.string(), file_access);
        file.writeDataset(result.eigvecs, dataset_path);

        file.writeAttribute(to_string(snapshot.matrix_path), group_path, "matrix");
        file.writeAttribute(snapshot.nev, group_path, "nev");
        file.writeAttribute(snapshot.ncv, group_path, "ncv");
        file.writeAttribute(snapshot.block_size, group_path, "block_size");
        file.writeAttribute(snapshot.max_basis_blocks, group_path, "max_basis_blocks");
        file.writeAttribute(to_string(snapshot.ritz), group_path, "ritz");
        file.writeAttribute(to_string(snapshot.residual_correction), group_path, "residual_correction");
        file.writeAttribute(static_cast<bool>(snapshot.use_refined_rayleigh_ritz), group_path, "use_refined_rayleigh_ritz");
        file.writeAttribute(static_cast<bool>(snapshot.use_adaptive_inner_tolerance), group_path, "use_adaptive_inner_tolerance");
        file.writeAttribute(snapshot.tol, group_path, "tol");
        file.writeAttribute(snapshot.sat_eigval_threshold, group_path, "sat_eigval_threshold");
        file.writeAttribute(snapshot.sat_rnorm_threshold, group_path, "sat_rnorm_threshold");
        file.writeAttribute(snapshot.auto_sat_eigval_threshold, group_path, "auto_sat_eigval_threshold");
        file.writeAttribute(snapshot.auto_sat_rnorm_threshold, group_path, "auto_sat_rnorm_threshold");
        file.writeAttribute(snapshot.auto_jd_start_rnorm_threshold, group_path, "auto_jd_start_rnorm_threshold");
        file.writeAttribute(snapshot.auto_cheap_probe_interval, group_path, "auto_cheap_probe_interval");
        file.writeAttribute(snapshot.auto_cheap_probe_factor, group_path, "auto_cheap_probe_factor");
        file.writeAttribute(snapshot.eigenvalue, group_path, "eigenvalue");
        file.writeAttribute(snapshot.rnorm, group_path, "rnorm");
        file.writeAttribute(snapshot.rrnorm, group_path, "rrnorm");
        file.writeAttribute(snapshot.iterations, group_path, "iterations");
        file.writeAttribute(snapshot.matvecs, group_path, "matvecs");
        file.writeAttribute(to_string(snapshot.stop_reason), group_path, "stop_reason");
    }

    void initialize_results_hdf5(const std::filesystem::path &path) {
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

        auto file   = h5pp::File(path.string(), h5pp::FileAccess::REPLACE);
        auto h5type = make_result_table_type();
        file.createTable(h5type, result_table_path, result_table_title);
    }

    void append_result_hdf5(const std::filesystem::path &path, const SolveResult &result) {
        auto file = h5pp::File(path.string(), h5pp::FileAccess::READWRITE);
        file.appendTableRecords(result.final, result_table_path);

        if(!result.snapshots.empty()) {
            auto h5type = make_result_table_type();
            auto path   = snapshot_table_path(result.final);
            file.createTable(h5type, path, snapshot_table_title);
            file.appendTableRecords(result.snapshots, path);
        }
    }

    void print_results_summary_hdf5(const std::filesystem::path &path) {
        auto file = h5pp::File(path.string(), h5pp::FileAccess::READONLY);
        if(!file.linkExists(result_table_path)) { throw std::runtime_error(std::format("HDF5 results file is missing table {}", result_table_path)); }

        auto rows = file.readTableRecords<std::vector<SolverSnapshot>>(result_table_path, h5pp::TableSelection::ALL);
        if(rows.empty()) {
            std::println("summary: no benchmark result rows in {}", path.string());
            return;
        }

        std::map<int32_t, SummaryGroup> groups;
        for(const auto &row : rows) groups[row.case_id].add(row);

        std::println("");
        std::println("summary: {}", path.string());
        std::println("{:<5} {:>5} {:>5} {:<17} {:>9} {:>9} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}", "case", "ncv", "blk", "correction", "conv",
                     "reps", "eigval mean", "rnorm mean", "rrnorm mean", "iter mean±se", "matvec mean±se", "time mean±se", "jd switch ±se");

        for(const auto &[case_id, group] : groups) {
            (void) case_id;
            std::println("{:<5} {:>5} {:>5} {:<17} {:>4}/{:<4} {:>9} {:>18.10e} {:>18.4e} {:>18.4e} {:>18} {:>18} {:>18} {:>18}", group.row.case_id,
                         group.row.ncv, group.row.block_size, to_string(group.row.residual_correction), group.converged, group.count, group.count,
                         group.eigval.mean(), group.rnorm.mean(), group.rrnorm.mean(), mean_stderr_text(group.iterations, "{:.1f}"),
                         mean_stderr_text(group.matvecs, "{:.1f}"), mean_stderr_text(group.seconds, "{:.4f}"),
                         mean_stderr_text(group.first_jd_switch, "{:.1f}"));
        }
    }
}
