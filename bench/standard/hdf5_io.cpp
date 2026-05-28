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
        constexpr std::string_view group_path         = "/grit/standard";
        constexpr std::string_view dataset_path       = "/grit/standard/eigvecs";
        constexpr std::string_view result_table_path  = "/grit/standard/results";
        constexpr std::string_view result_table_title = "GRIT standard benchmark results";

        struct BenchmarkResultRow {
            using vlen_type = h5pp::varr_t<int64_t>;

            int32_t               case_id  = 0;
            int32_t               rep      = 0;
            uint32_t              seed     = 0;
            uint32_t              rep_seed = 0;
            h5pp::fstr_t<1024>    matrix_path;
            h5pp::fstr_t<1024>    initial_guess;
            int32_t               nev                           = 0;
            int32_t               ncv                           = 0;
            int32_t               block_size                    = 0;
            int32_t               max_basis_blocks              = 0;
            int32_t               max_iters                     = 0;
            int32_t               max_matvecs                   = 0;
            int32_t               inner_max_iters               = 0;
            double                tol                           = 0.0;
            double                tol_rnorm_relative            = 0.0;
            double                sat_eigval_threshold          = 0.0;
            double                sat_rnorm_threshold           = 0.0;
            double                inner_tol                     = 0.0;
            int32_t               auto_min_dwell_iters          = 0;
            double                auto_sat_eigval_threshold     = 0.0;
            double                auto_sat_rnorm_threshold      = 0.0;
            double                auto_jd_start_rnorm_threshold = 0.0;
            int32_t               auto_cheap_probe_interval     = 0;
            double                auto_cheap_probe_factor       = 0.0;
            h5pp::fstr_t<16>      ritz;
            h5pp::fstr_t<32>      residual_correction;
            uint8_t               refined_rayleigh_ritz        = 0;
            uint8_t               use_relative_rnorm_tolerance = 0;
            uint8_t               use_adaptive_inner_tolerance = 0;
            h5pp::fstr_t<64>      stop_reason;
            double                eigenvalue               = 0.0;
            double                residual                 = 0.0;
            int64_t               iterations               = 0;
            int64_t               matvecs                  = 0;
            int64_t               outer_matvecs            = 0;
            int64_t               inner_matvecs            = 0;
            int64_t               jdops_inner              = 0;
            int64_t               first_cheap_to_jd_iter   = -1;
            int64_t               num_cheap_to_jd_switches = 0;
            int64_t               num_jd_to_cheap_switches = 0;
            h5pp::varr_t<int64_t> cheap_to_jd_switch_iters;
            h5pp::varr_t<int64_t> jd_to_cheap_switch_iters;
            double                seconds    = 0.0;
            double                vmrss_mib  = -1.0;
            double                vmhwm_mib  = -1.0;
            double                vmpeak_mib = -1.0;
        };

        template<typename Field>
        std::size_t member_offset(Field BenchmarkResultRow::*member) {
            BenchmarkResultRow row;
            const auto        *base  = reinterpret_cast<const std::byte *>(&row);
            const auto        *field = reinterpret_cast<const std::byte *>(&(row.*member));
            return static_cast<std::size_t>(field - base);
        }

        template<typename Field>
        void insert_field(h5pp::hid::h5t &h5type, std::string_view name, Field BenchmarkResultRow::*member) {
            auto field_type = h5pp::type::getH5Type<Field>();
            if(H5Tinsert(h5type, std::string(name).c_str(), member_offset(member), field_type) < 0) {
                throw std::runtime_error(std::format("Failed to insert HDF5 result table field '{}'", name));
            }
        }

        h5pp::hid::h5t make_result_table_type() {
            auto h5type = h5pp::hid::h5t(H5Tcreate(H5T_COMPOUND, sizeof(BenchmarkResultRow)));
            insert_field(h5type, "case_id", &BenchmarkResultRow::case_id);
            insert_field(h5type, "rep", &BenchmarkResultRow::rep);
            insert_field(h5type, "seed", &BenchmarkResultRow::seed);
            insert_field(h5type, "rep_seed", &BenchmarkResultRow::rep_seed);
            insert_field(h5type, "matrix_path", &BenchmarkResultRow::matrix_path);
            insert_field(h5type, "initial_guess", &BenchmarkResultRow::initial_guess);
            insert_field(h5type, "nev", &BenchmarkResultRow::nev);
            insert_field(h5type, "ncv", &BenchmarkResultRow::ncv);
            insert_field(h5type, "block_size", &BenchmarkResultRow::block_size);
            insert_field(h5type, "max_basis_blocks", &BenchmarkResultRow::max_basis_blocks);
            insert_field(h5type, "max_iters", &BenchmarkResultRow::max_iters);
            insert_field(h5type, "max_matvecs", &BenchmarkResultRow::max_matvecs);
            insert_field(h5type, "inner_max_iters", &BenchmarkResultRow::inner_max_iters);
            insert_field(h5type, "tol", &BenchmarkResultRow::tol);
            insert_field(h5type, "tol_rnorm_relative", &BenchmarkResultRow::tol_rnorm_relative);
            insert_field(h5type, "sat_eigval_threshold", &BenchmarkResultRow::sat_eigval_threshold);
            insert_field(h5type, "sat_rnorm_threshold", &BenchmarkResultRow::sat_rnorm_threshold);
            insert_field(h5type, "inner_tol", &BenchmarkResultRow::inner_tol);
            insert_field(h5type, "auto_min_dwell_iters", &BenchmarkResultRow::auto_min_dwell_iters);
            insert_field(h5type, "auto_sat_eigval_threshold", &BenchmarkResultRow::auto_sat_eigval_threshold);
            insert_field(h5type, "auto_sat_rnorm_threshold", &BenchmarkResultRow::auto_sat_rnorm_threshold);
            insert_field(h5type, "auto_jd_start_rnorm_threshold", &BenchmarkResultRow::auto_jd_start_rnorm_threshold);
            insert_field(h5type, "auto_cheap_probe_interval", &BenchmarkResultRow::auto_cheap_probe_interval);
            insert_field(h5type, "auto_cheap_probe_factor", &BenchmarkResultRow::auto_cheap_probe_factor);
            insert_field(h5type, "ritz", &BenchmarkResultRow::ritz);
            insert_field(h5type, "residual_correction", &BenchmarkResultRow::residual_correction);
            insert_field(h5type, "refined_rayleigh_ritz", &BenchmarkResultRow::refined_rayleigh_ritz);
            insert_field(h5type, "use_relative_rnorm_tolerance", &BenchmarkResultRow::use_relative_rnorm_tolerance);
            insert_field(h5type, "use_adaptive_inner_tolerance", &BenchmarkResultRow::use_adaptive_inner_tolerance);
            insert_field(h5type, "stop_reason", &BenchmarkResultRow::stop_reason);
            insert_field(h5type, "eigenvalue", &BenchmarkResultRow::eigenvalue);
            insert_field(h5type, "residual", &BenchmarkResultRow::residual);
            insert_field(h5type, "iterations", &BenchmarkResultRow::iterations);
            insert_field(h5type, "matvecs", &BenchmarkResultRow::matvecs);
            insert_field(h5type, "outer_matvecs", &BenchmarkResultRow::outer_matvecs);
            insert_field(h5type, "inner_matvecs", &BenchmarkResultRow::inner_matvecs);
            insert_field(h5type, "jdops_inner", &BenchmarkResultRow::jdops_inner);
            insert_field(h5type, "first_cheap_to_jd_iter", &BenchmarkResultRow::first_cheap_to_jd_iter);
            insert_field(h5type, "num_cheap_to_jd_switches", &BenchmarkResultRow::num_cheap_to_jd_switches);
            insert_field(h5type, "num_jd_to_cheap_switches", &BenchmarkResultRow::num_jd_to_cheap_switches);
            insert_field(h5type, "cheap_to_jd_switch_iters", &BenchmarkResultRow::cheap_to_jd_switch_iters);
            insert_field(h5type, "jd_to_cheap_switch_iters", &BenchmarkResultRow::jd_to_cheap_switch_iters);
            insert_field(h5type, "seconds", &BenchmarkResultRow::seconds);
            insert_field(h5type, "vmrss_mib", &BenchmarkResultRow::vmrss_mib);
            insert_field(h5type, "vmhwm_mib", &BenchmarkResultRow::vmhwm_mib);
            insert_field(h5type, "vmpeak_mib", &BenchmarkResultRow::vmpeak_mib);
            return h5type;
        }

        template<std::size_t N>
        h5pp::fstr_t<N> fixed_string(std::string_view text, std::string_view field_name) {
            if(text.size() >= N) throw std::runtime_error(std::format("HDF5 field '{}' is too long: {} >= {}", field_name, text.size(), N));
            return h5pp::fstr_t<N>{text};
        }

        std::string to_string(std::string_view text) { return std::string{text}; }

        std::string to_string(const h5pp::fstr_t<16> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string to_string(const h5pp::fstr_t<32> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string to_string(const h5pp::fstr_t<64> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::string to_string(const h5pp::fstr_t<1024> &text) { return to_string(static_cast<std::string_view>(text)); }

        std::vector<int64_t> to_i64_vector(const std::vector<Eigen::Index> &values) {
            std::vector<int64_t> result;
            result.reserve(values.size());
            for(auto value : values) result.push_back(static_cast<int64_t>(value));
            return result;
        }

        BenchmarkResultRow make_result_row(const SolveResult &result) {
            BenchmarkResultRow row;
            const auto        &opts           = result.options;
            row.case_id                       = static_cast<int32_t>(opts.case_id);
            row.rep                           = static_cast<int32_t>(result.rep);
            row.seed                          = opts.seed;
            row.rep_seed                      = result.rep_seed;
            row.matrix_path                   = fixed_string<1024>(opts.matrix_path, "matrix_path");
            row.initial_guess                 = fixed_string<1024>(opts.initial_guess, "initial_guess");
            row.nev                           = static_cast<int32_t>(opts.nev);
            row.ncv                           = static_cast<int32_t>(opts.ncv);
            row.block_size                    = static_cast<int32_t>(opts.block_size);
            row.max_basis_blocks              = static_cast<int32_t>(opts.max_basis_blocks);
            row.max_iters                     = static_cast<int32_t>(opts.max_iters);
            row.max_matvecs                   = static_cast<int32_t>(opts.max_matvecs);
            row.inner_max_iters               = static_cast<int32_t>(result.inner_iters);
            row.tol                           = opts.tol;
            row.tol_rnorm_relative            = opts.tol_rnorm_relative;
            row.sat_eigval_threshold          = opts.sat_eigval_threshold;
            row.sat_rnorm_threshold           = opts.sat_rnorm_threshold;
            row.inner_tol                     = result.inner_tol;
            row.auto_min_dwell_iters          = static_cast<int32_t>(opts.auto_min_dwell_iters);
            row.auto_sat_eigval_threshold     = opts.auto_sat_eigval_threshold;
            row.auto_sat_rnorm_threshold      = opts.auto_sat_rnorm_threshold;
            row.auto_jd_start_rnorm_threshold = opts.auto_jd_start_rnorm_threshold;
            row.auto_cheap_probe_interval     = static_cast<int32_t>(opts.auto_cheap_probe_interval);
            row.auto_cheap_probe_factor       = opts.auto_cheap_probe_factor;
            row.ritz                          = fixed_string<16>(std::string(grit::enum2sv(opts.ritz)), "ritz");
            row.residual_correction           = fixed_string<32>(residual_correction_name(opts.residual_correction), "residual_correction");
            row.refined_rayleigh_ritz         = static_cast<uint8_t>(opts.refined_rayleigh_ritz);
            row.use_relative_rnorm_tolerance  = static_cast<uint8_t>(opts.use_relative_rnorm_tolerance);
            row.use_adaptive_inner_tolerance  = static_cast<uint8_t>(opts.use_adaptive_inner_tolerance);
            row.stop_reason                   = fixed_string<64>(result.stop_reason, "stop_reason");
            row.eigenvalue                    = result.eigenvalue;
            row.residual                      = result.residual;
            row.iterations                    = static_cast<int64_t>(result.iterations);
            row.matvecs                       = static_cast<int64_t>(result.matvecs);
            row.outer_matvecs                 = static_cast<int64_t>(result.outer_matvecs);
            row.inner_matvecs                 = static_cast<int64_t>(result.inner_matvecs);
            row.jdops_inner                   = static_cast<int64_t>(result.jdops_inner);
            row.first_cheap_to_jd_iter        = static_cast<int64_t>(result.first_cheap_to_jd_iter);
            row.num_cheap_to_jd_switches      = static_cast<int64_t>(result.cheap_to_jd_switch_iters.size());
            row.num_jd_to_cheap_switches      = static_cast<int64_t>(result.jd_to_cheap_switch_iters.size());
            row.cheap_to_jd_switch_iters      = to_i64_vector(result.cheap_to_jd_switch_iters);
            row.jd_to_cheap_switch_iters      = to_i64_vector(result.jd_to_cheap_switch_iters);
            row.seconds                       = result.seconds;
            row.vmrss_mib                     = result.vmrss_mib;
            row.vmhwm_mib                     = result.vmhwm_mib;
            row.vmpeak_mib                    = result.vmpeak_mib;
            return row;
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
            BenchmarkResultRow row;
            int64_t            count     = 0;
            int64_t            converged = 0;
            RunningStats       eigval;
            RunningStats       residual;
            RunningStats       iterations;
            RunningStats       matvecs;
            RunningStats       seconds;
            RunningStats       vmhwm_mib;
            RunningStats       first_jd_switch;

            void add(const BenchmarkResultRow &value) {
                if(count == 0) row = value;
                count++;
                if(to_string(value.stop_reason).find("converged") != std::string::npos) converged++;
                eigval.add(value.eigenvalue);
                residual.add(value.residual);
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

        const auto &opts        = result.options;
        auto file_access = std::filesystem::exists(path) ? h5pp::FileAccess::READWRITE : h5pp::FileAccess::REPLACE;
        auto file        = h5pp::File(path.string(), file_access);
        file.writeDataset(result.eigvecs, dataset_path);

        file.writeAttribute(opts.matrix_path, group_path, "matrix");
        file.writeAttribute(opts.nev, group_path, "nev");
        file.writeAttribute(opts.ncv, group_path, "ncv");
        file.writeAttribute(opts.block_size, group_path, "block_size");
        file.writeAttribute(opts.max_basis_blocks, group_path, "max_basis_blocks");
        file.writeAttribute(grit::enum2sv(opts.ritz), group_path, "ritz");
        file.writeAttribute(grit::form::base<Scalar>::ResidualCorrectionToString(opts.residual_correction), group_path, "residual_correction");
        file.writeAttribute(opts.refined_rayleigh_ritz, group_path, "refined_rayleigh_ritz");
        file.writeAttribute(opts.use_adaptive_inner_tolerance, group_path, "use_adaptive_inner_tolerance");
        file.writeAttribute(opts.tol, group_path, "tol");
        file.writeAttribute(opts.sat_eigval_threshold, group_path, "sat_eigval_threshold");
        file.writeAttribute(opts.sat_rnorm_threshold, group_path, "sat_rnorm_threshold");
        file.writeAttribute(opts.auto_sat_eigval_threshold, group_path, "auto_sat_eigval_threshold");
        file.writeAttribute(opts.auto_sat_rnorm_threshold, group_path, "auto_sat_rnorm_threshold");
        file.writeAttribute(opts.auto_jd_start_rnorm_threshold, group_path, "auto_jd_start_rnorm_threshold");
        file.writeAttribute(opts.auto_cheap_probe_interval, group_path, "auto_cheap_probe_interval");
        file.writeAttribute(opts.auto_cheap_probe_factor, group_path, "auto_cheap_probe_factor");
        file.writeAttribute(result.eigenvalue, group_path, "eigenvalue");
        file.writeAttribute(result.residual, group_path, "residual");
        file.writeAttribute(result.iterations, group_path, "iterations");
        file.writeAttribute(result.matvecs, group_path, "matvecs");
        file.writeAttribute(result.stop_reason, group_path, "stop_reason");
    }

    void initialize_results_hdf5(const std::filesystem::path &path) {
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());

        auto file   = h5pp::File(path.string(), h5pp::FileAccess::REPLACE);
        auto h5type = make_result_table_type();
        file.createTable(h5type, result_table_path, result_table_title);
    }

    void append_result_hdf5(const std::filesystem::path &path, const SolveResult &result) {
        auto file = h5pp::File(path.string(), h5pp::FileAccess::READWRITE);
        auto row  = make_result_row(result);
        file.appendTableRecords(row, result_table_path);
    }

    void print_results_summary_hdf5(const std::filesystem::path &path) {
        auto file = h5pp::File(path.string(), h5pp::FileAccess::READONLY);
        if(!file.linkExists(result_table_path)) { throw std::runtime_error(std::format("HDF5 results file is missing table {}", result_table_path)); }

        auto rows = file.readTableRecords<std::vector<BenchmarkResultRow>>(result_table_path, h5pp::TableSelection::ALL);
        if(rows.empty()) {
            std::println("summary: no benchmark result rows in {}", path.string());
            return;
        }

        std::map<int32_t, SummaryGroup> groups;
        for(const auto &row : rows) groups[row.case_id].add(row);

        std::println("");
        std::println("summary: {}", path.string());
        std::println("{:<5} {:>5} {:>5} {:<17} {:>9} {:>9} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}", "case", "ncv", "blk", "correction", "conv", "reps",
                     "eigval mean", "rnorm mean", "iter mean±se", "matvec mean±se", "time mean±se", "jd switch ±se");

        for(const auto &[case_id, group] : groups) {
            (void) case_id;
            std::println("{:<5} {:>5} {:>5} {:<17} {:>4}/{:<4} {:>9} {:>18.10e} {:>18.4e} {:>18} {:>18} {:>18} {:>18}", group.row.case_id, group.row.ncv,
                         group.row.block_size, to_string(group.row.residual_correction), group.converged, group.count, group.count, group.eigval.mean(),
                         group.residual.mean(), mean_stderr_text(group.iterations, "{:.1f}"), mean_stderr_text(group.matvecs, "{:.1f}"),
                         mean_stderr_text(group.seconds, "{:.4f}"), mean_stderr_text(group.first_jd_switch, "{:.1f}"));
        }
    }
}
