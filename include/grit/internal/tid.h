#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace grit::tid {
    enum level { disabled, normal, higher, parent };

    class ur;

    class token {
        private:
        ur *t = nullptr;

        public:
        explicit token(ur &t_, double add_time = 0) noexcept;
        token(ur &t_, std::string_view, double add_time = 0) noexcept;
        ~token() noexcept;
        token(const token &)            = delete;
        token &operator=(const token &) = delete;
        void   tic() noexcept;
        void   toc() noexcept;
        ur    &ref() noexcept;
        ur    *operator->() noexcept;
    };

    class ur {
        private:
        using clock                     = std::chrono::high_resolution_clock;
        clock::time_point tic_timepoint = clock::now();
        clock::time_point lap_timepoint = clock::now();
        double            measured_time = 0.0;
        double            last_interval = 0.0;
        size_t            count         = 0;
        bool              running       = false;
        std::string       label;

        public:
        ur() = default;
        explicit ur(std::string_view label_, level = normal) noexcept : label(label_) {}
        void tic() noexcept {
            tic_timepoint = clock::now();
            running       = true;
        }
        void toc() noexcept {
            auto now       = clock::now();
            last_interval  = std::chrono::duration<double>(now - tic_timepoint).count();
            measured_time += last_interval;
            count++;
            running = false;
        }
        void reset() {
            measured_time = 0.0;
            last_interval = 0.0;
            count         = 0;
            running       = false;
            tic_timepoint = clock::now();
            lap_timepoint = tic_timepoint;
        }
        void                      set_label(std::string_view label_) noexcept { label = label_; }
        [[nodiscard]] std::string get_label() const noexcept { return label; }
        [[nodiscard]] double      get_time() const {
            if(!running) return measured_time;
            return measured_time + std::chrono::duration<double>(clock::now() - tic_timepoint).count();
        }
        [[nodiscard]] double get_last_interval() const { return last_interval; }
        [[nodiscard]] size_t get_tic_count() const { return count; }
        [[nodiscard]] double get_lap() const { return std::chrono::duration<double>(clock::now() - lap_timepoint).count(); }
        double               restart_lap() {
            auto now      = clock::now();
            auto lap      = std::chrono::duration<double>(now - lap_timepoint).count();
            lap_timepoint = now;
            return lap;
        }
        ur &operator+=(double seconds) noexcept {
            measured_time += seconds;
            return *this;
        }
        [[nodiscard]] token tic_token(double add_time = 0) noexcept { return token(*this, add_time); }
        [[nodiscard]] token tic_token(std::string_view prefix, double add_time = 0) noexcept { return token(*this, prefix, add_time); }
    };

    inline token::token(ur &t_, double) noexcept : t(&t_) { t->tic(); }
    inline token::token(ur &t_, std::string_view, double) noexcept : t(&t_) { t->tic(); }
    inline token::~token() noexcept {
        if(t) t->toc();
    }
    inline void token::tic() noexcept {
        if(t) t->tic();
    }
    inline void token::toc() noexcept {
        if(t) t->toc();
    }
    inline ur &token::ref() noexcept { return *t; }
    inline ur *token::operator->() noexcept { return t; }

    inline token tic_token(std::string_view, level = parent, double add_time = 0) {
        static ur timer("global");
        return token(timer, add_time);
    }

    inline token tic_scope(std::string_view key, level l = parent, double add_time = 0) { return tic_token(key, l, add_time); }
}
