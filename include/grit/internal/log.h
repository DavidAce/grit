#pragma once

#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace grit::Logger {
    using LoggerHandle = std::shared_ptr<spdlog::logger>;

    inline LoggerHandle getLogger(const std::string &name) {
        auto logger = spdlog::get(name);
        if(!logger) {
            logger = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
            logger->set_level(spdlog::level::warn);
        }
        return logger;
    }
}
