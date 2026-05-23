#pragma once

#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace grit::Logger {
    using LoggerHandle = std::shared_ptr<spdlog::logger>;

    inline LoggerHandle getLogger(const std::string &name) {
        auto logger = spdlog::get(name);
        if(!logger) logger = spdlog::stdout_color_mt(name, spdlog::color_mode::always);
        return logger;
    }

    inline void setLevel(LoggerHandle &logger, spdlog::level::level_enum level) {
        if(!logger) logger = getLogger("grit");
        logger->set_level(level);
    }
}

namespace grit {
    inline Logger::LoggerHandle log = Logger::getLogger("grit");
}
