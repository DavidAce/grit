#include "memory.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>

namespace bench_standard {
    std::string mem_size(double mib) {
        if(mib >= 1024.0) return std::format("{:.1f} GiB", mib / 1024.0);
        if(mib >= 0.0) return std::format("{:.1f} MiB", mib);
        return "n/a";
    }

    double mem_usage_in_mib(std::string_view info) {
        std::ifstream status("/proc/self/status");
        std::string   line;
        while(std::getline(status, line)) {
            if(line.rfind(info, 0) != 0) continue;
            auto pos = line.find(':');
            if(pos == std::string::npos) return -1.0;
            auto value = line.substr(pos + 1);
            value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) { return !std::isdigit(c); }), value.end());
            if(value.empty()) return -1.0;
            return static_cast<double>(std::stoll(value)) / 1024.0;
        }
        return -1.0;
    }
}
