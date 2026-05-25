#pragma once

#include <string>
#include <string_view>

namespace bench_standard {
    std::string mem_size(double mib);
    double      mem_usage_in_mib(std::string_view info);
}
