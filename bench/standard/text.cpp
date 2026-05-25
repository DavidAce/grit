#include "text.h"

#include <algorithm>
#include <cctype>

namespace bench_standard {
    std::string lower_copy(std::string str) {
        std::ranges::transform(str, str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return str;
    }

    std::string trim_copy(std::string str) {
        auto is_space = [](unsigned char c) { return std::isspace(c); };
        str.erase(str.begin(), std::ranges::find_if_not(str, is_space));
        str.erase(std::find_if_not(str.rbegin(), str.rend(), is_space).base(), str.end());
        return str;
    }
}
