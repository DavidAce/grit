from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version

required_conan_version = ">=2.0"


class GritConan(ConanFile):
    name = "grit"
    version = "0.1.0"
    description = "Generalized Ritz Iteration Toolkit"
    homepage = "https://github.com/DavidAce/grit"
    url = "https://github.com/DavidAce/grit"
    author = "DavidAce <aceituno@kth.se>"
    topics = ("linear-algebra", "eigensolver", "ritz", "davidson")
    license = "MIT"
    settings = "os", "compiler", "build_type", "arch"
    generators = ("CMakeDeps", "CMakeConfigDeps")
    no_copy_source = True
    short_paths = True
    options = {
        "with_bench": [True, False],
    }
    default_options = {
        "with_bench": False,
    }

    @property
    def _compilers_minimum_version(self):
        return {
            "gcc": "12",
            "Visual Studio": "15.7",
            "clang": "16",
            "apple-clang": "10",
        }

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, 23)
        minimum_version = self._compilers_minimum_version.get(str(self.settings.compiler), False)
        if minimum_version and Version(self.settings.compiler.version) < minimum_version:
            raise ConanInvalidConfiguration("grit requires C++23, which your compiler does not support.")
        if not minimum_version:
            self.output.warning("grit requires C++23. Your compiler is unknown. Assuming it supports C++23.")

    def requirements(self):
        self.requires("eigen/[>=3.4.0 <6.0.0]", force=True, transitive_headers=True)
        self.requires("fmt/[>=11 <13]", transitive_headers=True)
        self.requires("spdlog/[>=1.11.0 <2]", transitive_headers=True)
        if self.options.with_bench:
            self.requires("cli11/[>=2.6 <3]", transitive_headers=True)
            self.requires("h5pp/[>=1.11.3 <2]", transitive_headers=True)
