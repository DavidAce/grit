---
name: grit-build-test
description: Use when changing GRIT build files, dependency handling, CMake presets, or validation commands.
---

# Build And Test

Use the existing CMake presets. Do not create ad hoc build directories.

Common commands:

```bash
cmake --build --preset kraken
ctest --preset kraken --output-on-failure
cmake --build --preset kraken-debug
ctest --preset kraken-debug --output-on-failure
```

- `kraken` is the normal Release preset on this machine.
- `kraken-debug` is the Debug preset.
- Benchmarks are enabled through the hidden `bench` preset inherited by `kraken` and `kraken-debug`.
- Dependencies are selected with `GRIT_PACKAGE_MANAGER`: `find`, `conan`, or `cmake`.
- Do not hardcode Conan arguments in `CMakeLists.txt` or dependency-provider logic. Preset-level Conan arguments are acceptable when needed.
- xDMRGpp at `/home/cluster_users/x_aceituno/xDMRG` is read-only.
