from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def find_intel_openmp_library() -> Path | None:
    env_library = os.environ.get("OMP_LIBRARY") or os.environ.get(
        "INTEL_OPENMP_LIBRARY"
    )
    if env_library:
        library = Path(env_library).expanduser().resolve()
        if library.exists():
            return library

    search_roots = [
        os.environ.get("CMPLR_ROOT"),
        os.environ.get("ONEAPI_ROOT"),
        str(Path.home() / "intel" / "oneapi"),
    ]
    suffixes = [
        "lib/libiomp5.so",
        "compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so",
        "compiler/latest/lib/libiomp5.so",
        "compiler/2024.2/lib/libiomp5.so",
        "compiler/2023.2.0/linux/compiler/lib/intel64_lin/libiomp5.so",
    ]

    for root in search_roots:
        if not root:
            continue
        root_path = Path(root).expanduser()
        for suffix in suffixes:
            candidate = root_path / suffix
            if candidate.exists():
                return candidate.resolve()

    return None


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ".") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        if "MKL_THREADING" in os.environ:
            cmake_args.append(f"-DMKL_THREADING={os.environ['MKL_THREADING']}")

        openmp_library = find_intel_openmp_library()
        if openmp_library:
            cmake_args.extend(
                [
                    f"-DOMP_LIBRARY={openmp_library}",
                    f"-DOMP_LINK={openmp_library}",
                    f"-DCMAKE_BUILD_RPATH={openmp_library.parent}",
                    f"-DCMAKE_INSTALL_RPATH={openmp_library.parent}",
                ]
            )

        cmake_prefix_paths = []
        try:
            import pybind11

            cmake_prefix_paths.append(pybind11.get_cmake_dir())
        except Exception:
            pass

        try:
            import torch

            cmake_prefix_paths.append(torch.utils.cmake_prefix_path)
        except Exception:
            pass

        for prefix in (sys.prefix, os.environ.get("CONDA_PREFIX")):
            if prefix:
                cmake_prefix_paths.append(str(Path(prefix).resolve()))

        existing_prefix = os.environ.get("CMAKE_PREFIX_PATH")
        if existing_prefix:
            cmake_prefix_paths.append(existing_prefix)
        if cmake_prefix_paths:
            cmake_args.append(
                "-DCMAKE_PREFIX_PATH=" + ";".join(cmake_prefix_paths)
            )

        build_args = ["--config", cfg, "--target", "tf_lib"]
        if self.parallel:
            build_args.extend(["--parallel", str(self.parallel)])

        subprocess.check_call(
            ["cmake", "-S", str(ext.sourcedir), "-B", str(build_temp), *cmake_args]
        )
        subprocess.check_call(["cmake", "--build", str(build_temp), *build_args])


setup(
    name="tf-lib",
    version="0.0.1",
    description="Python bindings for the tile fusion library",
    ext_modules=[CMakeExtension("tf_lib")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
