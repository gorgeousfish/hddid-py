from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist


_PUBLIC_RELEASE_MODULES = frozenset(
    {
        "__init__.py",
        "basis.py",
        "estimation.py",
        "estimator.py",
        "inference.py",
        "inputs.py",
        "nuisance.py",
        "plotting.py",
        "results.py",
        "score.py",
        "splitting.py",
        "validation.py",
    }
)


def _is_space_suffixed_module(path: str) -> bool:
    return Path(path).name.endswith((" 2.py", " 3.py"))


def _is_internal_release_module(path: str) -> bool:
    stem = Path(path).stem
    return any(
        token in stem
        for token in ("probe", "trigger", "phase7", "automation_state")
    )


def _is_release_excluded_module(path: str) -> bool:
    module_name = Path(path).name
    return (
        _is_space_suffixed_module(path)
        or _is_internal_release_module(path)
        or module_name not in _PUBLIC_RELEASE_MODULES
    )


class build_py(_build_py):
    def run(self):
        package_build_dir = Path(self.build_lib) / "hddid"
        if package_build_dir.exists():
            shutil.rmtree(package_build_dir)
        super().run()

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if not _is_release_excluded_module(module[2])
        ]


class sdist(_sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        package_dir = Path(base_dir) / "src" / "hddid"
        if package_dir.exists():
            for path in package_dir.glob("*.py"):
                if _is_release_excluded_module(str(path)):
                    path.unlink()


setup(cmdclass={"build_py": build_py, "sdist": sdist})
