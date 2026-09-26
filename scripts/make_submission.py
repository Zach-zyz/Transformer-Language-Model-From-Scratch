"""Create the five-file HW1 submission directory and deterministic code archive."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = (
    "cs8803_hw1",
    "configs",
    "scripts",
    "tests",
)
CODE_FILES = (
    "README.md",
    "HANDOUT.md",
    "constraints-tested.txt",
    "requirements.txt",
    "pyproject.toml",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "instructor",
    "release-preview",
}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".bin", ".npy", ".npz", ".pyc"}


def code_paths() -> list[Path]:
    paths = []
    for root_name in CODE_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            paths.append(path)
    for name in CODE_FILES:
        path = ROOT / name
        if path.is_file():
            paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def write_code_zip(destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in code_paths():
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writeup", type=Path, required=True)
    parser.add_argument("--ai-disclosure", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.writeup, args.ai_disclosure, args.run_config, args.metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise SystemExit("output directory is not empty; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    write_code_zip(output / "code.zip")
    shutil.copy2(args.writeup, output / "writeup.pdf")
    shutil.copy2(args.ai_disclosure, output / "ai_disclosure.md")
    shutil.copy2(args.run_config, output / "run_config.yaml")
    shutil.copy2(args.metrics, output / "metrics.json")
    print(output)


if __name__ == "__main__":
    main()
