import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parents[1]
NOWMP_DIR = REPO_ROOT / "yalis" / "attention" / "nowmp_thresh"
DEFAULT_KERNEL = NOWMP_DIR / "kernels" / "thresh_attn_nowmp_cuda.cu"
CPP_SOURCE = NOWMP_DIR / "thresh_attn_nowmp_c.cpp"


def _resolve_kernel_source(kernel_arg: Optional[str]) -> Path:
    kernel_source = kernel_arg or os.getenv("YALIS_NOWMP_KERNEL") or str(DEFAULT_KERNEL)
    kernel_path = Path(kernel_source)
    if not kernel_path.is_absolute():
        kernel_path = (NOWMP_DIR / kernel_path).resolve()
    return kernel_path


def _ext_name_from_kernel(kernel_source: Path) -> str:
    stem = kernel_source.stem
    safe_stem = re.sub(r"[^0-9A-Za-z_]+", "_", stem)
    return f"nowmp_attn_cuda_{safe_stem}"


def build_nowmp_extension(kernel_source: Path, verbose: bool) -> Path:
    if not kernel_source.is_file():
        raise FileNotFoundError(f"Kernel source not found: {kernel_source}")
    if not CPP_SOURCE.is_file():
        raise FileNotFoundError(f"C++ source not found: {CPP_SOURCE}")

    ext_name = _ext_name_from_kernel(kernel_source)
    ext_dir = NOWMP_DIR / "_ext"
    build_dir = ext_dir / "build"
    ext_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    module = load(
        name=ext_name,
        sources=[str(CPP_SOURCE), str(kernel_source)],
        build_directory=str(build_dir),
        verbose=verbose,
        extra_cuda_cflags=["-O3"],
    )
    src_path = Path(module.__file__)
    dst_path = ext_dir / src_path.name
    if src_path.resolve() != dst_path.resolve():
        shutil.copy2(src_path, dst_path)
    return dst_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the nowmp attention CUDA extension once and cache it."
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Path to the .cu kernel (defaults to YALIS_NOWMP_KERNEL or the repo kernel).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose build output.",
    )
    args = parser.parse_args()

    kernel_source = _resolve_kernel_source(args.kernel)
    output_path = build_nowmp_extension(kernel_source, args.verbose)
    print(f"Built nowmp extension at: {output_path}")


if __name__ == "__main__":
    main()
