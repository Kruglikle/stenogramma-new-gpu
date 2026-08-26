"""Fail fast when the Docker container cannot use its requested NVIDIA GPU."""

import sys


def main() -> int:
    try:
        import ctranslate2
        import torch
    except ImportError as exc:
        print(f"GPU check failed: required runtime is not installed: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print(
            "GPU check failed: CUDA is not available to PyTorch. "
            "Check the NVIDIA driver, NVIDIA Container Toolkit, and Docker Compose GPU configuration.",
            file=sys.stderr,
        )
        return 1

    if ctranslate2.get_cuda_device_count() < 1:
        print(
            "GPU check failed: CTranslate2 cannot see a CUDA device.",
            file=sys.stderr,
        )
        return 1

    capability = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = torch.cuda.get_arch_list()
    if device_arch not in supported_arches:
        print(
            f"GPU check failed: device architecture {device_arch} is not supported "
            f"by this PyTorch build ({', '.join(supported_arches)}).",
            file=sys.stderr,
        )
        return 1

    print(
        f"GPU ready: {torch.cuda.get_device_name(0)} "
        f"({device_arch}, PyTorch CUDA {torch.version.cuda}, devices: {torch.cuda.device_count()})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
