from __future__ import annotations

import logging
from typing import Any


def _supported_cuda_major(arch: str) -> int | None:
    if not arch.startswith("sm_") or len(arch) < 5 or not arch[3:].isdigit():
        return None
    return int(arch[3:]) // 10


def _wheel_supports_capability(
    capability: tuple[int, int], supported_arches: list[str]
) -> bool:
    if not supported_arches:
        return True
    device_major = capability[0]
    return any(_supported_cuda_major(arch) == device_major for arch in supported_arches)


def _cuda_device_operational(torch_module: Any) -> None:
    probe = torch_module.ones(1, device="cuda")
    probe.add_(1)
    torch_module.cuda.synchronize(0)


def resolve_torch_device(torch_module: Any, requested_device: str | None) -> str:
    requested = (requested_device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        logging.warning("invalid GIGAAM_DEVICE=%r; using auto", requested_device)
        requested = "auto"

    if requested == "cpu":
        return "cpu"

    if not torch_module.cuda.is_available():
        if requested == "cuda":
            logging.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"

    try:
        capability = torch_module.cuda.get_device_capability(0)
        supported_arches = list(
            getattr(torch_module.cuda, "get_arch_list", lambda: [])()
        )
        architecture = f"sm_{capability[0]}{capability[1]}"
        if not _wheel_supports_capability(capability, supported_arches):
            logging.warning(
                "CUDA architecture %s is unsupported by this PyTorch wheel (%s); "
                "falling back to CPU",
                architecture,
                ", ".join(supported_arches) or "unknown architectures",
            )
            return "cpu"
        _cuda_device_operational(torch_module)
    except Exception:
        logging.exception("CUDA probe failed; falling back to CPU")
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass
        return "cpu"

    return "cuda"
