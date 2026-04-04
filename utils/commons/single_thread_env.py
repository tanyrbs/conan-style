import os
from collections.abc import Mapping


SINGLE_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
}


def configure_single_thread_env(
    defaults: Mapping[str, str] | None = None,
):
    resolved = (
        dict(defaults)
        if isinstance(defaults, Mapping)
        else dict(SINGLE_THREAD_ENV_DEFAULTS)
    )
    applied = {}
    for key, value in resolved.items():
        applied[key] = os.environ.setdefault(str(key), str(value))
    return applied


configure_single_thread_env()


__all__ = [
    "SINGLE_THREAD_ENV_DEFAULTS",
    "configure_single_thread_env",
]
