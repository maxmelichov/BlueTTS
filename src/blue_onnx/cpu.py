"""CPU defaults: how many ONNX Runtime threads to use, and the int8 vector estimator.

Two measured problems on x86 CPUs (see the README's Speed section):

- ``os.cpu_count()`` reports the host's CPUs, not the ones this process may use. In a container limited
  to 2 CPUs on a 16-thread host the old default (``min(8, cpu_count)``) ran 8 threads on 2 CPUs and was
  3x slower than 2 threads. :func:`available_cpus` honours the CPU affinity mask and the cgroup quota.
- The vector estimator (one call per flow step, about 80% of the time) is matrix multiplies. Dynamic int8
  quantization of those runs about 2x faster on x86 CPUs (and ARM Linux), with the same ASR error and
  speaker similarity as float32. The vocoder is left in float32: quantizing it lowered speaker similarity.
"""

import hashlib
import math
import os
import platform
from typing import Optional

INT8_VECTOR_ESTIMATOR = "vector_estimator.int8.onnx"
PRECISIONS = ("auto", "fp32", "int8")
MAX_DEFAULT_THREADS = 8


def cgroup_cpu_limit(root: str = "/sys/fs/cgroup") -> Optional[int]:
    """CPUs granted by a cgroup CPU quota (``docker --cpus``, Kubernetes limits, hosted containers), rounded
    up, or ``None`` when there is no quota (or no cgroup filesystem, as on macOS and Windows)."""
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        with open(os.path.join(root, "cpu.max")) as f:
            quota, period = f.read().split()[:2]
        if quota != "max" and int(period) > 0:
            return max(1, math.ceil(int(quota) / int(period)))
        return None
    except (OSError, ValueError):
        pass
    try:  # cgroup v1: quota is -1 when unlimited
        with open(os.path.join(root, "cpu", "cpu.cfs_quota_us")) as f:
            quota = int(f.read())
        with open(os.path.join(root, "cpu", "cpu.cfs_period_us")) as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """Logical CPUs this process can actually run on: the affinity mask, capped by a cgroup quota."""
    n = None
    if hasattr(os, "process_cpu_count"):  # Python 3.13+: affinity-aware
        n = os.process_cpu_count()
    if n is None and hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = None
    n = n or os.cpu_count() or 1
    limit = cgroup_cpu_limit()
    return min(n, limit) if limit else n


def default_num_threads() -> int:
    """ONNX Runtime intra-op threads: ``ORT_NUM_THREADS`` if set, else ``min(8, available_cpus())``."""
    env = os.environ.get("ORT_NUM_THREADS", "").strip()
    if env:
        return max(1, int(env))
    return min(MAX_DEFAULT_THREADS, available_cpus())


def _int8_helps(machine: Optional[str] = None, system: Optional[str] = None) -> bool:
    machine = (machine if machine is not None else platform.machine()).lower()
    system = (system if system is not None else platform.system()).lower()
    if machine in ("x86_64", "amd64", "x64", "i686", "x86"):
        return True
    # ARM Linux (Docker on an M5, 4 CPUs): int8 2.5x faster. macOS on the same chip: no faster, because
    # ONNX Runtime's float32 matmuls there already use the Apple matrix unit.
    return machine in ("aarch64", "arm64") and system != "darwin"


def resolve_precision(precision: Optional[str], providers: Optional[list], machine: Optional[str] = None,
                      system: Optional[str] = None) -> str:
    """``"fp32"`` or ``"int8"`` for the vector estimator.

    ``"auto"`` (the default; ``BLUE_ONNX_PRECISION`` overrides it) picks int8 only where it was measured to
    help: the plain ONNX Runtime CPU provider on x86 or on ARM Linux. macOS keeps float32 (int8 was no
    faster on Apple Silicon), and GPU or OpenVINO providers get the float32 graph.
    """
    p = (precision or os.environ.get("BLUE_ONNX_PRECISION") or "auto").lower()
    if p not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if p != "auto":
        return p
    cpu_only = not providers or all(
        (x[0] if isinstance(x, (tuple, list)) else x) == "CPUExecutionProvider" for x in providers
    )
    return "int8" if cpu_only and _int8_helps(machine, system) else "fp32"


def with_thread_options(providers: list, n_threads: int) -> list:
    """Give OpenVINO the same thread count as the CPU provider. It ignores ``intra_op_num_threads`` and
    otherwise starts one thread per visible CPU: in a 2-CPU container on a 16-thread host the vector
    estimator ran at 640 ms per step that way and at 300 ms with ``num_of_threads=2``. Options the caller
    passed in a ``(name, options)`` tuple are kept."""
    out = []
    for p in providers:
        if p == "OpenVINOExecutionProvider":
            p = (p, {"num_of_threads": str(n_threads)})
        elif isinstance(p, (tuple, list)) and p[0] == "OpenVINOExecutionProvider":
            p = (p[0], {"num_of_threads": str(n_threads), **dict(p[1])})
        out.append(p)
    return out


def _cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "blue_onnx")


def quantize_vector_estimator(src: str, dst: str) -> str:
    """Write a dynamic-int8 copy of the vector estimator graph ``src`` to ``dst``.

    MatMul weights only, int8 per output channel with ``reduce_range`` (7-bit weights): x86 CPUs without
    VNNI (AVX2 only) accumulate u8 x s8 products in int16 and saturate on full-range weights, which tripled
    the error in a test under AVX2; the reduced range keeps it at the VNNI / ARM level. Takes a few seconds.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = f"{dst}.tmp{os.getpid()}"
    try:
        quantize_dynamic(
            src, tmp, op_types_to_quantize=["MatMul", "Gemm"], weight_type=QuantType.QInt8,
            per_channel=True, reduce_range=True, extra_options={"MatMulConstBOnly": True},
        )
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst


def int8_vector_estimator_path(onnx_dir: str, src: str, create: bool = True) -> Optional[str]:
    """Path of the int8 vector estimator for a bundle: ``<onnx_dir>/vector_estimator.int8.onnx`` if the bundle
    ships one, else a cached copy (made from ``src`` on first use, in the bundle when it is writable, otherwise
    under ``~/.cache/blue_onnx``). ``None`` if it does not exist and ``create`` is false or quantizing fails."""
    shipped = os.path.join(onnx_dir, INT8_VECTOR_ESTIMATOR)
    if os.path.exists(shipped):
        return shipped
    st = os.stat(src)
    # str hash() is salted per process; the cache key must be stable across runs
    key = f"{os.path.realpath(src)}|{st.st_size}|{int(st.st_mtime)}".encode()
    tag = hashlib.sha1(key).hexdigest()[:12]
    cached = os.path.join(_cache_dir(), "int8", tag, INT8_VECTOR_ESTIMATOR)
    if os.path.exists(cached):
        return cached
    if not create:
        return None
    target = shipped if os.access(onnx_dir, os.W_OK) else cached
    print(f"Quantizing the vector estimator to int8 (one-off, a few seconds) -> {target}")
    try:
        return quantize_vector_estimator(src, target)
    except Exception as e:  # read-only filesystem, missing onnx package, ...: fall back to float32
        print(f"[WARN] int8 quantization failed ({e}); using the float32 vector estimator.")
        return None
