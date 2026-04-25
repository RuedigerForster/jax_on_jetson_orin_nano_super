# JAX on Jetson Orin Nano Super

Build scripts for compiling **JAX 0.6.2 from source** on the NVIDIA Jetson Orin Nano Super
(JetPack 6.2 / L4T r36.4 / CUDA 12.6 / Ubuntu 22.04 / Python 3.12).

Pre-built wheels are not available for this platform — they crash at runtime due to
PJRT/CUDA plugin ABI incompatibility. This repo documents how to build them from source
and every obstacle encountered along the way.

**Result:** `jax==0.6.2.dev` with `backend: gpu` and `devices: [CudaDevice(id=0)]` verified.

---

## Platform

| | |
|---|---|
| Board | Jetson Orin Nano Super |
| JetPack | 6.2.1 |
| L4T | r36.4.7 |
| CUDA | 12.6 |
| cuDNN | 9.3.0 |
| Python | 3.12 |
| Ubuntu | 22.04 |

---

## Quick start

These scripts are designed to be used with [jetson-containers](https://github.com/dusty-nv/jetson-containers).
Place this directory at `packages/ml/jax/` in your jetson-containers checkout, then:

```bash
PYTHONPATH=. PYTHON_VERSION=3.12 python3 -m jetson_containers.build jax:0.6.2-builder
```

The build takes approximately **60–90 minutes** on the Orin Nano Super (4 parallel Bazel jobs).

### Running the container

```bash
docker run --runtime nvidia -it --rm \
  jax:0.6.2-r36.4.tegra-aarch64-cp312-cu126-22.04 \
  python3 -c "import jax; print(jax.devices())"
```

Expected output: `[CudaDevice(id=0)]`

---

## Build failures and fixes

Eleven build attempts were required. Every obstacle is documented here so you don't have to repeat them.

### 1. Pre-built pip wheels crash at runtime

**Symptom:** `pip install jax[cuda12]` succeeds but importing jax raises a PJRT plugin error.

**Cause:** The PyPI wheels bundle a CUDA runtime that conflicts with the Tegra CUDA stack.
The PJRT plugin ABI is not compatible with the system cuDNN.

**Fix:** Build from source. There is no alternative for this platform.

---

### 2. LLVM version — Clang 22 not supported by JAX 0.6.2

**Symptom:** Bazel fails with errors in XLA rules referencing unsupported Clang features.

**Cause:** JAX 0.6.2's XLA Bazel rules do not support Clang 22. The `llvm:22` base image
from jetson-containers installs Clang 22.

**Fix:** Use `llvm:19` as the base image (`depends: [llvm:19]` in `config.py`).
Set `--clang_path=/usr/lib/llvm-19/bin/clang` in the build flags.

---

### 3. Linker cannot find `-lstdc++`

**Symptom:**
```
/usr/bin/ld: cannot find -lstdc++: No such file or directory
clang: error: linker command failed with exit code 1
```

**Cause:** Clang automatically selects the **highest** GCC version found in
`/usr/lib/gcc/aarch64-linux-gnu/` (e.g. GCC 12), but only the lower version's
`libstdc++-dev` package was installed (e.g. `libstdc++-11-dev`). The linker symlink
`libstdc++.so` lives in the versioned GCC directory and only exists when that version's
`-dev` package is installed.

This is a common issue when CUDA packages install GCC 12 as a transitive dependency
while the system default remains GCC 11.

**Fix:** Install `libstdc++-dev` for **every** GCC version found in
`/usr/lib/gcc/aarch64-linux-gnu/`:

```bash
for GCC_VER in $(ls /usr/lib/gcc/aarch64-linux-gnu/ | grep -oE '^[0-9]+' | sort -u); do
    apt-get install -y --no-install-recommends libstdc++-${GCC_VER}-dev || true
done
```

Also use the **highest** installed GCC version for isystem header paths, to match
what Clang selects:

```bash
GCC_VER=$(ls /usr/lib/gcc/aarch64-linux-gnu/ | grep -oE '^[0-9]+' | sort -n | tail -1)
```

---

### 4. Bazel hermetic Python picks 3.10 instead of 3.12

**Symptom:** Build succeeds but produces `cp310` wheels, which don't install into the
Python 3.12 venv.

**Cause:** The build was launched without the `PYTHON_VERSION=3.12` environment variable.
Without it, the jetson-containers python layer defaults to 3.10, and Bazel's hermetic
toolchain matches it.

**Fix:** Always launch the build with:
```bash
PYTHONPATH=. PYTHON_VERSION=3.12 python3 -m jetson_containers.build jax:0.6.2-builder
```
And pass `--python_version` explicitly to `build.py`:
```bash
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
python3 build/build.py build ... --python_version="${PYTHON_VER}" ...
```

---

### 4. Dev wheel date mismatch breaks installation

**Symptom:**
```
jax-cuda12-plugin==0.6.2.dev20260424 depends on jax-cuda12-pjrt==0.6.2.dev20260424
but jax-cuda12-pjrt==0.6.2.dev20260425 is available
```

**Cause:** The build straddled midnight. `jaxlib` and `jax-cuda12-plugin` were compiled
on day N, while `jax-cuda12-pjrt` finished on day N+1. The plugin has a hard pin to the
exact same-date pjrt build.

**Fix:** Install all wheels with `--no-deps`. All wheels come from the same Bazel build
session and are guaranteed mutually compatible regardless of date stamp:

```bash
uv pip install --no-deps jaxlib*.whl jax_cuda12_plugin*.whl jax_cuda12_pjrt*.whl jax-*.whl
```

---

### 5. `uv pip install jax` from PyPI overrides the compiled jaxlib

**Symptom:** After installing the compiled wheels, `uv pip install jax` pulls jax 0.10.0
from PyPI, which depends on jaxlib 0.10.0 and replaces the compiled 0.6.2 wheel.

**Cause:** The `jax` pure-Python package on PyPI carries hard version constraints on
`jaxlib`. Dev builds of jaxlib are not published to PyPI, so any published `jax` version
will conflict.

**Fix:** Build the `jax` pure-Python wheel from the JAX source tree after the Bazel build
and install with `--no-deps`. Do not run `uv pip install jax` or `pip install jax`
after installing the compiled wheels.

---

## NvMapMemAllocInternalTagged warnings

After a successful GPU operation you may see:
```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
```

These are **benign**. They are a known quirk of the Tegra NvMap memory allocator and do
not affect computation correctness. JAX handles them transparently.

---

## Verification

```python
import jax
jax.print_environment_info()
print(jax.devices())          # [CudaDevice(id=0)]
print(jax.default_backend())  # gpu

import jax.numpy as jnp
from jax import random
key = random.PRNGKey(0)
a = random.normal(key, (4, 4))
b = jnp.dot(a, a.T)
print(b.device)               # cuda:0
```

---

## Attribution

Build scripts adapted from [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers).

Developed by **Rüdiger Forster** with [Claude Code](https://claude.ai/code) (Anthropic).

*Eleven build attempts, one Jetson, zero prior documentation.*
