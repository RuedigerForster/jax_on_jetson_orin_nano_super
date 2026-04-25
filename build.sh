#!/usr/bin/env bash
# JAX builder for Jetson (architecture: ARM64, CUDA support)
set -ex

echo "Building JAX for Jetson"

# libstdc++-dev provides the linker symlink (libstdc++.so) needed by Clang/Bazel.
# Clang auto-selects the highest GCC version it finds in /usr/lib/gcc/, not necessarily
# the default g++. The linker symlink lives in the versioned GCC dir and only exists
# when that version's -dev package is installed. Install dev packages for every GCC
# version present so whichever one Clang picks will have its libstdc++.so symlink.
apt-get update -qq
for _GCC_DEV_VER in $(ls /usr/lib/gcc/aarch64-linux-gnu/ 2>/dev/null | grep -oE '^[0-9]+' | sort -u); do
    apt-get install -y --no-install-recommends libstdc++-${_GCC_DEV_VER}-dev || true
done
rm -rf /var/lib/apt/lists/*

# Clone JAX repository
# Note: JAX versions are typically v0.4.x. If v0.9.0 doesn't exist, this falls back to main.
git clone --branch "jax-v${JAX_BUILD_VERSION}" --depth=1 --recursive https://github.com/google/jax /opt/jax || \
git clone --depth=1 --recursive https://github.com/google/jax /opt/jax

cd /opt/jax

mkdir -p /opt/jax/wheels/

# Initialize flags
BUILD_FLAGS="--clang_path=/usr/lib/llvm-19/bin/clang --output_path=/opt/jax/wheels/ "

if [ "${IS_SBSA}" -eq 1 ]; then
    echo "Building for SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87,sm_89,sm_90,sm_100,sm_110,sm_120,sm_121" '
    BUILD_FLAGS+='--cuda_version=13.2.1 --cudnn_version=9.21.0 '

    # Bazel's rules_ml_toolchain detects Tegra by checking `uname -a` for "tegra"
    # (e.g. kernel 6.8.12-tegra), then constructs cuda13_tegra-aarch64-unknown-linux-gnu
    # as the cuDNN platform — which doesn't exist. Jetson Thor IS SBSA but the kernel
    # carries the tegra name. Replace uname so the rules select the SBSA platform
    # (cuda13_aarch64-unknown-linux-gnu) instead.
    cp /usr/bin/uname /usr/bin/uname.real
    printf '#!/bin/sh\n/usr/bin/uname.real "$@" | sed s/tegra/sbsa/g\n' > /usr/bin/uname
    chmod +x /usr/bin/uname

    BUILD_FLAGS+='--bazel_options=--repo_env=CUDA_REDIST_TARGET_PLATFORM=aarch64 '

    # --- BAZEL CONFIGURATION ---
    # 2. Fix Abseil: Disable nullability attributes.
    #    Clang 20+ enables them, but Abseil's headers place them incorrectly, causing syntax errors.
    BUILD_FLAGS+='--bazel_options=--copt=-DABSL_HAVE_NULLABILITY_ATTRIBUTES=0 '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-DABSL_HAVE_NULLABILITY_ATTRIBUTES=0 '
    BUILD_FLAGS+='--bazel_options=--copt=-D_Nullable= '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D_Nullable= '
    BUILD_FLAGS+='--bazel_options=--copt=-D_Nonnull= '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D_Nonnull= '

    # 3. Fix Protobuf: Polyfill the missing __is_bitwise_cloneable builtin.
    #    We alias it to __is_trivially_copyable which is supported.
    BUILD_FLAGS+='--bazel_options=--copt=-D__is_bitwise_cloneable=__is_trivially_copyable '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D__is_bitwise_cloneable=__is_trivially_copyable '

    # 4. Fix Abseil: Polyfill the missing __builtin_is_cpp_trivially_relocatable builtin.
    #    NVCC/Clang interaction causes this builtin to be undefined despite passing feature checks.
    #    We alias it to __is_trivially_copyable, which is a safe fallback for build purposes.
    BUILD_FLAGS+='--bazel_options=--copt=-D__builtin_is_cpp_trivially_relocatable=__is_trivially_copyable '
    BUILD_FLAGS+='--bazel_options=--cxxopt=-D__builtin_is_cpp_trivially_relocatable=__is_trivially_copyable '

else
    echo "Building for non-SBSA architecture"
    BUILD_FLAGS+='--cuda_compute_capabilities="sm_87" '
    # derive CUDA version (ensure x.y.z format) and cuDNN version from the system headers
    CUDA_VER_FULL="${CUDA_VERSION}"
    [ "$(echo "${CUDA_VERSION}" | tr '.' '\n' | wc -l)" -lt 3 ] && CUDA_VER_FULL="${CUDA_VERSION}.0"
    CUDNN_MAJOR=$(grep "^#define CUDNN_MAJOR" /usr/include/cudnn_version.h | awk '{print $3}')
    CUDNN_MINOR=$(grep "^#define CUDNN_MINOR" /usr/include/cudnn_version.h | awk '{print $3}')
    CUDNN_PATCH=$(grep "^#define CUDNN_PATCHLEVEL" /usr/include/cudnn_version.h | awk '{print $3}')
    CUDNN_VER_FULL="${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCH}"
    BUILD_FLAGS+="--cuda_version=${CUDA_VER_FULL} --cudnn_version=${CUDNN_VER_FULL} "
fi

# Clang needs to find GCC's C++ stdlib headers (cstdint, cstddef, etc.) — add them
# explicitly for both target and host/exec tool compilations. Without this, Bazel's
# crosstool wrapper fails to locate system C++ headers in either context.
# Use the highest installed GCC version to match what Clang will auto-select.
GCC_VER=$(ls /usr/lib/gcc/aarch64-linux-gnu/ 2>/dev/null | grep -oE '^[0-9]+' | sort -n | tail -1)
[ -z "${GCC_VER}" ] && GCC_VER=$(g++ -dumpversion | cut -d. -f1)
BUILD_FLAGS+="--bazel_options=--cxxopt=-isystem/usr/include/c++/${GCC_VER} "
BUILD_FLAGS+="--bazel_options=--cxxopt=-isystem/usr/include/aarch64-linux-gnu/c++/${GCC_VER} "
BUILD_FLAGS+="--bazel_options=--host_cxxopt=-isystem/usr/include/c++/${GCC_VER} "
BUILD_FLAGS+="--bazel_options=--host_cxxopt=-isystem/usr/include/aarch64-linux-gnu/c++/${GCC_VER} "

# Clang's GCC-detection may fail on LLVM-only images, so the linker won't auto-add
# /usr/lib/aarch64-linux-gnu to its search path. Pass it explicitly so -lstdc++ resolves.
BUILD_FLAGS+='--bazel_options=--linkopt=-L/usr/lib/aarch64-linux-gnu '
BUILD_FLAGS+='--bazel_options=--host_linkopt=-L/usr/lib/aarch64-linux-gnu '

# Run the build
# Note: $BUILD_FLAGS is unquoted to allow word splitting of the individual bazel arguments
BUILD_FLAGS+='--bazel_options=--jobs=4 '
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
python3 build/build.py build $BUILD_FLAGS --python_version="${PYTHON_VER}" --wheels=jaxlib,jax-cuda-plugin,jax-cuda-pjrt,jax

# Upload the wheels to mirror
CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
twine upload --verbose /opt/jax/wheels/jaxlib-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax_cuda${CUDA_MAJOR}_pjrt-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax_cuda${CUDA_MAJOR}_plugin-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/jax/wheels/jax-*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"

# Install them into the container.
# --no-deps bypasses cross-wheel version pins (dev builds can straddle midnight,
# giving plugin and pjrt different date stamps despite being from the same build).
# All four wheels (jaxlib, plugin, pjrt, jax) are built together so they are
# guaranteed to be mutually compatible.
cd /opt/jax/wheels/
uv pip install --no-deps jaxlib*.whl jax_cuda${CUDA_MAJOR}_plugin*.whl jax_cuda${CUDA_MAJOR}_pjrt*.whl jax-*.whl
uv pip install opt_einsum
cd /opt/jax
