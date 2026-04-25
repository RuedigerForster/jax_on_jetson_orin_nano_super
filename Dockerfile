#---
# name: jax
# group: jax
# config: config.py
# depends: [cuda, cudastack:standard, numpy, onnx, llvm:19]
# test: test.py
# docs: Containers for JAX with CUDA support.
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# set the CUDA architectures that JAX extensions get built for
# set the JAX cache directory to mounted /data volume
ARG JAX_CUDA_ARCH_ARGS \
    JAX_VERSION \
    JAX_BUILD_VERSION \
    ENABLE_NCCL \
    CUDA_VERSION \
    IS_SBSA \
    FORCE_BUILD=off

ENV JAX_CUDA_ARCH_LIST=${JAX_CUDA_ARCH_ARGS} \
    JAX_CACHE_DIR=/data/models/jax

# copy installation and build scripts for JAX
COPY install.sh link_cuda.sh /tmp/JAX/

# attempt to install JAX from pip first (fast path)
RUN /tmp/JAX/install.sh && echo "INSTALLED_FROM_PIP=1" > /tmp/JAX/install_method || echo "INSTALLED_FROM_PIP=0" > /tmp/JAX/install_method

# copy build script separately so changes to it only invalidate the build layer,
# not the pip-install layer above
COPY build.sh /tmp/JAX/

# fall back to building from source if pip install failed
RUN grep -q "INSTALLED_FROM_PIP=1" /tmp/JAX/install_method || /tmp/JAX/build.sh
