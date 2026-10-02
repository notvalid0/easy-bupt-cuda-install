
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# CUDA Research Environment Installer
# Target: Ubuntu 20.04 x86_64 + NVIDIA RTX 4090D
# CUDA Toolkit: 12.4
# Driver: preserve the existing working driver
# ============================================================

CUDA_VERSION="12.4"
CUDA_MAJOR_MINOR="12-4"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64"
KEYRING_DEB="/tmp/cuda-keyring_1.1-1_all.deb"

log()  { echo -e "\n[INFO] $*"; }
warn() { echo -e "\n[WARN] $*" >&2; }
die()  { echo -e "\n[ERROR] $*" >&2; exit 1; }

[[ "${EUID}" -eq 0 ]] || die "请使用 sudo bash $0 运行"

# 1. Check operating system
source /etc/os-release
[[ "${ID}" == "ubuntu" && "${VERSION_ID}" == "20.04" ]] \
    || die "此脚本针对 Ubuntu 20.04，当前为 ${PRETTY_NAME}"

[[ "$(dpkg --print-architecture)" == "amd64" ]] \
    || die "此脚本仅支持 x86_64 / amd64"

log "系统版本：${PRETTY_NAME}"
log "内核版本：$(uname -r)"

# 2. Check the existing NVIDIA driver.
# Do not install or replace drivers automatically on remote servers.
command -v nvidia-smi >/dev/null 2>&1 \
    || die "未检测到 nvidia-smi。请先确认 NVIDIA 驱动已安装并正常工作。"

nvidia-smi || die "NVIDIA 驱动无法正常访问 GPU"

GPU_INFO="$(nvidia-smi --query-gpu=name,compute_cap \
    --format=csv,noheader 2>/dev/null || true)"

log "GPU 信息："
echo "${GPU_INFO:-无法读取 GPU 名称和 Compute Capability}"

# 3. Install system dependencies
log "安装基础开发依赖"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
    ca-certificates \
    wget \
    curl \
    gnupg \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    git \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

# Kernel headers are only needed to build kernel modules (DKMS),
# e.g. when installing the NVIDIA driver from source.
# This script installs toolkit-only packages and does not touch the driver,
# so a missing header package must not abort the whole install.
HEADERS_PKG="linux-headers-$(uname -r)"

if apt-cache show "${HEADERS_PKG}" >/dev/null 2>&1; then
    log "安装内核头文件 ${HEADERS_PKG}"
    apt-get install -y "${HEADERS_PKG}" \
        || warn "安装 ${HEADERS_PKG} 失败，但 CUDA Toolkit 不依赖它，继续。"
else
    warn "软件源中没有 ${HEADERS_PKG}，跳过内核头文件安装。"
    warn "当前内核 $(uname -r) 的头文件不在 apt 源里（focal 已停止更新，5.15 HWE 源里最高只有 5.15.0-139/140~20.04.1）。"
    warn "nvcc / ncu / CUDA Toolkit 不需要内核头文件；如需为当前内核编译模块，请单独安装匹配版本的头文件。"
fi

# 4. Configure NVIDIA CUDA repository
log "配置 NVIDIA CUDA 软件源"

wget -q "${CUDA_REPO}/cuda-keyring_1.1-1_all.deb" \
    -O "${KEYRING_DEB}"

dpkg -i "${KEYRING_DEB}"
apt-get update

# 5. Install CUDA Toolkit and Nsight Compute.
# Install toolkit-only packages to avoid replacing the driver.
log "安装 CUDA Toolkit ${CUDA_VERSION}"

apt-get install -y \
    "cuda-toolkit-${CUDA_MAJOR_MINOR}"

log "安装 Nsight Compute"

# Package naming can differ between repository revisions.
# Try the versioned package first, then the generic package.
if apt-cache show "cuda-nsight-compute-${CUDA_MAJOR_MINOR}" \
    >/dev/null 2>&1; then
    apt-get install -y "cuda-nsight-compute-${CUDA_MAJOR_MINOR}"
elif apt-cache show cuda-nsight-compute >/dev/null 2>&1; then
    apt-get install -y cuda-nsight-compute
else
    warn "APT 未找到独立的 Nsight Compute 包。"
    warn "请检查 NVIDIA 软件源中的实际包名；不会自动升级驱动。"
fi

# 6. Configure environment variables system-wide
log "配置 CUDA 环境变量"

cat > /etc/profile.d/cuda-12-4.sh <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"

if [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
EOF

chmod 644 /etc/profile.d/cuda-12-4.sh

# Make CUDA visible to the current installation shell.
export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 7. Verify the toolkit
log "检查 nvcc"

[[ -x "${CUDA_HOME}/bin/nvcc" ]] \
    || die "没有找到 ${CUDA_HOME}/bin/nvcc"

"${CUDA_HOME}/bin/nvcc" --version

# Find ncu if it is not directly in PATH.
if ! command -v ncu >/dev/null 2>&1; then
    for candidate in \
        "${CUDA_HOME}/bin/ncu" \
        /usr/local/bin/ncu \
        /opt/nvidia/nsight-compute/ncu; do
        if [[ -x "${candidate}" ]]; then
            export PATH="$(dirname "${candidate}"):${PATH}"
            break
        fi
    done
fi

if command -v ncu >/dev/null 2>&1; then
    log "Nsight Compute 版本"
    ncu --version
else
    warn "当前 PATH 未找到 ncu，请检查 Nsight Compute 安装位置。"
fi

# 8. Compile and execute a CUDA smoke test
log "编译 CUDA 测试程序"

TEST_DIR="$(mktemp -d)"
trap 'rm -rf "${TEST_DIR}"' EXIT

cat > "${TEST_DIR}/smoke.cu" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_one(int *x) {
    *x += 1;
}

int main() {
    int *d = nullptr;
    int h = 41;

    cudaError_t err = cudaMalloc(&d, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D: %s\n", cudaGetErrorString(err));
        cudaFree(d);
        return 1;
    }

    add_one<<<1, 1>>>(d);

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }

    if (err == cudaSuccess) {
        err = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(d);

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA smoke test: result = %d\n", h);
    return h == 42 ? 0 : 1;
}
CU

"${CUDA_HOME}/bin/nvcc" \
    -O2 \
    -arch=sm_89 \
    "${TEST_DIR}/smoke.cu" \
    -o "${TEST_DIR}/smoke"

"${TEST_DIR}/smoke"

log "安装完成"
echo "CUDA_HOME=${CUDA_HOME}"
echo "请新开一个 SSH 会话，或执行：source /etc/profile.d/cuda-12-4.sh"
echo "检查命令：nvcc --version"
echo "检查 GPU：nvidia-smi"
echo "检查 NCU：ncu --version"
