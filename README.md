# easy-bupt-cuda-install

## About / 简介
A script used for installing CUDA 12.4 Toolkit on bupt-gpuhub(人人有算力) platform.

## Usage / 使用
```bash
sudo -i
git clone https://github.com/notvalid0/easy-bupt-cuda-install.git
#Clone失败请尝试gh-proxy加速
#git clone https://gh-proxy.org/https://github.com/notvalid0/easy-bupt-cuda-install.git
cd easy-bupt-cuda-install
chmod +x ./cuda-install.sh
bash ./cuda-install.sh
```

## Test / 测试
```bash
nvidia-smi
nvcc --version
ncu -v
```
