Below are instructions for setting up a custom runner, running in a Docker container, for GitHub actions.

```bash
# Setup docker

docker pull ubuntu:16.04
docker run -i -t ubuntu:16.04 /bin/bash

# Install apt dependencies

apt update && apt upgrade -y
apt install -y git vim curl z3 libz3-dev curl libssl-dev gcc pkg-config make ninja-build python zip openjdk-8-jdk python-software-properties software-properties-common
add-apt-repository ppa:git-core/ppa -y
apt update && apt upgrade -y

# Install Rust

curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
echo 'source $HOME/.cargo/env' >> ~/.bashrc
rustup target add wasm32-unknown-unknown
cargo install mdbook

# Install Cmake

cd ~/
curl -L https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1.tar.gz --output cmake.tar.gz
tar -xf cmake.tar.gz
cd ~/cmake
./bootstrap
make
make install

# Install LLVM

curl -L https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz --output llvm.tar.xz
tar -xf llvm.tar.xz
cd llvm
export PATH=~/llvm/bin:$PATH
export LD_LIBRARY_PATH=~/llvm/lib:$LD_LIBRARY_PATH
echo 'export PATH=~/llvm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/llvm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Install SBT / Scala / Java

curl https://piccolo.link/sbt-1.3.13.zip -L --output sbt.zip
unzip sbt.zip
export PATH=~/sbt/bin:$PATH
echo 'export PATH=~/sbt/bin:$PATH' >> ~/.bashrc

# Install GitHub Actions Runner

# Follow this tutorial (Which generates a unique token):
#
#     https://github.com/cda-group/arc/settings/actions/add-new-runner?arch=x64&os=linux
#
# NOTE: You need to run this
export RUNNER_ALLOW_RUNASROOT=1

# Setup Runner

./run.sh &
disown <PID>

# To exit the container: <C-p><C-q>

# [OPTIONAL] Check that everything builds

cd ~
git clone https://github.com/cda-group/arc
cd ~/arc
git checkout mlir
git submodule update --init --recursive

# Check if arc-script builds

cd ~/arc/arc-script
cargo check --all-features --tests --bins --examples --benches

# Check if arc-mlir builds

cd ~/arc/arc-mlir/
./arc-mlir-build
```
