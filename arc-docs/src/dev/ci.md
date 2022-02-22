Below are instructions for setting up a custom runner, running in a Docker container, for GitHub actions.

```bash
# Setup docker

docker pull ubuntu:18.04
docker run -i -t ubuntu:18.04 /bin/bash

# Setup user

passwd # change root password
adduser arc-runner sudo
su -l arc-runner

# Install apt dependencies

sudo add-apt-repository ppa:git-core/ppa -y
sudo apt update && apt upgrade -y
sudo apt install -y git vim curl z3 libz3-dev curl libssl-dev gcc pkg-config make ninja-build python zip openjdk-8-jdk software-properties-common texlive-xetex latexmk gettext ccache

# Install Rust

curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
echo 'source $HOME/.cargo/env' >> ~/.bashrc
rustup toolchain add nightly
rustup target add wasm32-unknown-unknown
rustup default nightly
cargo install mdbook
cargo install sccache

# Install Cmake

cd ~/
curl -L https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1.tar.gz --output cmake.tar.gz
tar -xf cmake.tar.gz
cd cmake-3.18.1/
./bootstrap
make
make install

# Install LLVM

curl -L https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz --output llvm.tar.xz
tar -xf llvm.tar.xz
mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 llvm
cd llvm
export PATH=~/llvm/bin:$PATH
export LD_LIBRARY_PATH=~/llvm/lib:$LD_LIBRARY_PATH
echo 'export PATH=~/llvm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=~/llvm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Install OCaml

sudo bash -c "sh <(curl -fsSL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)"
opam init # Make sure to disable sandboxing
eval $(opam env)
opam switch create 4.13.1
eval $(opam env)
opam install core
opam install dune
opam install menhir

# Install GitHub Actions Runner

# Follow this tutorial (Which generates a unique token):
#     https://github.com/cda-group/arc/settings/actions/runners/new?arch=x64&os=linux

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

# Check if arc-lang builds

cd ~/arc/arc-lang
cargo check --all-features --tests --bins --examples --benches

# Check if arc-mlir builds

cd ~/arc/arc-mlir/
./arc-mlir-build
```
