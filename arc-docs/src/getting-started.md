# Getting Started

This section explains how to get started with Arc-Lang.

## Prerequisites

The following dependencies are required to build Arc-Lang from source:

* CMake
* Clang
* Ninja
* Rust

### macOS

On macOS, dependencies can be installed with:

```
brew install cmake ninja clang
curl https://sh.rustup.rs -sSf | sh
```

### Ubuntu

On Ubuntu, dependencies can be installed with:

```bash
sudo apt install cmake ninja-build clang
curl https://sh.rustup.rs -sSf | sh
```

## Installation

To install Arc-Lang, clone the repo and run the build script:

```bash
git clone https://github.com/cda-group/arc/
cd arc
git submodule update --init --recursive
./build
```

The build script installs the `arc` command-line utility along with the arc-runtime library.

## Hello World

Arc-Lang files have the `.arc` file extension. For example:

```arc-lang
# hello-world.arc

def main() = print("Hello World")
```

To execute the above program, run:

```bash
arc run hello-world.arc
```
