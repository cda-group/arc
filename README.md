<h1 align="center">The Arc Programming Language</h1>

[![Build Status](https://travis-ci.org/cda-group/arc.svg?branch=master)](https://travis-ci.org/cda-group/arc/)

**Arc** is a scripting language with an [*intermediate representation*](https://en.wikipedia.org/wiki/Intermediate_representation) for expressing transformations over batch and streaming data. The output of the Arc compiler is a *dataflow graph* that can be deployed on distributed stream processing runtimes. While the goal is to be runtime-agnostic, Arc is primarily intended to run on top of [Arcon](https://github.com/cda-group/arcon) - a native Rust-based runner.

This repository is divided into two parts:

* A *front-end* compiler for Arc implemented in Scala, which takes care of
  * Lexing/Parsing through [ANTLR](https://www.antlr.org/).
  * Macro expansion/Name resolution.
  * Type inference.
  * [MLIR](https://github.com/tensorflow/mlir) code generation.
* A *middle-end* optimizer, implemented in C++ using [MLIR](https://github.com/tensorflow/mlir), which (will) take care of:
  * Standard compiler optimizations.
  * Dataflow optimizations.
  * Domain-specific optimizations.
  * Hardware-acceleration.
  * Dataflow code generation.

More information about the project can be found [here](https://cda-group.github.io/).

# Getting Started

## macOS / Linux

Assuming Scala and `sbt` are installed, the following clones and builds the project.

```bash
git clone https://github.com/cda-group/arc.github

cd arc/

git submodule update --init --recursive

# Compile arc-script
cd arc-script/; cargo build --release --workspace; cd -

# Compile arc-mlir
cd arc-mlir/; ./arc-mlir-build; cd -

# Run tests
cd arc-mlir/build/llvm-build/; ninja check-arc-mlir; cd -
```

# Documentation

[Documentation](www.github.com/cda-group/arc)

<p align="center">
  <img src="https://github.com/segeljakt/assets/raw/master/arc-script.png">
</p>
