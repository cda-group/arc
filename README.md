## Arc

[![](https://img.shields.io/badge/docs-online-brightgreen)](https://segeljakt.github.io/arc-website/arc-script/doc.html)
[![](https://img.shields.io/badge/shell-online-brightgreen)](https://cda-group.github.io/arc/)

A programming language for writing applications oriented around data streams and collections.

## Project Structure

* [`arc-script`](https://github.com/cda-group/arc/tree/master/arc-script) - A standalone user-facing surface language for Arc.
* [`arc-mlir`](https://github.com/cda-group/arc/tree/master/arc-mlir) - An intermediate representation built in [MLIR](https://mlir.llvm.org/) which `arc-script` programs translate into for optimisation.
* [`arcorn`](https://github.com/cda-group/arc/tree/master/arc-script/arc-script-api/arcorn) - A code generation library for different runtime backends which `arc-mlir` programs translate into for execution.
* [`arctime`](https://github.com/cda-group/arc/tree/master/arctime) - A local runtime targeted by `arcorn`.

## Related Projects

* [`arcon`](https://github.com/cda-group/arcon) - Distributed runtime targeted by `arcorn`.
* [`kompact`](https://github.com/kompics/kompact) - Distributed middleware which `arctime` and `arcon` are written in.

# Getting Started

```bash
git clone https://github.com/cda-group/arc.github

cd arc/

git submodule update --init --recursive

# Compile arc-script
(cd arc-script/; cargo build --release --workspace)

# Compile arc-mlir
(cd arc-mlir/; ./arc-mlir-build)

# Run tests
(cd arc-mlir/build/llvm-build/; ninja check-arc-mlir)
```
