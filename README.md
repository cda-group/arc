# Arc

[![](https://img.shields.io/badge/docs-online-brightgreen)](https://cda-group.github.io/arc/)
[![](https://img.shields.io/badge/report-online-brightgreen)](https://cda-group.github.io/arc/Arc-Report.pdf)

Programming language for data stream analysis.

## Requirements

OCaml (and dune), Rust (and cargo), and C++ (and CMake and Ninja).

## Examples

A basic streaming word-count application can be written in functional-style as follows:
```
val wordcounts = lines
  .flatmap(_.split(" "))
  .keyby(_)
  .window(
    length = 10min,
    stride = 3min
  )
  .count()
```

The same code can also be written using a more declarative, relational-style, syntax. This concept is borrowed from [Morel](https://github.com/julianhyde/morel) and applied to streaming data.

```
val wordcounts =
  from
    line in lines,
    word in line.split(" ")
  keyby word
  window
    length = 10min
    stride = 3min
  reduce count
    identity 1;
```

## Feature highlights

* Statically typed with global type inference.
* Parametric polymorphism (generics and rows) and ad-hoc polymorphism (type classes).
* Mix of functional syntax, imperative control-flow/mutation, and relational operators.
* Algebraic data types.
* First-class data streams.
* Complex event processing using tasks.
* Window-based computation.
* Low-level compilation and distributed execution.
* Command-line interface for data ingestion.

Note: All features have not yet been implemented :)

## Installation

```bash
git clone git@github.com:cda-group/arc.git
cd arc/
git submodule update --init --recursive
./build
```

## Documentation

* [Getting started](https://cda-group.github.io/arc/docs/getting-started.html)
* [Language Reference](https://cda-group.github.io/arc/docs/arc-lang/mod.md.html)
* [Developers](https://cda-group.github.io/arc/arc-lang/docs/arc-lang/contributing.html)

## Project Structure

* [`arc-lang`](https://github.com/cda-group/arc/tree/master/arc-lang) - A compiler for Arc-Lang.
* [`arc-mlir`](https://github.com/cda-group/arc/tree/master/arc-mlir) - An optimizer for Arc-Lang.
* [`arc-runtime`](https://github.com/cda-group/arc/tree/master/arc-runtime) - A local runtime which supports the execution of Arc-Lang programs.
* [`arc-python`](https://github.com/cda-group/arc/tree/master/arc-python) - A Python library for writing Arc-Lang applications.

## Related Projects

* [`arcon`](https://github.com/cda-group/arcon) - A distributed runtime which will support execution of Arc-Lang.
* [`kompact`](https://github.com/kompics/kompact) - A component-actor middleware which Arc-Runtime and Arcon are both implemented in.

## Other

> Arc-Lang ain't done until the fat lady sings. - Peter Van-Roy
