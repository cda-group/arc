# Arc

[![](https://img.shields.io/badge/docs-online-brightgreen)](https://cda-group.github.io/arc/)
[![](https://img.shields.io/badge/report-online-brightgreen)](https://cda-group.github.io/arc/Arc-Report.pdf)

Programming language for data stream analysis.

## Requirements

OCaml (and dune), Rust (and cargo), and C++ (and CMake and Ninja).

## Examples

A streaming word-count application can be implemented in Arc-Lang as follows.

```
def main() =
    from line: String in source(topic: "text"),
         word in line.split(" ") {
        group word
        window count as w {
            length 10min
            step 3min
            compute count
        }
        select {word, w.count}
        into sink(topic: "wordcount")
    }
```

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
* [`arc-sys`](https://github.com/cda-group/arc/tree/master/arc-sys) - A distributed system for executing Arc-Lang programs.

## Other

> Arc-Lang ain't done until the fat lady sings. - Peter Van-Roy
