+++
title = "Introduction"
description = "Introducing Arc-Lang, a programming language for building data science applications."
date = 2022-03-17
updated = 2022-03-17
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["segeljakt"]

[extra]
lead = ""
+++

# Arc-Lang

This blog post gives an introduction to **Arc-Lang** - a programming language for big-data analytics - and sheds light on its vision and implementation. The Arc-Lang research project is funded by [SSF](https://strategiska.se/en/) and has gone through many revisions prior to this blog post. This history will be discussed in a later post.

## Introduction

Data analytics pipelines are becoming increasingly more complicated due to the growing number of requirements imposed by data science. Not only must data be processed and analyzed scalably with respect to its volume and velocity, but also intricately by involving many different types of data. Arc-Lang is a programming language for data analytics that supports parallel operations over multiple data types including datastreams and dataframes. As an example, a basic word-count application can be implemented as follows in Arc-Lang:

```
val lines = read_stdin();

val word_counts =
    from line in lines,
         word in line.split(" ")
    yield #{word, count:1}
    group word
      window 10min
      reduce (+) of count
      identity 0;

write_stdout(word_counts);
```

The goal of Arc-Lang is to make big-data analytics easy. Arc-Lang targets streaming analytics (i.e., processing data continuously as it is being generated) and batch analytics (i.e., processing data in large chunks all-at-once). From the streaming-perspective, Arc-Lang must be able to manage data at a fine granularity that is generated by many types of sensors, arriving at varying rates, in different formats, sizes, qualities, and possibly out-of-order. Datastreams can in addition be massive in numbers, ranging into the billions, due to the plethora of data sources that have emerged in the recent IoT boom. From the batch-perspective, Arc-Lang must be able to handle different kinds of collection-based data types whose sizes can scale to massive sizes, e.g., tensors and dataframes.

To cope with the requirements of batch and stream data management, a runtime system is needed which can exploit distributed programming to enable scalability through partitioning and parallelism. Distributed programming is however difficult without abstraction. Application developers must manage problems such as fault tolerance, exactly-once-processing, and coordination while considering tradeoffs in security and efficiency. To this end, distributed systems leverage high-level DSLs which are more friendly towards end-users. DSLs in the form of query languages, frameworks, and libraries allow application developers to focus on domain-specific problems, such as the development of algorithms, and to disregard engineering-related issues. In addition, DSLs that are intermediate languages have been adopted by multiple systems both as a solution to enable reuse by breaking the dependence between the user and runtime, and to enable target-independent optimisation. There is always a tradeoff that must be faced in DSL design. DSLs make some problems easier to solve at the expense of making other problems harder to solve. How a DSL is implemented can also have an impact on its ability to solve problems.

<embed src="/images/DSL-Hierarchy.jpg" width="100%" height="100%">

Different categories of DSLs are highlighted in the figure above.

### Approach

In relation to other DSLs, Arc-Lang is a standalone compiled DSL implemented in OCaml. The idea of Arc-Lang's is to combine general purpose imperative and functional programming over *small data* with declarative programming over *big data*. As an example, it should be possible to perform both fine-grained processing over individual data items of a datastream, while also being able to compose pipelines of relational operations through SQL-style queries. Arc-Lang is statically typed for the purpose of performance and safety, but at the same time also inferred and polymorphic to enable ease of use and reuse.

The approach of implementing the language as a standalone DSL allows for more creative freedom in the language design. At the same time, this approach requires everything, including optimisations and libraries, to be implemented from scratch.

To address the issue of optimisation, we are using the [MLIR](https://mlir.llvm.org/) compiler framework to implement Arc-MLIR - an intermediate language - which Arc-Lang programs translate into for optimisations. MLIR defines a universal intermediate language which can be extended with custom dialects. A dialect includes a set of operations, types, type rules, analyses, rewrite rules (to the same dialect), and lowerings (to other dialects). All dialects adhere to the same meta-syntax and meta-semantics which allows them to be interweaved in the same program code. The MLIR framework handles parsing, type checking, line information tracking among other things. Additionally, MLIR provides tooling for testing, parallel compilation, documentation, CLI usage, etc. The plan is to extend Arc-MLIR with custom domain-specific optimisations for the declarative part of Arc-Lang and to capitalize on MLIR's ability to derive general-purpose optimisations such as constant propagation for Arc-Lang's functional and imperative side.

To address the shortcoming of libraries, Arc-Lang allows both types and functions to be defined externally (inside Rust) and imported into the language. Most of the external functionality is encapsulated inside a runtime library named Arc-Runtime. Arc-Runtime builds on the [kompact](https://github.com/kompics/kompact) Component-Actor framework to provide distributed abstractions.

## Summary

In summary, Arc-Lang as a whole consists of three parts:

* **Arc-Lang**: A high-level programming language for big data analytics.
* **Arc-MLIR**: An intermediate language for optimising Arc-Lang.
* **Arc-Runtime**: A distributed runtime for executing Arc-Lang.

