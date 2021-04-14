# Discussion

This section contains some extra discussion behind the problem and motivation of Arc-Script.

## Why Arc-Script?

Arc-Script is a high-level language for data analytics which under-the-hood translates into [Arcon](https://github.com/cda-group/arcon). Arcon is a distributed system for stream processing and data warehousing which is implemented in Rust. Similar to other streaming systems (e.g., [Flink](https://flink.apache.org/) and [Timely](https://github.com/TimelyDataflow/timely-dataflow)) Arcon provides a *framework* in Rust - a general purpose programming language. Since anything which is a value in Rust can be hidden behind a function abstraction, a valid argument against Arc-Script is *"Why not just use Arcon and Rust"*?. Rust particularly has the following benefits:

* Rust is *flexible* by being able to express general computations.
* Rust is *fast* by compiling into efficient machine code.
* Rust is *safe* by preventing memory-management errors.

## Core Problems

A core problem is that Rust was not designed for the specific problem of data analytics. Rust exposes constructs which a language for data analytics does not benefit from and should ideally not have. Notably, Rust has references, lifetimes, affine types, and low-level memory management. While these features give programmers control over *how* to solve problems (which may lead to more efficient solutions), they have less impact on *what* problems can be solved. Data analysts may therefore get stuck on trying to optimise solutions instead of solving problems. Optimisation should ideally be the job of a compiler and not a human. Since these features are at the core of Rust, any abstractions built on top of them within Rust are inevitably going to be leaking their complexity as well. In other words, Rust cannot hide the fact that it has references, affine types, and other details. This phenomena is by-design since Rust programs should be efficient even at high levels of abstraction.

The second problem is that Arcon is a framework written and exposed in Rust and cannot therefore solve the previous problem. The only solution to completely decouple program specification from execution is to develop a language which compiles into Rust. Note that these problems are not just exclusive to Arcon, but apply more generally to all DSLs which are exposed as frameworks. The reason why frameworks such as Flink have been successful is that they are embedded in languages whose features more closely resemble what is expected of a language for data analytics. Notably, Scala has briefer syntax and by-value semantics. Furthermore, embedding DSLs as frameworks only makes sense if the framework is written in the same language as the system's runtime. It is for example difficult to expose the entirety of Arcon's framework in Python in a natural way since many of its components (e.g., operator implementations) cannot simply be passed around as values in Rust. Arcon can however be exposed as a library in Python at the cost of flexibility by exposing hard-coded versions of Arcon operators (e.g., map, filter). Moreover, exposing Rust in languages with virtual-machines such as Scala would not work since Scala UDFs cannot execute in Rust. Python UDFs are more feasible as they do not need to execute in a new process.

Arc-Script's goal is to be high-level, but not too high-level. Very high-level languages such as SQL come at a loss of generality by heavily constraining what problems can be solved. Instead, Arc-Script aims to be relatively high-level and offer support for abstraction. Unlike Rust, these abstractions should not leak complexity, but may come at the cost of performance. Performance is an issue addressed by Arc-Script's MLIR middle-end optimiser and not the developer.

## Problems

This section pinpoints additional problems and their consequences.

* **P1**:
  * **Problem**: Arcon does not have data analytics-specific *formalism*.
  * **Consequence**: Hard to define what an Arcon program exactly is.
* **P2**:
  * **Problem**: Arcon does not have data analytics-specific *syntax*.
  * **Consequence**: Programs are more difficult to express than they should be (boilerplate code, unfamiliarity).
* **P3**:
  * **Problem**: Arcon executes programs exactly as they are written without high-level optimisations.
  * **Consequence**: Programmers need to think about how they write their programs to achieve good performance.
* **P4**:
  * **Problem**: Arcon has no information about the code which it is executing
  * **Consequence**: Cannot support runtime optimisation like operator fusion
* **P5**:
  * **Problem**: Arcon does not support a type system at the data analytics level
  * **Consequence**: Hard to statically verify certain properties while providing helpful error messages
* **P6**:
  * **Problem**: Rust's type system is restrictive.
  * **Consequence**: Arcon has no option but to not support generalisations that would be helpful at the data analytics-level (e.g., operator arity).
* **P7**:
  * **Problem**: Arcon combines driver- and worker-code, and compiles code eagerly and only once, for all machines at startup.
  * **Consequences**: Driver and workers must have homogeneous hardware 
* **P7**:
  * **Problem**: Arcon's Rust-based framework is tightly coupled with its runtime
  * **Consequence**: Programs written in Arcon cannot run on other runtimes (e.g., Timely, Flink) 

## Outline

In response to the previously mentioned problems, the question is, can we get *"the best of both worlds?"* of Rust's and Arc-Script's benefits. In other words, can we achieve both efficiency and simplicity? The coming sections will go through base language concepts of Arc-Script. Then, we will delve deeper into dataflows and collections.

[^1] In contrast to libraries, frameworks rely on the concept of [*inversion of control*](https://en.wikipedia.org/wiki/Inversion_of_control) which in basic terms mean that users can extend the framework with custom code (e.g., UDFs and operators), but is not in charge of how that code will be executed.
