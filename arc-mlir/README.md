# MLIR Use in Arc-lang

MLIR is used in the Arc-lang compiler back-end. It is used for both
standard compiler optimizations and transforms but also as convenient
framework in which to express Arc-lang-specific transforms. The
MLIR/LLVM infrastructure for testing is also used to run Arc-lang
regression and unit tests.

The purpose of the Arc-lang compiler back-end is to do domain
specific transforms and optimizations which improves the efficiency of
the program when running on Arcon. Doing transforms such as operator
fusion and operator reordering requires a number of standard compiler
techniques such as liveness analysis, constant propagation and common
sub-expression elimination. By using the MLIR infrastructure, which
implements many of these standard algorithms, we can concentrate on
what is specific to Arc-lang. By using MLIR we also have a robust
and extensible intermediary representation with tools for parsing,
printing and verifying structure invariants of the processed program
representation.

## MLIR

TODO: MLIR blurb and flesh out the bullets.

* [MLIR](https://mlir.llvm.org/) is a Multi-Level Intermediate
  Representation

* Extensibility

* Dialects

* Types

* Standard transforms and optimizations on custom dialects

* Tooling infrastructure: command line parsing, debug flags, pass
  ordering, error reporting.

* Testing support: Powerful DAG-matching tool to verify structure and
  syntax of output; Error report verification integration with the
  error reporting in the tooling infrastructure

## Structure

The Arc-lang front-end processes the Arc-lang source code and
produces a representation of the program in the arc-lang MLIR
dialect for further processing. The parts of the Arc-lang compiler
pipeline which uses MLIR is implemented in a tool called
`arc-mlir`. The tool is implemented using the MLIR tooling framework
and allows the user to, on the command line, select which
optimizations and transforms to run. Input to the `arc-mlir` tool is
MLIR-IR in the arc-lang dialect and output is in either: the
arc-lang IR dialect, the Rust dialect or textual Rust source code.

## The Arc MLIR Dialect

The Arc MLIR dialect is an MLIR dialect in which it is possible to
represent all arc-lang language constructs in a way that allows the
generation of a syntactically and semantically valid Rust program. The
dialect consists of operations from the `standard`, `scf`, and `arith`
dialects provided by upstream MLIR, but also a number of custom
operations and types specific to arc-lang.

### Arc Dialect Types

The `arc` dialect includes a number of types which are not provided by
one of the upstream dialects, these include:

- `arc.adt<`*string*`>` An opaque type which wraps a Rust type. It is
  preserved by all IR transformations. When Rust source code is
  output, values of this type will be declared as type *string*.

- `arc.enum` A Rust-style enum. A discriminated union where each named
  variant maps to a type. Structural equality applies to enum types.

- `arc.struct` An aggregate type which aggregates a set of named and
  typed fields. Structural equality applies to enum types.

- `arc.stream` A type which corresponds to event streams in
  arc-lang. The stream is instantiated with the type of the event it
  carries.

### Arcon Dialect Types

The `arc` dialect includes a number of types which are direct mappings
onto types in the Arcon runtime, these include:

- `arcon.appender` A type for an entity which aggregates multiple
  values of a single type and produces a single value.

- `arcon.map` A type for an entity which maps values of a singly type
  to other typed values.

- `arcon.value` Instantiated with an arc dialect type, this type
  indicates that values of this type are to be included in the arcon
  operator state.

### Custom Arc Dialect Operations

In the `arith` dialect, MLIR provides arithmetic operations on integer
and floating point values. MLIR provides three integer types: one type
which only specifies the number of bits `i<n>`, an explicitly signed
integer type `si<n>`, and an explicitly unsigned integer type. The
arithmetic operations on integers in `arith` are only specified for
the `i<n>` integer type. In that, `arith` follows the model chosen by
LLVM in that the signed/unsigned semantics for an operation is
selected by the operation, for example `divi`/`divui` for
signed/unsigned integer division. As both our input and output
languages (Arc-lang and Rust respectively) derive the signed/unsigned
semantics from the type, we have chosen to use the explicitly
signed/unsigned integer types. The alternative would require the
component responsible for Rust output to derive the type of integer
variables from the operations applied to them, something that is not
always possible if no operations with different semantics are applied
to them. Therefore the `arc` dialect defines its own polymorphic
arithmetic operations operating on signed/unsigned integers.

TODO: operations

TODO: event handler

TODO: Structure of the arc-lang program: Each block produces a
result, SSA-ish. No branches between blocks.

TODO: structured control flow

### Rust MLIR Dialect

TODO: Operations which capture the structure of Rust.

TODO: Types are the rust type as a string.

TODO: Name mangling to produce Rust type names for the aggregate
types.

TODO: Not intended to be the subject of any transforms or
optimizations, that is done by rustc.

## Standard Transforms and Optimizations

TODO: canonicalization; CSE; constant propagation and folding;
constant lifting.

## Custom Transforms

TODO: From SCF to BBs: In order to use additional optimizations and
transforms.

TODO: From BBs to SCF, needed for Rust (no goto).

TODO: FSM-transform, for selective and nested receive in event
handlers.

## Rust Output

TODO: Abstracting away reference counting, borrows etc. Handled by
macros in the runtime system libraries.

TODO: No formatting, rustfmt handles that.

## Testing

TODO: Use Lit for unit and regression tests

TODO: Use built-in support in tooling to check that errors occur where
we expect them.
