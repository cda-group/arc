# Futures

A **future** is a single asynchronously computed value. This value is initially unknown and will eventually become known as soon as its computation finishes evaluating. While futures are are created by **tasks** (asynchronous functions), they are indistinguishable from ordinary synchronously-computed values in Arc-Script. The properties of futures are:

Futures are *implicit*:
* *Explicit* futures must be explicitly blocked to get their value, e.g., `future.get_value()`.
* *Implicit* futures are implicitly blocked when their value is needed in an expression.

Futures are *eager*:
* Evaluation of *eager* futures start immediately when they are created.
* Evaluation of *lazy* futures start only when their value is needed in an expression.
