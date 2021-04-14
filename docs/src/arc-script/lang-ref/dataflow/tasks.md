# Tasks

A **task** is an asynchronous first-order function over *asynchronous values* (i.e., streams or futures). Internally, a task is implemented as an *actor* which operates incrementally (per-event) over its streams and futures. Each task has an input and output *interface* of *ports* which streams can connect to. Tasks *receive* and *process* one event from one input port at a time. While processing, the task may *emit* multiple events at multiple output ports. Additional extensions to tasks are described in following sections.

```text
Item ::=
  | 'task' Name '(' (Name ':' Type ',')+ ')' ':' Interface '->' Interface '{' TaskItem* '}'
  | ..

Interface ::=
  | Type  # An interface with a single port
  | ..

TaskItem ::=
  | 'on' '{' (Pattern ('if' Expr)? '=>' Expr ',')* (Control ',')* '}'  # Event handler
  | ..

Control ::=
  | ..  # Handler for special control-events

Expr ::=
  | 'emit' Expr  # Emit event to output port
  | ..
```

## Example 1

The identity task over streams of integers can be represented and used as follows:

```text
task Identity(): ~i32 -> ~i32 {
    on event => emit event
}

fun test(s: ~i32) -> ~i32 {
    Identity() (s)
}
```

## Example 2

The map task over streams of integers can be represented and used as follows (generics will eventually be supported):

```text
task Map(f: fun(i32): i32) ~i32 -> ~i32 {
    on event => emit f(event)
}

fun test(s: ~i32) -> ~i32 {
    Map(fun(x): x + 1) (s)
}
```
