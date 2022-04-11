# Tasks

A **task** is an asynchronous function which can suspend its execution to wait for events. The `receive`, `on`, and `!` expressions can (for now) only be used inside tasks.

```grammar
Task ::= "task" [Name] [Generics]? [Params] ":" [Params] [Block]
```

## Examples

### Basic task

```arc-lang
{{#include ../../../arc-lang/examples/task-identity.arc:example}}
```

### Lambda tasks

```arc-lang
{{#include ../../../arc-lang/examples/task-lambda.arc:example}}
```

### Multi-Input Tasks

```arc-lang
{{#include ../../../arc-lang/examples/task-merge.arc:example}}
```

### Multi-Output Tasks
