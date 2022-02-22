# Tasks

A **task** is an asynchronous function which can suspend its execution to wait for events. The `receive`, `on`, and `!` expressions can only be used inside tasks.

```grammar
Task ::= "task" [Name] [Generics]? [Params] ":" [Params] [Block]
```

## Examples

### Basic task

```text
{{#include ../../../arc-lang/examples/task-identity.arc:example}}
```

### Lambda tasks

```text
{{#include ../../../arc-lang/examples/task-lambda.arc:example}}
```

### Multi-Input Tasks

```text
{{#include ../../../arc-lang/examples/task-merge.arc:example}}
```

### Multi-Output Tasks
