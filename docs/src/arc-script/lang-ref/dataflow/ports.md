# Ports

To allow the implementation of more advanced operators that for example *merge* and *split* several streams, interfaces of tasks can be provided with additional **ports**. Inside the task, input and output ports are discriminated by-name to know where events come and go. The order in which ports are declared in the interface directly corresponds to the order in which streams connect to the interface.

```text
Item ::=
  | 'task' Name '(' (Name ':' Type ',')* ')' ':' Interface '->' Interface '{' TaskItem* '}'
  | ..

Interface ::=
  | Type                             # An interface with a single port
  | '(' (Name '(' Type ')' ',')+ ')' # An interface with discriminated ports
```

## Examples

`Clone` is a task which clones a stream into two.

```text
task Clone(): ~i32 -> (A(~i32), B(~i32)) {
    on event => {
        emit A(event);
        emit B(event)
    }
}
```

`Merge` is a task which merges two streams into one.

```text
task Merge(): (A(~i32), B(~i32)) -> ~i32 {
    on {
        A(event) => emit event,
        B(event) => emit event,
    }
}
```

Following is an example of how to use `Clone` and `Merge`. The example clones a stream into two streams, maps those streams to get their odd and even numbers, and then merges the resulting streams into one.

```text
fun test(s0: ~i32): ~i32 {
    val (s1, s2) = Clone() (s0);
    val s3 = Map(fun(x): x % 2 == 0) (s1);
    val s4 = Map(fun(x): x % 2 == 1) (s2);
    val s5 = Merge() (s3, s4);
    s5
}
```

Visually the pipeline looks something like this:

```text
         +-> Map(fun(x): x % 2 == 0) -+
         |                            |
         |                            v
-> Clone()                            Merge() ->
         |                            ^
         |                            |
         +-> Map(fun(x): x % 2 == 1) -+
```
