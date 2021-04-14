# Pipelines

If a **task** is a *first-order* function over streams, then a **pipeline** is a *higher-order* function over streams. Pipelines are in reality just ordinary functions that take streams or pipelines as input and return streams or pipelines as output.

## Example

The following code represents a pipeline of three map tasks:

```text
fun my_pipeline(stream0: ~i32): ~i32 {
    val stream1 = Map(fun(x): x-1) (stream0);
    val stream2 = Map(fun(x): x+2) (stream1);
    val stream3 = Map(fun(x): x*2) (stream2);
    stream3
}
```

Visually the pipeline looks like this:

```text
    -> Map(fun(x): x-1) -> Map(fun(x): x+2) -> Map(fun(x): x*2) ->
```
