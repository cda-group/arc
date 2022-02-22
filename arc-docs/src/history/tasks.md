# Evolution of Tasks

In the original design, tasks were able to accept a single input and output stream.

```
task Identity(): ~i32 -> ~i32 {
    var x = 0;
    on event => {
        x += 1;
        emit x;
    }
}
```

This was then extended so that tasks could accept multiple input and output streams.


