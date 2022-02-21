# Tasks

Arc-Lang introduces the concept of tasks in order to support complex event processing. A task is an asynchronous function over data streams. Tasks can `receive` and `emit` streaming events on incoming and outgoing streams respectively.

```
task merge(s0, s1, f): (s2) {
    loop {
        val x = receive s0;
        val y = receive s1;
        emit f(x, y) in s2;
    }
}

val stream0 = read_numbers_stream();
val stream1 = read_numbers_stream();
val stream2 = merge(stream0, stream1, (+));
```

While `receive` pulls an event selectively from a specific stream, it is also possible to await events non-selectively, from a set of streams, using the `on` syntax:

```
```

The current limitations of tasks are that:
* Tasks compile into streaming operators in a static dataflow graph. Therefore, the set of input and output streams going in and out of the task must be known statically at compile-time. It is not possible to send a vector of streams into the task. Streams may only be passed in as bare parameters.
* Since the graph is static, a task cannot create another task.
