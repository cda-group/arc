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

## Task Exceptions

Operations on streams are by default assumed to succeed without failure, but may also throw *exceptions* which can be handled by the user. Exceptions are thrown in two scenarios:

1) When receiving data from a channel which has no producer and whose buffer is empty.
2) When emitting data into a channel which has no consumers.

By catching exceptions, tasks can prolong their execution beyond the lifetime of their channels. This is especially useful when tasks have multiple input and output channels, as uncaught exceptions will terminate the task.

```arc-lang
{{#include ../../../arc-lang/examples/task-exception-at-consumer.arc:example}}
```

In the above code, a `producer` task is sending data to a `consumer` task over a stream. The producer terminates once it is done sending all of its data. Since there are no more tasks which can emit data into the stream (i.e., no more references to the output stream), an exception will be raised when trying to receive data from the input stream.

```arc-lang
{{#include ../../../arc-lang/examples/task-exception-at-producer.arc:example}}
```

The above code shows the opposite scenario, where a consumer can potentially terminate and raise an exception at the producer.
