# Tasks and Channels

Datastreams have become a prominent data model for analysing live data. A *datastream* is conceptually an infinite sequence of events, where an *event* is a data point produced by the stream at a specific moment in time. Arc-Lang provides the concept of tasks and channels to support fine grained processing of datastreams. A *channel* is a buffer of events which supports two basic operations. Data can be *pushed* and *pulled* into and out of the channel's buffer by tasks. In this context, a *task* is a stackless coroutine. A coroutine is a function that can suspend its execution to be resumed at a later time. Tasks can thus await events to arrive on ingoing channels, do some processing, and emit new events onto outgoing channels. To suspend, the coroutine must save its execution state at the point of suspension. While a stackful coroutine (i.e., lightweight thread) saves its call-stack when suspending, a stackless coroutine does not. Stackful coroutines are therefore able to suspend inside deeply nested stack-frames whereas stackless coroutines may only suspend in their outermost stack-frame. The motivation of having stackless coroutines however is their predictability, since the size of their state is constant and known at compile-time.

```arc-lang
{{#include ../../../arc-lang/examples/task-map.arc:example}}
```

In the above code, `map` is a task which takes a stream on an input channel `i`, a function `f`, and produces a stream on an output channel `o`. The execution of the task enters an infinite loop which receives events from `i` using the `receive` operator, applies function `f`, and emits the result to `o` using the `!` operator. The `main` function instantiates the task using function application syntax.

## Directions

All channels have a *direction*. Data can either be pushed into a channel or pulled out of it, but not both. Tasks which push data are referred to as "*producers*" and tasks which pull data are referred to as "*consumers*". It is possible for a task to simultaneously be both a producer and a consumer, but not for the same channel. We refer to pull-only channels as streams (`Stream[T]`) and push-only channels as drains (`Drain[T]`). To make the idea more clear, consider this fully type annotated example:

```arc-lang
{{#include ../../../arc-lang/examples/task-map.arc:annotated}}
```

## Deadlock Avoidance

The purpose of introducing tasks and directions is to avoid deadlocks that may occur from cyclic channel-dependencies. To give an example, channels in the Go language have directions like Arc-Lang but no task-abstraction. Go instead exposes the more general concept of goroutines which are lightweight threads that can suspend execution anywhere. For example, it is possible in Go to write the following:

```go
package main

func map(i <-chan int, f func(int) int, o chan<- int) {
    for x := range i {
        o <- f(x);
    }
}

func main() {
    c := make(chan int);
    go map(c, c);
}
```

The above code has a problem. The `map` function is both a consumer and producer of the same channel. This cyclic dependency causes a deadlock. In contrast, Arc-Lang restricts programs such that channels can only be created by instantiating tasks. Once a task is created its inputs cannot change. The output channel of a task cannot end up as an input channel to the same task or any upstream task. This prevents arbitrary cycles from being created in the dataflow graph, and thus prevents deadlocks. Code such as the following will result in a syntax error:

```arc-lang
{{#include ../../../arc-lang/examples/fail-task-map.arc:example}}
```

## Exceptions

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

## Flow Control

Channels in Arc-Lang are bounded which means that their buffers have a maximum capacity. It is thus important that buffers do not overflow since this would lead to undefined behavior. For example:

```arc-lang
{{#include ../../../arc-lang/examples/task-flow-control.arc:example}}
```

In a purely push-based system, the producer would produce data at a faster rate than the consumer can keep up. This would would inevitably cause a buffer overflow in the channel. To prevent such circumstances, a channel will block its producer from pushing events if its buffer is full. Similarly, a channel will block its consumers from pulling events if its buffer is empty. This behavior is implemented by storing multiple queues inside of channels. In particular, each channel has:

* A *push-queue* of capacity `P` which stores promises to push data into the channel as soon as its buffer is not full. The respective producer is blocked until the promise is fulfilled.
* A *data-queue* of capacity `N` which stores in-flight events.
* A *pull-queue* of capacity `C` which stores promises to pull data from the channel as soon as its buffer is not empty. The respective consumer is blocked until the promise is fulfilled.

We have that `P = 1` (i.e., a channel can only have one producer), `N = <dynamic>`, and `C = <number of consumers>`. Because channels are multicast, all consumers of a channel will pull the same event sequence. To this end, an offset is stored together with each promise in the pull-queue that indicates which event in the data-queue should be pulled. Events can only be evicted from the data-queue if they have been pulled by all consumers.

## Ordering of Events

Tasks are not limited to just one input and output channel. In fact, a task can have a dynamic number of input channels (by for example passing the channels in a vector) and a fixed number of output channels.

```arc-lang
{{#include ../../../arc-lang/examples/task-merge.arc:example}}
```

In the above code, `merge` is a task which can be used to combine events from two channels `s0` and `s1` into one `s2`. By deterministic, we mean the task pulls events from the input channels in a specific order. It is also possible to receive events non-deterministically from a set of channels, as soon as they arrive, using the `on` syntax:

```arc-lang
{{#include ../../../arc-lang/examples/task-union.arc:example}}
```

In the above code, `union` is a task which can be used to combine two channels `s0` and `s1` non-deterministically into one `s2`. No ordering guarantees are made between the two channels.

## Streaming State

Streaming state is typically exposed explicitly by stream processors. Users need to declare the type of data they want to persistently store and insert read and write operations in their application code to access the state. In Arc-Lang, state is managed implicitly by automatically translating tasks into FSMs (Finite State Machines) that are persisted by the system.

```arc-lang
{{#include ../../../arc-lang/examples/task-state.arc:example}}
```

The code above shows a rolling `reduce` task which reduces a stream using an aggregation function `f`, beginning at an initial state `init`.

Note that state is always encapsulated within a task. It is not possible (as of yet) to share state between tasks. Any value sent over a channel is copied from the producer to the consumer. This allows Arc-Lang programs to be parallelised without having to consider the risk of race conditions. How values are copied is abstracted and can for example be deep (for data with interior mutability), shallow (for immutable data), or lazy (copy-on-write). CRDTs and transactional memory are under consideration for enabling shared state.

## Parallelism

Tasks and channels enable three forms of parallelism:
* **Pipeline parallelism**: Multiple tasks can pipeline their execution through channels.
* **Task parallelism**: Multiple tasks can process data from the same channel in parallel.
* **Data parallelism**: The same task can process data from multiple channels in parallel.

An example of pipeline parallelism:

```arc-lang
{{#include ../../../arc-lang/examples/parallelism.arc:pipeline}}
```

An example of task parallelism:

```arc-lang
{{#include ../../../arc-lang/examples/parallelism.arc:task}}
```

Arc-Lang chooses to make data parallelism implicit since this is the most common form of parallelism. All of the above examples are thus data parallel. The idea of data parallelism is that channels are divided into partitions that can transport events in parallel. Each event has an associated *key* that determines which partition the event is assigned to. A key can for example be the unique identifier of a sensor that generates data, a location tag, or an article number. Events are required to have a key at the point where they enter the system. It is also possible to change the key of an event. This will be covered in the next section. Note that the method for assigning events to partitions is abstract. This allows Arc-Lang to choose the partitioning approach based on the scenario. Though, currently a static partitioning approach is used. 

## Summary

Channels of Arc-Lang are planned to support the following properties:

* **Directions**: Channels can be used to either pull or push data, but not both. Tasks which push data are referred to as "producers" and tasks which pull data are referred to as "consumers". A task can simultaneously be both a producer and a consumer but not for the same channel.
* **Bounded**: Channels have a maximum capacity which, when exceeded, will block the producer from pushing events. Similarly, an empty channel will block the consumer from pulling events.
* **SPMC**: Channels have a single producer and possibly multiple consumers.
* **Multicast**: All consumers of a channel will pull the same event sequence.
* **Exceptions**: A channel can be closed by a producer or consumer. When a channel is closed, all tasks waiting on the channel will be notified.
* **Data parallelism**: A logical channel consists of multiple physical channels which are partitioned using consistent hashing. Data can be transferred over different physical channels in parallel.
* **Deadlock avoidance**: Channels can only be created by instantiating tasks. The output channel of a task cannot be passed as an input channel to the same task or any upstream task. This prevents arbitrary cycles from being created in the dataflow graph, and thus prevents deadlocks.
* **Uni-directional**: Channels can only be used to send data in a single direction, i.e., from producer to consumer. Dataflow graphs created through channels and tasks are always directed and acyclic.
* **Network transparency**: Channels can be used for both local and networked communication. Whether the producer and consumer are on the same machine or on different machines is transparent to the user.
* **Persistence**: Channels should be able to snapshot their state and replay it from an earlier offset.

## Limitations

The following restrictions are for now enforced on channels and tasks but may eventually be lifted:

* Tasks cannot create new tasks. Tasks may only be created by the main-thread of a program. This is because 
* It is only possible to push and pull events from channels inside of tasks.
