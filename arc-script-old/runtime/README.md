# The Arctime

The Arctime is a non-distributed runtime for the [Arc](https://github.com/cda-group/arc) language, built in [kompact](https://github.com/cda-group/kompact). The purpose of the Arctime is to test out new experimental features which could potentially be added to the [Arcon](https://github.com/cda-group/arcon/) distributed runtime.

## Example

Following is a basic example

```rust
fn main() {
    let pipeline = Pipeline::new();
    
    pipeline
        .source(0..200, Duration::new(0, 5_000_000))
        .apply(Task::new("Map", (), |task, event| task.emit(event + 1)))
        .apply(Task::new("Filter", (), |task, event| {
            if event % 2 == 0 {
                task.emit(event);
            }
        }))
        .apply(Task::new("Reduce", 0, |task, event| {
            task.state += event;
            task.emit(event);
        }))
        .sink(Task::new("Print", (), |task, event| {
            info!(task.ctx.log(), "{}", event);
        }));


    pipeline.execute();
}
```

## Supported functionality so far

- [x] **Tasks** (functions over streams)
- [x] **Ephemeral state**
- [x] **Timers**
- [x] **Sources** (Tasks which only produce data)
- [x] **Sinks** (Tasks which only consume data)
- [x] **Transformations** (Tasks which consume and produce data)
- [x] **Multiplexing** (Connecting the same stream to multiple different tasks)
- [x] **Multiporting** (Tasks can have multiple different input and output streams)
- [x] **Duplexing** (Consumers can send control-events to their producers)
- [x] **Bounded buffers** (Producers buffer their output data)
- [x] **Finite streams** (Tasks terminate when their inputs are depleted)
- [x] **Nested pipelines** (It is possible to spawn a pipeline inside another)
- [ ] **Short running tasks** (Tasks can return values when they terminate)
- [ ] **Flow control** (Consumers pull data from their producers)
- [ ] **Event time**
- [ ] **Data parallelism**
