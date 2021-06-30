#![allow(unused)]

use arctime::prelude::*;
use kompact::prelude::*;
use std::time::Duration;

fn main() {
    let executor = Executor::new();

    executor
        .pipeline()
        .serial_source(0..200, Duration::new(0, 50_000_000))
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
        .apply(Task::new("Nested", (), |task, event: i32| {
            task.pipeline()
                .serial_source(event..100, Duration::new(0, 100_000_000))
                .sink(Task::new("Inner print", (), |task, event| {
                    info!(task.ctx.log(), "Inner: {}", event);
                }))
                .build();
            task.emit(event)
        }))
        .sink(Task::new("Print", (), |task, event| {
            info!(task.ctx.log(), "Outer: {}", event);
        }))
        .build();

    executor.execute();
}
