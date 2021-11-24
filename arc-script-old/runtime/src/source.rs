// Window operator can forward
// Watermark emitted after trigger
use kompact::prelude::*;
use std::time::Duration;
use time::PrimitiveDateTime as DateTime;

use crate::pipeline::Pipeline;
use crate::port::DataReqs;
use crate::port::StreamEvent;
use crate::stream::Stream;
use crate::task::create_connector;
use crate::task::Task;

impl<S: SystemHandle> Pipeline<S> {
    /// Constructs a source `Task` which continuously produces data by pulling an iterator.
    pub fn source<T, O: DataReqs>(&self, iter: T, duration: Duration) -> Stream<O>
    where
        T: IntoIterator<Item = (DateTime, O)>,
        <T as IntoIterator>::IntoIter: DataReqs,
    {
        let iter = iter.into_iter();
        let task = self.system.create(move || {
            Task::producer("Source", iter).set_periodic_timer(duration, Task::produce)
        });
        let connect = create_connector(task.clone());
        let client = self.client.clone();
        self.startup
            .borrow_mut()
            .push(Box::new(move || client.system().start(&task)));
        Stream::new(self.client.clone(), connect, self.startup.clone())
    }

    #[allow(deprecated)]
    pub fn serial_source<T, O: DataReqs>(&self, iter: T, duration: Duration) -> Stream<O>
    where
        T: IntoIterator<Item = O>,
        <T as IntoIterator>::IntoIter: DataReqs,
    {
        let iter = iter
            .into_iter()
            .enumerate()
            .map(|(i, x)| (DateTime::from_unix_timestamp(i as i64), x));
        self.source(iter, duration)
    }
}

impl<T: Iterator<Item = (DateTime, O)> + DataReqs, O: DataReqs> Task<T, Never, O, ()> {
    /// Function for producing data inside a task.
    fn produce(&mut self) {
        if let Some((time, data)) = self.state.next() {
            if time > self.lowest_observed_watermarks[0] {
                self.lowest_observed_watermarks[0] = time;
                self.data_oport.trigger(StreamEvent::Data(time, data));
            } else {
                info!(self.ctx.log(), "Discarded late event with time {}", time);
            }
        } else {
            self.exit(());
        }
    }
}
