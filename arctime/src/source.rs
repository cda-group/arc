// Window operator can forward
// Watermark emitted after trigger
use crate::control::*;
use kompact::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use time::PrimitiveDateTime as DateTime;

use crate::client::*;
use crate::pipeline::*;
use crate::port::*;
use crate::stream::*;
use crate::task::*;

impl<S: SystemHandle> Pipeline<S> {
    pub fn source<T, O: DataReqs>(&self, iter: T, duration: Duration) -> Stream<O>
    where
        T: IntoIterator<Item = (DateTime, O)>,
        <T as IntoIterator>::IntoIter: DataReqs,
    {
        let iter = iter.into_iter();
        let task = Task::new_periodic(
            "Source",
            iter,
            duration,
            |task: &mut Task<<T as IntoIterator>::IntoIter, Never, O, ()>| {
                if let Some((time, data)) = task.state.next() {
                    if time > task.lowest_observed_watermarks[0] {
                        task.lowest_observed_watermarks[0] = time;
                        task.data_oport.trigger(StreamEvent::Data(time, data));
                    } else {
                        info!(task.ctx.log(), "Discarded late event with time {}", time);
                    }
                } else {
                    task.exit(());
                }
            },
        )
        .set_role(Role::Producer);
        let task = self.system.create(move || task);
        let connect = create_connector(task.clone());
        let client = self.client.clone();
        self.startup
            .borrow_mut()
            .push(Box::new(move || client.system().start(&task)));
        Stream::new(self.client.clone(), connect, self.startup.clone())
    }
}
