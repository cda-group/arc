use crate::prelude::*;
use crate::task::*;

impl<I: DataReqs> Stream<I> {
    /// Transform a stream into a new stream.
    pub fn apply_terminating<S: DataReqs, O: DataReqs, R: DataReqs>(
        self,
        task: Task<S, I, O, R>,
    ) -> Stream<O> {
        task(self)
    }
    /// Transform a stream into a new stream.
    pub fn apply<S: DataReqs, O: DataReqs>(self, task: Task<S, I, O, Never>) -> Stream<O> {
        task(self)
    }

    /// Transform two streams into one.
    pub fn merge<S: DataReqs, X: DataReqs, O: DataReqs>(
        self,
        other: Stream<X>,
        task: Task<S, Either<I, X>, O, Never>,
    ) -> Stream<O> {
        let mergel = self.client.system().create(|| {
            Task::new(
                "Merge Left",
                (),
                |task: &mut Task<(), I, Either<I, X>, Never>, event| task.emit(Either::L(event)),
            )
        });
        let merger = self.client.system().create(|| {
            Task::new(
                "Merge Right",
                (),
                |task: &mut Task<(), X, Either<I, X>, Never>, event| task.emit(Either::R(event)),
            )
        });
        let task = self.client.system().create(|| task);
        biconnect_components::<StreamPort<_>, _, _>(&task, &mergel).expect("biconnect");
        biconnect_components::<StreamPort<_>, _, _>(&task, &merger).expect("biconnect");
        mergel.on_definition(|c| (self.connector)(&mut c.data_iport));
        merger.on_definition(|c| (other.connector)(&mut c.data_iport));
        let connect = create_connector(task.clone());
        let client = self.client.clone();
        self.start_fns
            .borrow_mut()
            .push(Box::new(move || client.system().start(&mergel)));
        let client = self.client.clone();
        self.start_fns
            .borrow_mut()
            .push(Box::new(move || client.system().start(&merger)));
        let client = self.client.clone();
        self.start_fns
            .borrow_mut()
            .push(Box::new(move || client.system().start(&task)));
        Stream::new(self.client, connect, self.start_fns)
    }

    /// Transform one stream into two.
    pub fn split<S: DataReqs, X: DataReqs, O: DataReqs>(
        self,
        task: Task<S, I, Either<O, X>, Never>,
    ) -> (Stream<O>, Stream<X>) {
        let taskl = Task::new(
            "Split Left",
            (),
            |task: &mut Task<(), Either<O, X>, O, Never>, event| {
                if let Either::L(event) = event {
                    task.emit(event)
                }
            },
        );
        let taskr = Task::new(
            "Split Right",
            (),
            |task: &mut Task<(), Either<O, X>, X, Never>, event| {
                if let Either::R(event) = event {
                    task.emit(event)
                }
            },
        );
        let stream = self.apply(task);
        let streaml = taskl(stream.clone());
        let streamr = taskr(stream);
        (streaml, streamr)
    }
}
