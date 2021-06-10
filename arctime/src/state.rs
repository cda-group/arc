pub enum State<T: Clone> {
    Initialised(T),
    Uninitialised,
}

impl<T: Clone> State<T> {
    pub fn set(&mut self, data: T) {
        *self = Self::Initialised(data);
    }

    pub fn get(&self) -> T {
        match self {
            State::Initialised(data) => data.clone(),
            State::Uninitialised => panic!("Attempted to access uninitialised cell"),
        }
    }
}
