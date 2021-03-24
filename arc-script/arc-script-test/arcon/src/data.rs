#[arcorn::rewrite]
pub(crate) struct InputData1 {
    pub(crate) val: InputValue1,
    pub(crate) key: i32,
}

#[derive(prost::Message)]
pub(crate) struct Data {
    #[prost(message, required)]
    x: (),
}

#[arcorn::rewrite]
pub(crate) struct InputData2 {
    pub(crate) val: InputValue2,
    pub(crate) key: i32,
}

#[arcorn::rewrite]
pub(crate) struct OutputData1 {
    pub(crate) val: OutputValue1,
    pub(crate) key: i32,
}

#[arcorn::rewrite]
pub(crate) struct OutputData2 {
    pub(crate) val: OutputValue2,
    pub(crate) key: i32,
}

#[arcorn::rewrite]
#[derive(PartialEq)]
pub(crate) struct TimerData1 {
    pub(crate) val: TimerValue1,
    pub(crate) key: i32,
}

#[arcorn::rewrite]
#[derive(PartialEq)]
pub(crate) struct TimerData2 {
    pub(crate) val: TimerValue2,
    pub(crate) key: i32,
}

#[arcorn::rewrite]
pub(crate) struct InputValue1 {
    pub(crate) val: i32,
}

#[arcorn::rewrite]
pub(crate) struct InputValue2 {
    pub(crate) val: i32,
}

#[arcorn::rewrite]
pub(crate) struct OutputValue1 {
    pub(crate) val: i32,
}

#[arcorn::rewrite]
pub(crate) struct OutputValue2 {
    pub(crate) val: i32,
}

#[arcorn::rewrite]
#[derive(PartialEq)]
pub(crate) struct TimerValue1 {
    pub(crate) val: i32,
}

#[arcorn::rewrite]
#[derive(PartialEq)]
pub(crate) struct TimerValue2 {
    pub(crate) val: i32,
}
