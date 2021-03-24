#![allow(unused)]
#![allow(clippy::all)]
// Copyright (c) 2020, KTH Royal Institute of Technology.
// SPDX-License-Identifier: AGPL-3.0-only

mod data;

use data::*;

use arcon::prelude::*;

use arcorn::state::ArcMap;
use arcorn::state::ArcMapOps;
use arcorn::state::ArcRef;
use arcorn::state::ArcRefOps;
use arcorn::state::ArcSet;
use arcorn::state::ArcSetOps;
use arcorn::state::ArcVec;
use arcorn::state::ArcVecOps;

use arcorn::state::ArcKey;
use arcorn::state::ArcValue;

use std::collections::HashMap as Map;
use std::collections::HashSet as Set;

pub struct Task<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition> {
    data: &'i mut TaskData,
    ctx: &'i mut OperatorContext<'source, 'timer, 'channel, TaskData, B, C>,
    timestamp: Option<u64>,
}

pub(crate) struct TaskData {
    state: TaskState,
    param1: i32,
    param2: u32,
}

#[derive(ArconState)]
pub(crate) struct TaskState {
    state0: ArcRef<InputValue1, Sled>,
    state1: ArcVec<InputValue1, Sled>,
    state2: ArcMap<u64, InputValue1, Sled>,
    state3: ArcSet<u64, Sled>,
}

impl StateConstructor for TaskState {
    type BackendType = Sled;

    #[rustfmt::skip]
    fn new(backend: Arc<Self::BackendType>) -> Self {
        Self {
            state0: ArcRefOps::arc_ref_new("state0", backend.clone(), InputValue1 { val: 0 }).unwrap(),
            state1: ArcVecOps::arc_vec_new("state1", backend.clone(), Vec::new()).unwrap(),
            state2: ArcMapOps::arc_map_new("state2", backend.clone(), Map::new()).unwrap(),
            state3: ArcSetOps::arc_set_new("state3", backend.clone(), Set::new()).unwrap(),
        }
    }
}

pub(crate) enum Handle {
    Event(Input),
    Timer(Timer),
}

pub(crate) type Duration = u64;

pub(crate) enum Emit {
    Event(Output),
    Timer(Timer, Duration),
}

pub(crate) enum Input {
    Input1(InputData1),
    Input2(InputData2),
}

pub(crate) enum Output {
    Output1(OutputData1),
    Output2(OutputData2),
}

pub(crate) enum Timer {
    Timer1(TimerData1),
    Timer2(TimerData2),
}

impl Operator for TaskData {
    type IN = InputData1;
    type OUT = OutputData1;
    type TimerState = TimerData1;
    type OperatorState = TaskState;

    fn handle_element(
        &mut self,
        element: ArconElement<Self::IN>,
        ref mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
    ) -> OperatorResult<()> {
        let ArconElement { timestamp, data } = element;

        Task {
            data: self,
            ctx,
            timestamp,
        }
        .handle(Handle::Event(Input::Input1(data)));

        Ok(())
    }

    fn handle_timeout(
        &mut self,
        data: Self::TimerState,
        ref mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
    ) -> OperatorResult<()> {
        Task {
            data: self,
            ctx,
            timestamp: None,
        }
        .handle(Handle::Timer(Timer::Timer1(data)));

        Ok(())
    }

    fn persist(&mut self) -> OperatorResult<()> {
        self.state.persist()
    }

    fn state(&mut self) -> &mut Self::OperatorState {
        &mut self.state
    }
}

impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
    Task<'i, 'source, 'timer, 'channel, B, C>
{
    fn handle(&mut self, event: Handle) {
        match event {
            Handle::Event(Input::Input1(event)) => todo!(),
            Handle::Event(Input::Input2(event)) => {
                // User-Logic
                let x = TimerData1 {
                    val: TimerValue1 {
                        val: event.val.val - 1,
                    },
                    key: event.key,
                };
                self.emit(Emit::Timer(Timer::Timer1(x), 60))
            }
            Handle::Timer(Timer::Timer1(event)) => todo!(),
            Handle::Timer(Timer::Timer2(event)) => {
                // User-Logic
                let x = OutputData1 {
                    val: OutputValue1 {
                        val: event.val.val + 1,
                    },
                    key: event.key,
                };
                self.emit(Emit::Event(Output::Output1(x)));
            }
        }
    }

    fn emit(&mut self, event: Emit) {
        match event {
            Emit::Event(Output::Output2(data)) => todo!(),
            Emit::Event(Output::Output1(data)) => self.ctx.output(ArconElement {
                data,
                timestamp: self.timestamp,
            }),
            Emit::Timer(Timer::Timer2(data), after) => todo!(),
            Emit::Timer(Timer::Timer1(data), after) => {
                let key: u64 = data.get_key();
                let time = self.ctx.current_time().unwrap() + after;
                self.ctx.schedule_at(key, time, data);
            }
        }
    }
}

// Events constructed in body are always output-events
// Timer events are treated specially
// Is subtyping needed?
// Handle = Input1 \/ Input2 \/ .. \/ InputN \/ Timer1 \/ Timer2 \/ .. \/ TimerN
// Emit = Output1 \/ Output2 \/ .. \/ OutputN \/ Timer1 \/ Timer2 \/ .. \/ TimerN
//
// if x:T and T = X0 \/ X1 \/ .. \/ XN where forall X,
// * X ∈ Handle
// * X ∈ Emit
