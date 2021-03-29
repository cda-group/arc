#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

macro_rules! test {
    { $name:ident , $file:literal } => { mod $name { arc_script::include!($file); }}
}

mod expect_pass {
    test!(t00, "src/tests/expect_pass/basic_pipe.rs");
    test!(t01, "src/tests/expect_pass/binops.rs");
    test!(t02, "src/tests/expect_pass/enum_pattern.rs");
    test!(t03, "src/tests/expect_pass/enum_pattern_nested.rs");
    test!(t04, "src/tests/expect_pass/enums.rs");
    test!(t05, "src/tests/expect_pass/fib.rs");
    test!(t06, "src/tests/expect_pass/fun.rs");
    test!(t07, "src/tests/expect_pass/if_let.rs");
    test!(t08, "src/tests/expect_pass/ifs.rs");
    test!(t09, "src/tests/expect_pass/lambda.rs");
    test!(t10, "src/tests/expect_pass/literals.rs");
    test!(t11, "src/tests/expect_pass/nested_if.rs");
    test!(t12, "src/tests/expect_pass/option.rs");
    test!(t13, "src/tests/expect_pass/path.rs");
    test!(t14, "src/tests/expect_pass/pattern.rs");
    test!(t15, "src/tests/expect_pass/if.rs");
    test!(t16, "src/tests/expect_pass/structs.rs");
    test!(t17, "src/tests/expect_pass/basic_by.rs");
    test!(t18, "src/tests/expect_pass/sort_fields.rs");
}

mod expect_mlir_fail_todo {
    test!(t00, "src/tests/expect_mlir_fail_todo/pipe.rs");
    test!(t01, "src/tests/expect_mlir_fail_todo/task_filter.rs");
    test!(t02, "src/tests/expect_mlir_fail_todo/task_id_untagged.rs");
    test!(t03, "src/tests/expect_mlir_fail_todo/task_map.rs");
    test!(t04, "src/tests/expect_mlir_fail_todo/task_with_funs.rs");
    test!(t05, "src/tests/expect_mlir_fail_todo/map_state.rs");
    test!(t06, "src/tests/expect_mlir_fail_todo/task_unique.rs");
    test!(t07, "src/tests/expect_mlir_fail_todo/task_window.rs");

    // Integration tests
    mod extern_fun;
    mod extern_state_update;
}

// mod x {
//
//     use arc_script::arcorn;
//     use arc_script::arcorn::state::{ArcMapOps, ArcRefOps, ArcSetOps, ArcVecOps};
//     use arcon::prelude::*;
//     pub struct TumblingWindowSum<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition> {
//         pub data: &'i mut TumblingWindowSumData,
//         pub ctx: &'i mut OperatorContext<'source, 'timer, 'channel, TumblingWindowSumData, B, C>,
//         pub timestamp: Option<u64>,
//         pub key: u64,
//     }
//     pub struct TumblingWindowSumData {
//         pub state: TumblingWindowSumState,
//     }
//     #[derive(ArconState)]
//     pub struct TumblingWindowSumState {
//         pub agg_0: arc_script::arcorn::state::ArcRef<i32, arcon::prelude::Sled>,
//     }
//     impl StateConstructor for TumblingWindowSumState {
//         type BackendType = arcon::prelude::Sled;
//         fn new(backend: Arc<Self::BackendType>) -> Self {
//             Self {
//                 agg_0:
//                     <arc_script::arcorn::state::ArcRef<i32, arcon::prelude::Sled>>::arc_ref_default(
//                         "agg_0",
//                         backend.clone(),
//                     )
//                     .unwrap(),
//             }
//         }
//     }
//     impl TumblingWindowSumData {
//         fn new() -> OperatorBuilder<TumblingWindowSumData> {
//             OperatorBuilder {
//                 constructor: Arc::new(move |backend| {
//                     TumblingWindowSumData {
//                 state: TumblingWindowSumState {
//                     agg_0:
//                         <arc_script::arcorn::state::ArcRef<i32, arcon::prelude::Sled>>::arc_ref_new(
//                             "agg_0",
//                             backend.clone(),
//                             {
//                                 let y_1_0 = 0i32;
//                                 y_1_0
//                             },
//                         )
//                         .unwrap(),
//                 },
//             }
//                 }),
//                 conf: Default::default(),
//             }
//         }
//     }
//     impl Operator for TumblingWindowSumData {
//         type IN = Struct3vali323keyi32End;
//         type OUT = Struct3vali323keyi32End;
//         type TimerState = Struct3valUnit3keyu64End;
//         type OperatorState = TumblingWindowSumState;
//         fn handle_element(
//             &mut self,
//             element: ArconElement<Self::IN>,
//             ref mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
//         ) -> OperatorResult<()> {
//             let ArconElement { timestamp, data } = element;
//             let mut task = TumblingWindowSum {
//                 data: self,
//                 ctx,
//                 timestamp,
//                 key: data.get_key(),
//             };
//             task.handle_element(data);
//             Ok(())
//         }
//         fn state(&mut self) -> &mut Self::OperatorState {
//             &mut self.state
//         }
//         fn handle_timeout(
//             &mut self,
//             timeout: Self::TimerState,
//             ref mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
//         ) -> OperatorResult<()> {
//             let mut task = TumblingWindowSum {
//                 data: self,
//                 key: timeout.key,
//                 ctx,
//                 timestamp: None,
//             };
//             task.handle_timeout(timeout.val);
//             Ok(())
//         }
//         fn persist(&mut self) -> OperatorResult<()> {
//             self.state.persist();
//             Ok(())
//         }
//     }
//     impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
//         TumblingWindowSum<'i, 'source, 'timer, 'channel, B, C>
//     {
//         fn handle_element(&mut self, x_0: Struct3vali323keyi32End) -> OperatorResult<()> {
//             {
//                 let y_1_0 = x_0.val;
//                 let y_1_1 = x_0.key;
//                 let y_1_2 = self.data.state.agg_0.arc_ref_read().unwrap();
//                 let y_1_3 = y_1_2 + y_1_0;
//                 self.data.state.agg_0.arc_ref_write(y_1_3).unwrap()
//             };
//             Ok(())
//         }
//         fn emit(&mut self, data: Struct3vali323keyi32End) {
//             let element = ArconElement {
//                 data,
//                 timestamp: self.timestamp,
//             };
//             self.ctx.output(element);
//         }
//         fn handle_timeout(&mut self, x_1: Struct3valUnit3keyu64End) {
//             {
//                 let y_1_0 = x_1.val;
//                 let y_1_1 = ();
//                 let y_1_2 = y_1_0 == y_1_1;
//                 let y_1_3 = if y_1_2 {
//                     let y_2_0 = x_1.key;
//                     let y_2_1 = self.data.state.agg_0.arc_ref_read().unwrap();
//                     let y_2_2 = Struct3vali323keyi32End {
//                         val: y_2_1,
//                         key: y_2_0,
//                     };
//                     let y_2_3 = 0i32;
//                     let y_2_4 = ();
//                     let y_2_5 = Struct3valUnit3keyi32End {
//                         val: y_2_4,
//                         key: y_2_0,
//                     };
//                     let y_2_6 = 60u64;
//                     let y_2_7 = Struct3valStruct3valUnit3keyi32End3durdurationEnd {
//                         val: y_2_5,
//                         dur: y_2_6,
//                     };
//                     let y_2_8 = self.data.state.agg_0.arc_ref_write(y_2_3).unwrap();
//                     self.trigger(y_2_7);
//                     let y_2_9 = self.emit(y_2_2);
//                     y_2_8;
//                     y_2_9
//                 } else {
//                     let y_2_0 = todo!();
//                     y_2_0
//                 };
//                 y_1_3
//             }
//         }
//         fn trigger(&mut self, timer: Struct3valUnit3keyu64End) {
//             let time = self.ctx.current_time().unwrap() + timer.dur;
//             self.ctx.schedule_at(self.key, time, timer.val);
//         }
//     }
//     #[arcorn::rewrite]
//     #[derive(Copy)]
//     pub struct Struct3valUnit3keyu64End {
//         val: (),
//         key: u64,
//     }
//     #[arcorn::rewrite]
//     #[derive(Copy)]
//     pub struct Struct3valStruct3valUnit3keyi32End3durdurationEnd {
//         val: Struct3valUnit3keyi32End,
//         dur: u64,
//     }
//     #[arcorn::rewrite]
//     #[derive(Copy)]
//     pub struct Struct3vali323keyi32End {
//         val: i32,
//         key: i32,
//     }
//     #[arcorn::rewrite]
//     #[derive(Copy)]
//     pub struct Struct3valUnit3keyi32End {
//         val: (),
//         key: i32,
//     }
// }
