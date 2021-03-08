// // compile-flags: --error-format=human
// #[arc_script::compile("stateful.arc")]
// mod script {}
//
// fn main() {}
//
// use arcon::prelude::Backend;
// use arcon::prelude::ComponentDefinition;
//
// impl<'i, 'source, 'timer, 'channel, B: Backend, C: ComponentDefinition>
//     script::Stateful<'i, 'source, 'timer, 'channel, B, C>
// {
//     fn update(&mut self) -> i32 {
//         1
// //         self.data.state.value += 1;
// //         self.data.state.value
//     }
// }
