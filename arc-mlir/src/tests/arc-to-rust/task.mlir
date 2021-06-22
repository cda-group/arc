// RUN: arc-mlir-rust-test %t %s
// RUN: arc-mlir-rust-test %t-canon %s -canonicalize

module @toplevel {
  func @my_handler(%this : !arc.struct<x : si32>,
                   %in   : !arc.enum<A : si32, B : si32>,
                   %out  : !arc.stream<!arc.enum<C : si32, D : si32>>)
                -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_event_handler"}
  {
    %isA = arc.enum_check (%in : !arc.enum<A : si32, B : si32>) is "A" : i1
    "arc.if"(%isA) ( {
      %a = arc.enum_access "A" in (%in : !arc.enum<A : si32, B : si32>) : si32
      %e = arc.make_enum (%a : si32) as "C" : !arc.enum<C : si32, D : si32>
      "arc.emit" (%e, %out) : (!arc.enum<C : si32, D : si32>, !arc.stream<!arc.enum<C : si32, D : si32>>) -> ()
      "arc.block.result"(%isA) : (i1) -> ()
    },  {
      %b = arc.enum_access "B" in (%in : !arc.enum<A : si32, B : si32>) : si32
      %e = arc.make_enum (%b : si32) as "D" : !arc.enum<C : si32, D : si32>
      "arc.emit" (%e, %out) : (!arc.enum<C : si32, D : si32>, !arc.stream<!arc.enum<C : si32, D : si32>>) -> ()
      "arc.block.result"(%isA) : (i1) -> ()
    }) : (i1) -> (i1)
    return
  }

  func @init(%this : !arc.struct<x : si32>) -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_init"}
  {
    return
  }
}
