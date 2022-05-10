// RUN: arc-mlir -arc-to-rust -split-input-file -verify-diagnostics %s

module @toplevel {
  // expected-error@+2 {{'rust.func' op : task event handlers are expected to have 3 arguments, found 2}}
  // expected-note@+1 {{see current operation:}}
  func.func @my_handler(%in   : !arc.enum<A : si32, B : si32>,
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

  func.func @init(%this : !arc.struct<x : si32>) -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_init"}
  {
    return
  }
}

// -----

module @toplevel {
  // expected-error@+2 {{'rust.func' op : The first argument to a task event handler is expected to be a struct}}
  // expected-note@+1 {{see current operation:}}
  func.func @my_handler(%in   : !arc.enum<A : si32, B : si32>,
                   %this : !arc.struct<x : si32>,
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

  func.func @init(%this : !arc.struct<x : si32>) -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_init"}
  {
    return
  }
}

// -----

module @toplevel {
  // expected-error@+2 {{'rust.func' op : The second argument to a task event handler is expected to be an enum}}
  // expected-note@+1 {{see current operation:}}
  func.func @my_handler(%this : !arc.struct<x : si32>,
                   %out  : !arc.stream<!arc.enum<C : si32, D : si32>>,
                   %in   : !arc.enum<A : si32, B : si32>)
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

  func.func @init(%this : !arc.struct<x : si32>) -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_init"}
  {
    return
  }
}

// -----

module @toplevel {
  // expected-error@+2 {{The third argument to a task event handler is expected to be a stream}}
  // expected-note@+1 {{see current operation:}}
  func.func @my_handler(%this : !arc.struct<x : si32>,
                   %in   : !arc.enum<A : si32, B : si32>,
                   %out  : !arc.enum<C : si32, D : si32>)
                -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_event_handler"}
  {
    %isA = arc.enum_check (%in : !arc.enum<A : si32, B : si32>) is "A" : i1
    "arc.if"(%isA) ( {
      %a = arc.enum_access "A" in (%in : !arc.enum<A : si32, B : si32>) : si32
      %e = arc.make_enum (%a : si32) as "C" : !arc.enum<C : si32, D : si32>

      "arc.block.result"(%isA) : (i1) -> ()
    },  {
      %b = arc.enum_access "B" in (%in : !arc.enum<A : si32, B : si32>) : si32
      %e = arc.make_enum (%b : si32) as "D" : !arc.enum<C : si32, D : si32>

      "arc.block.result"(%isA) : (i1) -> ()
    }) : (i1) -> (i1)
    return
  }

  func.func @init(%this : !arc.struct<x : si32>) -> ()
                attributes { "arc.mod_name" = "my_task",
                             "arc.task_name" = "MyTask",
			     "arc.is_init"}
  {
    return
  }
}
