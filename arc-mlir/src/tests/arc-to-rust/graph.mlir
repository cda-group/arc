// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize

module @program {
    func.func @_f0(%p: !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>, %env: !arc.struct<>) -> i1 {
        %_x0 = "arc.struct_access"(%p) {field="age"} : (!arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>) -> si32
        %_x1 = arc.constant 10 : si32
        %_x2 = arc.cmpi "gt", %_x0, %_x1 : si32
        %_x3 = "arc.struct_access"(%p) {field="age"} : (!arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>) -> si32
        %_x4 = arc.constant 100 : si32
        %_x5 = arc.cmpi "lt", %_x3, %_x4 : si32
        %_x6 = arith.andi %_x2, %_x5 : i1
        return %_x6 : i1
    }

    func.func @_f1(%p: !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>, %env: !arc.struct<>) -> !arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32> {
        %_x0 = "arc.struct_access"(%p) {field="name"} : (!arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>) -> !arc.adt<"Str">
        %_x1 = "arc.struct_access"(%p) {field="height"} : (!arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>) -> si32
        %_x2 = "arc.struct_access"(%p) {field="weight"} : (!arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>) -> si32
        %_x3 = arc.make_struct(%_x0, %_x1, %_x2 : !arc.adt<"Str">, si32, si32) : !arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>
        return %_x3 : !arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>
    }

    func.func @_f2(%p: !arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>, %env: !arc.struct<>) -> !arc.struct<name:!arc.adt<"Str">> {
        %_x0 = "arc.struct_access"(%p) {field="name"} : (!arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>) -> !arc.adt<"Str">
        %_x1 = "arc.struct_access"(%p) {field="height"} : (!arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>) -> si32
        %_x2 = "arc.struct_access"(%p) {field="weight"} : (!arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>) -> si32
        %_x3 = arc.constant 2 : si32
        %_x4 = arc.muli %_x1, %_x1 : si32
        %_x5 = arc.divi %_x2, %_x3 : si32
        %_x6 = arc.make_struct (%_x0 : !arc.adt<"Str">) : !arc.struct<name:!arc.adt<"Str">>
        return %_x6 : !arc.struct<name:!arc.adt<"Str">>
    }

    func.func @_f1_thunk() -> !arc.struct<> {
        %env_for_f1 = arc.make_struct() : !arc.struct<>
	return %env_for_f1 : !arc.struct<>
    }

    func.func @_f2_thunk() -> !arc.struct<> {
        %env_for_f2 = arc.make_struct() : !arc.struct<>
	return %env_for_f2 : !arc.struct<>
    }

    func.func @_f0_env() -> !arc.struct<> {
      %env_for_f0 = arc.make_struct() : !arc.struct<>
      return %env_for_f0 : !arc.struct<>
  }

  func.func @flip(%in : !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>) -> ui32 {
    %r = arc.constant 1 : ui32
    return %r : ui32
  }

  func.func @graph(%a : !arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>,
                   %b : !arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>)
		   -> !arc.stream<ui32, !arc.struct<name:!arc.adt<"Str">>>
        attributes { arc.is_graph,
		     arc.source_params="{\"0\": {\"arg0\":[\"localhost:8080\", \"pl\\\"o\\\"nk\"]},\"1\": {\"arg0\":[\"localhost:8083\", \"plutt\"]}}",
		     arc.sink_params="{\"0\": {\"arg0\": [\"localhost:8081\", \"plunk\"], \"arg1\" : 4711}}" } {
	%c = "arc.union"(%a, %b) : (!arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>,
                                    !arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>)
				    -> !arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>
	%_x1 = "arc.keyby"(%c) { key_fun=@flip} : (!arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>)
	       		       	 		   -> !arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>
        %_x5 = "arc.filter"(%_x1) {predicate=@_f0,predicate_env_thunk=@_f0_env} : (!arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>) ->
	       !arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>

	%_x9 = "arc.map"(%_x5) {map_fun=@_f1, map_fun_env_thunk=@_f1_thunk} : (!arc.stream<ui32, !arc.struct<age:si32, name:!arc.adt<"Str">, height:si32, weight:si32>>) ->
	     !arc.stream<ui32, !arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>>

	%_x13 = "arc.map"(%_x9) {map_fun=@_f2, map_fun_env_thunk=@_f2_thunk} : (!arc.stream<ui32, !arc.struct<name:!arc.adt<"Str">, height:si32, weight:si32>>) ->
	      !arc.stream<ui32, !arc.struct<name:!arc.adt<"Str">>>

	return %_x13 : !arc.stream<ui32, !arc.struct<name:!arc.adt<"Str">>>
    }
}
