// Check uniquing and that round-tripping works
// RUN: arc-mlir -canonicalize %s | arc-mlir

module @toplevel {

  func.func @ok0(%in : !arc.struct<foo : i32>) -> () {
    return
  }

  func.func @ok1(%in : !arc.struct<foo : i32, bar : f32>) -> () {
    return
  }

  func.func @ok2(%in : !arc.struct<foo : i32, bar : f32>) -> !arc.struct<foo : i32, bar : f32> {
    return %in : !arc.struct<foo : i32, bar : f32>
  }

  func.func @ok3(%in : !arc.struct<foo : i32, bar : f32, inner_struct : !arc.struct<nested : i32>>) -> () {
    return
  }

  func.func @ok4(%in : !arc.struct<<foo : i32>>) -> () {
    return
  }

}
