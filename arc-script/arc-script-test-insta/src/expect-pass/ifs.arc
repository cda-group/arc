# RUN: arc-script run --output=MLIR %s | FileCheck %s
# RUN: arc-script run --output=MLIR %s | arc-mlir | FileCheck %s

fun main() { () }

fun test() -> i32 {
  let a_i32: i32 = 65 in
  let b_i32: i32 = 66 in

  let true_bool: bool = true in

# Check that the conditional ends up in the right place and that the
# results in the two branches are correct

#CHECK-DAG: [[A:%[^ ]+]] = arc.constant 65 : si32
#CHECK-DAG: [[B:%[^ ]+]] = arc.constant 66 : si32
#CHECK-DAG: [[COND:%[^ ]+]] = constant true

  if true_bool {
      a_i32
  } else {
      b_i32
  }
}

#CHECK: {{%[^ ]+}} = "arc.if"([[COND]]) (
#CHECK: "arc.block.result"([[A]])
#CHECK: "arc.block.result"([[B]])
