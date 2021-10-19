# RUN: arc-script --no-prelude run --output=MLIR %s | FileCheck %s
# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir | FileCheck %s

fun main() { unit }

fun test(): i32 {
  val a_i32: i32 = 65;
  val b_i32: i32 = 66;

  val true_bool: bool = true;

# Check that the conditional ends up in the right place and that the
# results in the two branches are correct

#CHECK-DAG: [[A:%[^ ]+]] = arc.constant 65 : si32
#CHECK-DAG: [[B:%[^ ]+]] = arc.constant 66 : si32
#CHECK-DAG: [[COND:%[^ ]+]] = arith.constant true

  if true_bool {
      a_i32
  } else {
      b_i32
  }
}

#CHECK: {{%[^ ]+}} = "arc.if"([[COND]]) (
#CHECK: "arc.block.result"([[A]])
#CHECK: "arc.block.result"([[B]])
