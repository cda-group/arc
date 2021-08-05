# RUN: arc-script --no-prelude run --output=MLIR %s | FileCheck %s
# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir | FileCheck %s

fun test(): i32 {

    val a_i32: i32 = 65;
    val b_i32: i32 = 66;
    val c_i32: i32 = 67;

    val true_bool: bool = true;
    val false_bool: bool = false;

# Check that the conditionals end up in the right places and that the
# results in the branches are in the correct order

#CHECK-DAG: [[A:%[^ ]+]] = arc.constant 65 : si32
#CHECK-DAG: [[B:%[^ ]+]] = arc.constant 66 : si32
#CHECK-DAG: [[C:%[^ ]+]] = arc.constant 67 : si32
#CHECK-DAG: [[CONDOUTER:%[^ ]+]] = constant true
#CHECK-DAG: [[CONDINNER:%[^ ]+]] = constant false

    if true_bool {
        a_i32
    } else {
        if false_bool {
          b_i32
        } else {
          c_i32
        }
    }

#CHECK: {{%[^ ]+}} = "arc.if"([[CONDOUTER]]) (
#CHECK: "arc.block.result"([[A]])
#CHECK: [[INNERRESULT:%[^ ]+]] = "arc.if"([[CONDINNER]]) (
#CHECK: "arc.block.result"([[B]])
#CHECK: "arc.block.result"([[C]])
#CHECK: "arc.block.result"([[INNERRESULT]])
}
