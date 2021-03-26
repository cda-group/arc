# RUN: arc-script run --output=MLIR %s | FileCheck %s
# RUN: arc-script run --output=MLIR %s | arc-mlir | FileCheck %s

fun test(): i32 {

    let a_i32: i32 = 65 in
    let b_i32: i32 = 66 in
    let c_i32: i32 = 67 in

    let true_bool: bool = true in
    let false_bool: bool = false in

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
