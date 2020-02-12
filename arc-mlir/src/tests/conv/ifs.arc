# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

let a_i32 : i32 = 65;
let b_i32 : i32 = 66;

let true_bool : bool = true;

# Check that the conditional ends up in the right place and that the
# results in the two branches are correct

#CHECK-DAG: [[A:%[^ ]+]] = constant 65 : i32
#CHECK-DAG: [[B:%[^ ]+]] = constant 66 : i32
#CHECK-DAG: [[COND:%[^ ]+]] = constant 1 : i1

if(true_bool, a_i32, b_i32)

#CHECK: {{%[^ ]+}} = "arc.if"([[COND]]) (
#CHECK: {{%[^ ]+}} = "arc.block.result"([[A]])
#CHECK: {{%[^ ]+}} = "arc.block.result"([[B]])
