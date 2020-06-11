# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

let a_i32 : i32 = 65;
let b_i32 : i32 = 66;
let c_i32 : i32 = 67;

let true_bool : bool = true;
let false_bool : bool = false;

# Check that the conditionals end up in the right places and that the
# results in the branches are in the correct order

#CHECK-DAG: [[A:%[^ ]+]] = arc.constant 65 : si32
#CHECK-DAG: [[B:%[^ ]+]] = arc.constant 66 : si32
#CHECK-DAG: [[C:%[^ ]+]] = arc.constant 67 : si32
#CHECK-DAG: [[CONDOUTER:%[^ ]+]] = constant true
#CHECK-DAG: [[CONDINNER:%[^ ]+]] = constant false

if(true_bool, a_i32, if(false_bool, b_i32, c_i32))

#CHECK: {{%[^ ]+}} = "arc.if"([[CONDOUTER]]) (
#CHECK: {{%[^ ]+}} = "arc.block.result"([[A]])
#CHECK: [[INNERRESULT:%[^ ]+]] = "arc.if"([[CONDINNER]]) (
#CHECK: {{%[^ ]+}} = "arc.block.result"([[B]])
#CHECK: {{%[^ ]+}} = "arc.block.result"([[C]])
#CHECK: {{%[^ ]+}} = "arc.block.result"([[INNERRESULT]])
