# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

let a_i32 : i32 = 65;
let b_i32 : i32 = 66;
let c_i32 : i32 = 67;

let true_bool : bool = true;
let false_bool : bool = false;

# Check that the conditionals end up in the right places and that the
# results in the branches are in the correct order

#CHECK-DAG: [[A:%[^ ]+]] = constant 65 : i32
#CHECK-DAG: [[B:%[^ ]+]] = constant 66 : i32
#CHECK-DAG: [[C:%[^ ]+]] = constant 67 : i32
#CHECK-DAG: [[CONDOUTER:%[^ ]+]] = constant 1 : i1
#CHECK-DAG: [[CONDINNER:%[^ ]+]] = constant 0 : i1

if(true_bool, a_i32, if(false_bool, b_i32, c_i32))

#CHECK: {{%[^ ]+}} = "arc.if"([[CONDOUTER]]) (
#CHECK: {{%[^ ]+}} = "arc.yield"([[A]])
#CHECK: [[INNERRESULT:%[^ ]+]] = "arc.if"([[CONDINNER]]) (
#CHECK: {{%[^ ]+}} = "arc.yield"([[B]])
#CHECK: {{%[^ ]+}} = "arc.yield"([[C]])
#CHECK: {{%[^ ]+}} = "arc.yield"([[INNERRESULT]])
