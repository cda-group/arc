//===- Rust dialect operation definitions -------------------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 KTH Royal Institute of Technology.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines the operations of the Rust dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RUST_OPS
#define RUST_OPS

#ifndef OP_BASE
include "../../mlir/include/mlir/IR/OpBase.td"
include "../../mlir/include/mlir/IR/FunctionInterfaces.td"
include "../../mlir/include/mlir/Interfaces/CallInterfaces.td"
include "../../mlir/include/mlir/Interfaces/InferTypeOpInterface.td"
include "../../mlir/include/mlir/Interfaces/SideEffectInterfaces.td"
include "../../mlir/include/mlir/IR/SymbolInterfaces.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Rust Dialect
//===----------------------------------------------------------------------===//

def Rust_Dialect : Dialect {
  let name = "rust";
  let cppNamespace = "rust";
  let extraClassDeclaration = [{
      Type parseType(DialectAsmParser &parser) const override;
      void printType(Type type, DialectAsmPrinter &os) const override;
      types::RustType floatTy, doubleTy, boolTy,
                      float16Ty, bFloat16Ty,
                      i8Ty, i16Ty, i32Ty, i64Ty,
                      u8Ty, u16Ty, u32Ty, u64Ty,
                      noneTy;
  }];
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// Rust Operation Classes
//===----------------------------------------------------------------------===//

class Rust_Op<string mnemonic, list<Trait> traits = []>
    : Op<Rust_Dialect, mnemonic, traits>;

def AnyRustType : Type<CPred<"$_self.isa<RustType>() || $_self.isa<RustStructType>() || $_self.isa<RustEnumType>() || isRustFunctionType($_self) || $_self.isa<RustStreamType>() || $_self.isa<RustSinkStreamType>() || $_self.isa<RustSourceStreamType>() || $_self.isa<RustGenericADTType>()">, "any RustType">;

def AnyRustStream :
    Type<CPred<"$_self.isa<RustStreamType>()">, "any RustStream">;

def AnyRustSinkStream :
    Type<CPred<"$_self.isa<RustSinkStreamType>()">, "any RustSinkStream">;

def AnyRustSourceStream :
    Type<CPred<"$_self.isa<RustSourceStreamType>()">, "any RustSourceStream">;

def AnyRustStruct :
    Type<CPred<"$_self.isa<RustStructType>()">, "any RustStruct">;

def AnyRustEnum :
    Type<CPred<"$_self.isa<RustEnumType>()">, "any RustEnum">;

def BoolRustType : Type<CPred<"$_self.isa<RustType>() && $_self.cast<RustType>().isBool()">, "a Rust boolean type">;

//===----------------------------------------------------------------------===//
// Rust Operations
//===----------------------------------------------------------------------===//
def RustCallOp : Rust_Op<"call", [CallOpInterface]> {
    // Stolen in large parts from std.call
  let summary = "call operation";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state,
            SymbolRefAttr::get($_state.getContext(), callee), results,
            operands);
    }]>];

  let extraClassDeclaration = [{
    // For the CallOpInterface
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];

}

def RustCallIndirectOp : Rust_Op<"call_indirect", [
      CallOpInterface,
      TypesMatchWith<"callee input types match argument types",
                     "callee", "ops",
                     "$_self.cast<FunctionType>().getInputs()">,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "$_self.cast<FunctionType>().getResults()">
    ]> {
  // Stolen from std.call
  let summary = "indirect call operation";
  let description = [{
    The `call_indirect` operation represents an indirect call to a value of
    function type. Functions are first class types in MLIR, and may be passed as
    arguments and merged together with block arguments. The operands and result
    types of the call must match the specified function type.

    Function values can be created with the
    [`constant` operation](#stdconstant-constantop).

    Example:

    ```mlir
    %31 = call_indirect %15(%0, %1)
            : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    ```
  }];

  let arguments = (ins FunctionType:$callee, Variadic<AnyType>:$ops);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$ops), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(ops);
      $_state.addTypes(callee.getType().cast<FunctionType>().getResults());
    }]>];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}


def Rust_RustFuncOp : Rust_Op<"func",
  [FunctionOpInterface, IsolatedFromAbove, Symbol]> {
  let summary = "A function";

  let description = [{
  }];

  let regions = (region AnyRegion:$body);

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() {
      return getFunctionType().cast<FunctionType>().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().cast<FunctionType>().getResults();
    }

    /// Hook for FunctionOpInterface verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();

    /// Verifies the structure of a JSON attribute.
    LogicalResult verifyValidJSONAttrib(std::string attribName,
                                        unsigned noofParams);

    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

def Rust_RustExtFuncOp : Rust_Op<"extfunc",
  [FunctionOpInterface, IsolatedFromAbove, Symbol]> {
  let summary = "An external function";

  let description = [{
  }];

  let regions = (region AnyRegion:$empty_body);
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() {
      return getFunctionType().cast<FunctionType>().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().cast<FunctionType>().getResults();
    }

    /// Hook for FunctionOpInterface verifier.
    LogicalResult verifyType();

    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

// Stolen from SCF
def Rust_RustLoopOp : Rust_Op<"loop", [RecursiveMemoryEffects]> {
  let summary = "A Rust loop ";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyRustType>:$inits);
  let results = (outs Variadic<AnyRustType>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

  let extraClassDeclaration = [{
    RustLoopConditionOp getConditionOp();
    RustLoopYieldOp getYieldOp();

    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];

}

// Stolen from SCF
def Rust_RustLoopConditionOp : Rust_Op<"loop.condition", [
  HasParent<"RustLoopOp">,
  Pure,
  Terminator
]> {
  let summary = "loop continuation condition";
  let description = [{
    Corresponds to the scf.condition operation.
  }];

  let arguments = (ins BoolRustType:$condition, Variadic<AnyRustType>:$args);

  let extraClassDeclaration = [{
    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

// Stolen from SCF
def Rust_RustLoopYieldOp : Rust_Op<"loop.yield",
                              [Pure, Terminator,
                               HasParent<"RustLoopOp">]> {
  let summary = "loop yield and termination operation";
  let description = [{
    As "scf.yield".
  }];

  let arguments = (ins Variadic<AnyRustType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let extraClassDeclaration = [{
    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

def Rust_RustLoopBreakOp : Rust_Op<"loop.break",
                              [Pure, Terminator]> {
  let summary = "break out of a loop";
  let description = [{
     Break out of a loop producing a result
  }];

  let arguments = (ins Variadic<AnyRustType>:$results);

  let extraClassDeclaration = [{
    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];

  let hasVerifier = 1;
}

def Rust_RustReturnOp : Rust_Op<"return", [Terminator]> {
  let summary = "Terminator for Rust functions.";
  let description = [{
    A terminator operation for regions that appear in the body of  `rust.func`
    functions. The operands to the `rust.return` are the result values returned
    by an incovation of the `rust.func`.
  }];

  let arguments = (ins Optional<AnyRustType>:$returned_value);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def RustSpawnOp : Rust_Op<"spawn", [CallOpInterface]> {
    // Stolen in large parts from std.call
  let summary = "spawn operation";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);

  let extraClassDeclaration = [{
    // For the CallOpInterface
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    // Write this op as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

def Rust_RustConstantOp : Rust_Op<"constant", [Pure]> {
  let summary = "a rust constant.";
  let description = [{
    A Rust constant, the value attribute is the string representation of the
    constant in Rust syntax.
  }];

  let arguments = (ins StrAttr : $value);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    StringAttr getValueAsStringAttr() {
      return (*this)->getAttr("value").dyn_cast<StringAttr>();
    }
    // Write this constant as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustEnumAccessOp : Rust_Op<"enum_access", [Pure]> {
  let summary = "Access to a variant of an enum";
  let description = [{
    Access to a variant of an enum.
  }];

  let arguments = (ins AnyRustEnum : $theEnum, StrAttr : $variant);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    // Write this constant as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustEnumCheckOp : Rust_Op<"enum_check", [Pure]> {
  let summary = "Check if an enum is of a particular variant";
  let description = [{
   A pure operation which checks if an enum is of a particular variant.
  }];

  let arguments = (ins AnyRustEnum : $theEnum, StrAttr : $variant);
  let results = (outs BoolRustType);

  let extraClassDeclaration = [{
    // Write this constant as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustFieldAccessOp : Rust_Op<"field_access", [Pure]> {
  let summary = "Access to a field";
  let description = [{
    Access to a field, for example in a struct.
  }];

  let arguments = (ins AnyRustType : $aggregate, StrAttr : $field);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    // Write this constant as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustUnaryOp : Rust_Op<"unaryop", [SameOperandsAndResultType]> {
  let summary = "a Rust unary operation.";
  let description = [{
    A Rust unary operation, the op attribute is the string representation of
    the operator in Rust syntax.
  }];

  let arguments = (ins StrAttr : $op, AnyRustType : $arg);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    StringRef getOperator() {
      return (*this)->getAttrOfType<StringAttr>("op").getValue();
    }

    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustMethodCallOp : Rust_Op<"method_call", []> {
  let summary = "a call to a Rust method on an object.";
  let description = [{
    A Rust method call, calls the `method` method of the object `obj`.
  }];

  let arguments = (ins StrAttr : $method, AnyRustType : $obj,
                   Variadic<AnyRustType>:$ops);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustBinaryOp : Rust_Op<"binaryop", [SameTypeOperands]> {
  let summary = "a Rust binary operation.";
  let description = [{
    A Rust binary operation, the op attribute is the string representation of
    the operator in Rust syntax.
  }];

  let arguments = (ins StrAttr : $op, AnyRustType : $LHS, AnyRustType : $RHS);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    StringRef getOperator() {
      return (*this)->getAttrOfType<StringAttr>("op").getValue();
    }

    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustBinaryRcOp : Rust_Op<"binaryrcop", [SameTypeOperands]> {
  let summary = "a Rust binary operation operating on a Rc value.";
  let description = [{
    A Rust binary operation, the op attribute is the string representation of
    the operator in Rust syntax. The operands will be dereferenced and the
    result wrapped in a Rc.
  }];

  let arguments = (ins StrAttr : $op, AnyRustType : $LHS, AnyRustType : $RHS);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    StringRef getOperator() {
      return (*this)->getAttrOfType<StringAttr>("op").getValue();
    }

    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustCompOp : Rust_Op<"compop", [SameTypeOperands]> {
  let summary = "a Rust comparison operation.";
  let description = [{
    A Rust comparison operation, the predicate attribute is the string
    representation of the operator in Rust syntax.
  }];

  let arguments = (ins StrAttr : $op, AnyRustType : $LHS, AnyRustType : $RHS);
  let results = (outs BoolRustType);

  let extraClassDeclaration = [{
    StringRef getOperator() {
      return (*this)->getAttrOfType<StringAttr>("op").getValue();
    }

    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_FilterOp : Rust_Op<"filter", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Standard filter dataflow op";
  let description = [{
    Standard filter dataflow op.
  }];

  let arguments = (ins
      AnyRustSourceStream:$input,
      FlatSymbolRefAttr:$predicate,
      OptionalAttr<FlatSymbolRefAttr>:$predicate_env_thunk
  );
  let results = (outs AnyRustSourceStream:$output);
}

def Rust_RustBlockResultOp
    : Rust_Op<"block.result",
              [HasParent<"RustIfOp">, Terminator]> {
  let summary = "specifies the value of a block";
  let description = [{
    "rust.block.result" is a special terminator operation for the block inside
    "rust.if". It terminates the region. It should have the same type as the
    parent if.

    ```mlir
      "rust.block.result"(%b) : (!rust<"f64">) -> ()
    ```
  }];
  let arguments = (ins Variadic<AnyRustType> : $result);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustIfOp
    : Rust_Op<"if", [SingleBlock]> {
  let summary = "if-then-else operation";
  let description = [{
    The "rust.if" operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. The operation produces a single result. For example:

    ```mlir
    "rust.if"(%a) ( {
      "rust.block.result"(%b) : (!rust<"f64">) -> !rust<"f64">
    },  {
      "rust.block.result"(%c) : (!rust<"f64">) -> !rust<"f64">
    }) : (!rust<"bool">) -> !rust<"f64">
    ```
  }];
  let arguments = (ins BoolRustType : $condition);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);
  let results = (outs Variadic<AnyRustType>);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
  let hasVerifier = 1;
}

def Rust_RustMakeEnumOp
    : Rust_Op<"make_enum", []> {
  let summary = "constructs an enum";
  let description = [{
    "rust.enum" constructs an enum
  }];
  let arguments = (ins Variadic<AnyRustType>:$values, StrAttr:$variant);
  let results = (outs AnyRustEnum);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustMakeStructOp
    : Rust_Op<"make_struct", []> {
  let summary = "constructs a struct";
  let description = [{
    "rust.struct" constructs a struct
  }];
  let arguments = (ins Variadic<AnyRustType>:$operands);
  let results = (outs AnyRustStruct);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustReceiveOp
    : Rust_Op<"receive", []> {
  let summary = "Receive a value from a stream.";
  let arguments = (ins AnyRustSourceStream:$source);
  let results = (outs AnyRustType:$value);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustSendOp
    : Rust_Op<"send", []> {
  let summary = "Send data on a sink stream";
  let arguments = (ins AnyRustType:$value, AnyRustSinkStream:$sink);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustPanicOp : Rust_Op<"panic", []> {
  let summary = "Trigger a run-time panic";
  let description = [{
    Trigger a run-time panic
  }];
  let arguments = (ins OptionalAttr<StrAttr>:$msg);
  let results = (outs );

  let assemblyFormat = [{
    `(` ($msg^)? `)` attr-dict
  }];
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_MapOp : Rust_Op<"map", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Standard map dataflow op";
  let description = [{
    Standard map dataflow op.
  }];

  let arguments = (ins
      AnyRustSourceStream:$input,
      FlatSymbolRefAttr:$map_fun,
      OptionalAttr<FlatSymbolRefAttr>:$map_fun_env_thunk
  );
  let results = (outs AnyRustSourceStream:$output);
}

#endif // RUST_OPS
