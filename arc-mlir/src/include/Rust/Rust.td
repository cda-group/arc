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
include "../../mlir/include/mlir/Interfaces/CallInterfaces.td"
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
                      i8Ty, i16Ty, i32Ty, i64Ty,
                      u8Ty, u16Ty, u32Ty, u64Ty;
  }];

}

//===----------------------------------------------------------------------===//
// Rust Operation Classes
//===----------------------------------------------------------------------===//

class Rust_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Rust_Dialect, mnemonic, traits>;

def AnyRustType : Type<CPred<"$_self.isa<RustType>() || $_self.isa<RustStructType>() || $_self.isa<RustTupleType>() || $_self.isa<RustTensorType>()">, "any RustType">;
def AnyRustStruct :
    Type<CPred<"$_self.isa<RustStructType>()">, "any RustStruct">;

def BoolRustType : Type<CPred<"$_self.isa<RustType>() && $_self.cast<RustType>().isBool()">, "a Rust boolean type">;

def RustTensorElementType :
    Type<CPred<"$_self.isa<RustType>()">, "a rust tensor element type">;

def RustTensorType :
    Type<CPred<"$_self.isa<RustTensorType>()">, "a rust tensor element type">;


//===----------------------------------------------------------------------===//
// Rust Operations
//===----------------------------------------------------------------------===//
def RustCallOp : Rust_Op<"call", [CallOpInterface]> {
    // Stolen in large parts from std.call
  let summary = "call operation";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilderDAG<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilderDAG<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
    }]>];

  let extraClassDeclaration = [{
    // For the CallOpInterface
    StringRef getCallee() { return callee(); }
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

def Rust_RustFuncOp : Rust_Op<"func", [FunctionLike, IsolatedFromAbove, Symbol]> {
  let summary = "A function";

  let description = [{
  }];

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the type of the function this Op defines.
    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<RustFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();

    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

def Rust_RustExtFuncOp : Rust_Op<"extfunc", [FunctionLike, IsolatedFromAbove, Symbol]> {
  let summary = "An external function";

  let description = [{
  }];

  let arguments = (ins StrAttr : $language);

  let regions = (region AnyRegion:$empty_body);

  let extraClassDeclaration = [{
    /// Returns the type of the function this Op defines.
    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    // FunctionLike trait needs access to the functions below.
    friend class OpTrait::FunctionLike<RustExtFuncOp>;

    /// Hooks for the input/output type enumeration in FunctionLike .
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    /// Hook for FunctionLike verifier.
    LogicalResult verifyType();

    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &os);
  }];
}

def Rust_RustReturnOp : Rust_Op<"return", [HasParent<"RustFuncOp">,
    Terminator]> {
  let summary = "Terminator for Rust functions.";
  let description = [{
    A terminator operation for regions that appear in the body of  `rust.func`
    functions. The operands to the `rust.return` are the result values returned
    by an incovation of the `rust.func`.
  }];

  let arguments = (ins Optional<AnyRustType>:$operands);

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    // Write this function as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustConstantOp : Rust_Op<"constant", [NoSideEffect]> {
  let summary = "a rust constant.";
  let description = [{
    A Rust constant, the value attribute is the string representation of the
    constant in Rust syntax.
  }];

  let arguments = (ins StrAttr : $value);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
    // Write this constant as Rust code to os
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustFieldAccessOp : Rust_Op<"field_access", [NoSideEffect]> {
  let summary = "Access to a field";
  let description = [{
    Access to a field, for example in a tuple or a struct.
  }];

  let arguments = (ins AnyRustType : $aggregate, StrAttr : $field);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    StringRef getField() {
      return (*this)->getAttrOfType<StringAttr>("field").getValue();
    }
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
                   Variadic<AnyRustType>:$operands);
  let results = (outs AnyRustType);

  let extraClassDeclaration = [{
    StringRef getMethod() {
      return (*this)->getAttrOfType<StringAttr>("method").getValue();
    }

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
  let arguments = (ins AnyRustType : $result);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustIfOp
    : Rust_Op<"if", [SingleBlockImplicitTerminator<"RustBlockResultOp">]> {
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
  let results = (outs AnyRustType);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustDependencyOp : Rust_Op<"dependency", [NoSideEffect]> {
  let summary = "Specify a dependency on a crate.";
  let description = [{
    This is used to specify a dependency on an external crate.
  }];

  let arguments = (ins StrAttr : $crate, StrAttr : $version);
  let results = (outs);

  let extraClassDeclaration = [{
    Attribute getCrate() { return (*this)->getAttr("crate"); }
    Attribute getVersion() { return (*this)->getAttr("version"); }
  }];
}

def Rust_RustMakeStructOp
    : Rust_Op<"make_struct", []> {
  let summary = "constructs a struct";
  let description = [{
    "rust.struct" constructs a tuple
  }];
  let arguments = (ins Variadic<AnyRustType>:$operands);
  let results = (outs AnyRustStruct);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}

def Rust_RustModuleDirectiveOp : Rust_Op<"module_directive", [NoSideEffect]> {
  let summary = "Rust-level module directives.";
  let description = [{
    The specified string will be output at the top of the Rust output file.
    Strings with the same key will be de-duplicated.
  }];

  let arguments = (ins StrAttr : $key, StrAttr : $str);
  let results = (outs);

  let extraClassDeclaration = [{
    Attribute getKey() { return (*this)->getAttr("key"); }
    Attribute getStr() { return (*this)->getAttr("str"); }
  }];
}

def Rust_RustTensorOp : Rust_Op<"make_tensor", []> {
  let summary = "create a tensor from a sequence of values";
  let description = [{
    A pure operation which turns its arguments into a tensor. The
    tensor values are given in column-major order
  }];
  let arguments = (ins Variadic<RustTensorElementType>:$values);
  let results = (outs RustTensorType:$result);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}



def Rust_RustTupleOp
    : Rust_Op<"tuple", []> {
  let summary = "constructs a tuple";
  let description = [{
    "rust.tuple" constructs a tuple
  }];
  let arguments = (ins Variadic<AnyRustType>:$operands);
  let results = (outs AnyRustType);
  let extraClassDeclaration = [{
    // Write this operation as Rust code to the stream
    void writeRust(RustPrinterStream &);
  }];
}
#endif // RUST_OPS
