//===- Arc dialect operation definitions -------------------*- tablegen -*-===//
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
// Defines the operations of the Arc dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARC_OPS
#define ARC_OPS

#ifndef OP_BASE
include "../../mlir/include/mlir/IR/EnumAttr.td"
include "../../mlir/include/mlir/IR/OpBase.td"
include "../../mlir/include/mlir/IR/SymbolInterfaces.td"
include "../../mlir/include/mlir/IR/BuiltinAttributeInterfaces.td"
include "../../mlir/include/mlir/Interfaces/CallInterfaces.td"
include "../../mlir/include/mlir/Interfaces/InferTypeOpInterface.td"
include "../../mlir/include/mlir/Interfaces/SideEffectInterfaces.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Arc Dialect
//===----------------------------------------------------------------------===//

def Arc_Dialect : Dialect {
  let name = "arc";
  let cppNamespace = "arc";
  let extraClassDeclaration = [{
      Type parseType(DialectAsmParser &parser) const override;
      void printType(Type type, DialectAsmPrinter &os) const override;
  }];
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Arc Operation Classes
//===----------------------------------------------------------------------===//

class Arc_Op<string mnemonic, list<Trait> traits = []>
    : Op<Arc_Dialect, mnemonic, traits>;

/* The structure for these unary operands is stolen from the standard dialect */
class Arc_UnaryOp<string mnemonic, list<Trait> traits = []>
    : Op<Arc_Dialect, mnemonic, !listconcat(traits, [Pure])> {
  let results = (outs AnyType);
}

class Arc_UnaryOpSameOperandAndResultType<string mnemonic,
                                         list<Trait> traits = []>
    : Arc_UnaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
}

class Arc_FloatUnaryOp<string mnemonic, list<Trait> traits = []>
    : Arc_UnaryOpSameOperandAndResultType<mnemonic, traits>,
      Arguments<(ins FloatLike:$operand)>,
      Results<(outs FloatLike:$result)> {
    let assemblyFormat = "$operand attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Arc Predicates
//===----------------------------------------------------------------------===//

def AnyArcInteger : Type<Or<[
  AnyUnsignedInteger.predicate,
  AnySignedInteger.predicate,
]>, "any Arc integer type">;

def AnyValue : Type<Or<[
  AnySignlessInteger.predicate, // TODO: Should this be replaced with
                                // an AnyArcInteger?
  AnyFloat.predicate,
]>, "any value type">;

// Type constraint for arc integer-like types: arc-integers integers, indices,
// vectors of arc integers of arc integers.
def ArcIntegerLike : TypeConstraint<Or<[
        AnyArcInteger.predicate, Index.predicate,
        VectorOf<[AnyArcInteger]>.predicate]>,
    "arc-integer-like">;

def ADTType : Type<Or<[
        CPred<"$_self.isa<ADTType>()">,
        CPred<"$_self.isa<ADTGenericType>()">]>,
    "arc-adt">;
def ADTSimpleType : Type<CPred<"$_self.isa<ADTType>()">, "an ADT type">;
def ADTGenericType : Type<CPred<"$_self.isa<ADTGenericType>()">, "an ADT template type">;

def AnyBuilder : Type<CPred<"$_self.isa<BuilderType>()">, "any builder">;
def AnyEnum : Type<CPred<"$_self.isa<EnumType>()">, "any enum">;
def AnyStruct : Type<CPred<"$_self.isa<StructType>()">, "any struct">;
def ArcSinkStream : Type<CPred<"$_self.isa<SinkStreamType>()">,
                               "a sink stream">;
def ArcSourceStream : Type<CPred<"$_self.isa<SourceStreamType>()">,
                                 "a source stream">;
def ArcStream : Type<CPred<"$_self.isa<StreamType>()">, "a stream">;
def ArcStateValue : Type<CPred<"$_self.isa<ArconValueType>()">,
                                "an arcon value type">;
def ArcStateMap : Type<CPred<"$_self.isa<ArconMapType>()">,
                             "an arcon map type">;

def StreamElementType : Type<Or<[
  FloatLike.predicate,
  ArcIntegerLike.predicate,
  AnyStruct.predicate,
  AnyEnum.predicate
]>, "any type valid in a stream">;


//===----------------------------------------------------------------------===//
// Arc Traits
//===----------------------------------------------------------------------===//

def Linear : PredOpTrait<
  "result must have exactly one use",
  CPred<"$_op.getResult(0).hasOneUse()">
>;

def Affine : PredOpTrait<
  "result must have at most one use",
  Or<[
    CPred<"$_op.getResult(0).use_empty()">,
    CPred<"$_op.getResult(0).hasOneUse()">
  ]>
>;

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//

def ArcKeepOp : Arc_Op<"keep", []> {
  let summary = "Operation consuming anything and having side effects.";
  let description = [{
    An operation used for debugging and testing. Can be used to prevent
    optimization passes from removing dummy results.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
}

def TanOp : Arc_FloatUnaryOp<"tan"> {
  let summary = "tangent of the specified value";
  let description = [{
    The `tan` operation computes the tangent of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float. It has no standard attributes.
  }];
}

def AcosOp : Arc_FloatUnaryOp<"acos"> {
  let summary = "arcus cosine of the specified value";
  let description = [{
    The `sin` operation computes the arcus cosine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float. It has no standard attributes.
  }];
}

def AsinOp : Arc_FloatUnaryOp<"asin"> {
  let summary = "arcus sine of the specified value";
  let description = [{
    The `sin` operation computes the arcus sine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float. It has no standard attributes.
  }];
}

def CoshOp : Arc_FloatUnaryOp<"cosh"> {
  let summary = "hyperbolic cosine of the specified value";
  let description = [{
    The `cosh` operation computes the hyperbolic cosine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float. It has no standard attributes.
  }];
}

def SinhOp : Arc_FloatUnaryOp<"sinh"> {
  let summary = "hyperbolic sine of the specified value";
  let description = [{
    The `sinh` operation computes the hyperbolic sine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float. It has no standard attributes.
  }];
}

def ErfOp : Arc_FloatUnaryOp<"erf"> {
  let summary = "error function of the specified value";
  let description = [{
    The `erf' operation computes the Gauss error function of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float. It has no standard attributes.
  }];
}

def MakeStructOp : Arc_Op<"make_struct", [Pure]> {
  let summary = "create a struct from a sequence of values";
  let description = [{
    A pure operation which turns its arguments into a struct.
    ```
    %r = arc.make_struct(%a, %b : i32, f32) : !arc.struct<a : i32, b : f32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyStruct:$result);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` ($values^ `:` type($values) )? `)` attr-dict `:` type($result)
  }];
}

def MakeEnumOp : Arc_Op<"make_enum", [Pure]> {
  let summary = "create an enum from a values";
  let description = [{
    A pure operation which turns its argument into an enum.
    ```
    %r = arc.make_enum "a" (%a : i32) : !arc.enum<a : i32, b : f32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$values, StrAttr:$variant);
  let results = (outs AnyEnum:$result);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;

  let assemblyFormat =
    "`(` ($values^ `:` type($values))? `)` `as` $variant attr-dict `:` type($result)";
}

def ArcBlockResultOp :
    Arc_Op<"block.result", [HasParent<"IfOp">,
                            Terminator]> {
  let summary = "specifies the value of a block";
  let description = [{
    "arc.block.result" is a special terminator operation for the block inside
    "arc.if". It terminates the region. It should have the same type as the
    parent if.

    ```mlir
      "arc.block.result"(%b) : (f32) -> ()
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$result);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;
}

def IfOp : Arc_Op<"if", [SingleBlock]> {
  let summary = "if-then-else operation";
  let description = [{
    The "arc.if" operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. The operation produces a single result. For example:

    ```mlir
    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (f64) -> f64
    },  {
      "arc.block.result"(%c) : (f64) -> f64
    }) : (i1) -> f64
    ```
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);
  let results = (outs Variadic<AnyType>:$result);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def LoopBreakOp : Arc_Op<"loop.break", [Pure, Terminator]> {
  let summary = "break out of a loop";
  let description = [{
     Break out of a loop producing a result
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let hasVerifier = 1;
}

def Arc_ConstantADTOp : Arc_Op<"adt_constant", [Pure]> {
  let summary = "Declare an ADT constant";
  let description = [{
    ```
    %x = arc.adt_constant "some Rust expression" : !arc.adt<"a Rust type">
    ```
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs ADTType:$result);
  let assemblyFormat = " $value attr-dict `:` type($result)";
}

def Arc_ConstantIntOp : Arc_Op<"constant", [ConstantLike, Pure]> {
  let summary = "The op that declares an Arc integer constant";

  let description = [{
  }];

  let arguments = (ins
    TypedAttrInterface:$value
  );

  let results = (outs
    AnyArcInteger:$constant
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Arc_CMPI_P_EQ : I64EnumAttrCase<"eq", 0>;
def Arc_CMPI_P_NE : I64EnumAttrCase<"ne", 1>;
def Arc_CMPI_P_LT : I64EnumAttrCase<"lt", 2>;
def Arc_CMPI_P_LE : I64EnumAttrCase<"le", 3>;
def Arc_CMPI_P_GT : I64EnumAttrCase<"gt", 4>;
def Arc_CMPI_P_GE : I64EnumAttrCase<"ge", 5>;

def Arc_CmpIPredicateAttr : I64EnumAttr<
    "Arc_CmpIPredicate", "",
    [Arc_CMPI_P_EQ, Arc_CMPI_P_NE, Arc_CMPI_P_LT,
     Arc_CMPI_P_LE, Arc_CMPI_P_GT, Arc_CMPI_P_GE]> {
  let cppNamespace = "arc";
}

def Arc_CmpIOp : Arc_Op<"cmpi",
    [Pure, SameTypeOperands, SameOperandsAndResultShape,
     TypesMatchWith<
       "result type has i1 element type and same shape as operands",
       "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "integer comparison operation";
  let description = [{
    As std.cmp but supporting signed/unsigned values.
  }];

  let arguments = (ins
      Arc_CmpIPredicateAttr:$predicate,
      ArcIntegerLike:$lhs,
      ArcIntegerLike:$rhs
  );
  let results = (outs BoolLike:$result);

  let extraClassDeclaration = [{
    static Arc_CmpIPredicate getPredicateByName(StringRef name);
  }];

  let hasFolder = 1;

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Arc_ReceiveOp : Arc_Op<"receive", []> {
  let summary = "Receive a value from a stream";
  let description = [{
    Receive a value from a stream. Must be located inside a task.
  }];

  let arguments = (ins ArcSourceStream:$source);
  let results = (outs StreamElementType:$value);
  let hasVerifier = 1;

  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
}

def Arc_SelectOp : Arc_Op<"select", [Pure, SameOperandsAndResultShape,
     AllTypesMatch<["true_value", "false_value", "result"]>,
     TypesMatchWith<"condition type matches i1 equivalent of result type",
                     "result", "condition",
                     "getI1SameShape($_self)">]> {
  let summary = "select operation";
  let description = [{
    As std.cmp but supporting signed/unsigned values.
  }];

  let arguments = (ins BoolLike:$condition,
                       ArcIntegerLike:$true_value,
                       ArcIntegerLike:$false_value);
  let results = (outs ArcIntegerLike:$result);

  let hasFolder = 1;

  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value attr-dict `:` type($result)
  }];
}

def Arc_SendOp : Arc_Op<"send", []> {
  let summary = "Send a value on a stream";
  let description = [{
    Send a value on a stream
  }];

  let arguments = (ins
      StreamElementType:$value,
      ArcSinkStream:$sink
  );
  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
}

def EnumAccessOp : Arc_Op<"enum_access", [Pure]> {
  let summary = "access a variant of an enum";
  let description = [{
    A pure operation which acccesses a variant of an enum.

    ```
    %e = arc.make_enum "a" (%a : i32) : !arc.enum<a : i32, b : f32>
    %v = arc.enum_access "a" in %e : i32
    ```
  }];
  let arguments = (ins AnyEnum:$value, StrAttr:$variant);
  let results = (outs AnyType:$result);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat =
    "$variant `in` `(` $value `:` type($value) `)` attr-dict `:` type($result)";
}

def EnumCheckOp : Arc_Op<"enum_check", [Pure]> {
  let summary = "Check if an enum is of a particular variant";
  let description = [{
    A pure operation which checks if an enum is of a particular variant.

    ```
    %e = arc.make_enum "a" (%a : i32) : !arc.enum<a : i32, b : f32>
    %v = arc.enum_check (%a : i32) is "a"
    ```
  }];
  let arguments = (ins AnyEnum:$value, StrAttr:$variant);
  let results = (outs BoolLike:$result);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat =
    "`(` $value `:` type($value) `)` `is` $variant attr-dict `:` type($result)";
}

def StructAccessOp : Arc_Op<"struct_access", [Pure]> {
  let summary = "access a field of a struct";
  let description = [{
    A pure operation which acccesses a field of a struct.

    ```
    %s = arc.make_struct(%x : si32) : !arc.struct<x : si32>
    %elem = "arc.struct_access"(%s) { field = "x" } : (!arc.struct<x : si32>) -> si32
    ```
  }];
  let arguments = (ins AnyStruct:$value, StrAttr : $field);
  let results = (outs AnyType);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

// Base class for standard arithmetic operations.  Requires operands
// and results to be of the same type, but does not constrain them to
// specific types. Individual classes will have `lhs` and `rhs`
// accessor to operands.  Stolen from standard dialect.
class ArcArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Op<Arc_Dialect, mnemonic,
       !listconcat(traits, [Pure, SameOperandsAndResultType])> {

  let results = (outs AnyType);
  let hasCustomAssemblyFormat = 1;
}

// Base class for standard arithmetic operations on integers, vectors
// thereof. This operation takes two operands and returns
// one result, each of these is required to be of the same type.  This
// type may be an integer scalar type, a vector whose element type is
// an integer type. The custom assembly form of
// the operation is as follows. Stolen from standard dialect.
//
//     <op>i %0, %1 : i32
class ArcIntArithmeticOp<string mnemonic, list<Trait> traits = []> :
    ArcArithmeticOp<mnemonic, traits>,
    Arguments<(ins ArcIntegerLike:$lhs, ArcIntegerLike:$rhs)>,
   Results<(outs ArcIntegerLike:$result)> {
   let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Arc_AddIOp : ArcIntArithmeticOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    As std.addi but supporting signed/unsigned values.
  }];

  let hasFolder = 1;
}

def Arc_AndOp : ArcIntArithmeticOp<"and", [Commutative]> {
  let summary = "integer binary and";
  let hasFolder = 1;
}

def Arc_DivIOp : ArcIntArithmeticOp<"divi"> {
  let summary = "integer division operation";
  let hasFolder = 1;
}

def Arc_OrOp : ArcIntArithmeticOp<"or", [Commutative]> {
  let summary = "integer binary or";
  let hasFolder = 1;
}

def Arc_MulIOp : ArcIntArithmeticOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  let hasFolder = 1;
}

def Arc_SubIOp : ArcIntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  let hasFolder = 1;
}

def Arc_RemIOp : ArcIntArithmeticOp<"remi"> {
  let summary = "integer division remainder operation";
  let hasFolder = 1;
}

def Arc_XOrOp : ArcIntArithmeticOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  let hasFolder = 1;
}

def PanicOp : Arc_Op<"panic", []> {
  let summary = "Trigger a run-time panic";
  let description = [{
    Trigger a run-time panic

    ```
    "arc.panic"() : () -> ()
    "arc.panic"() {msg = "foo"} : () -> ()
    arc.panic()
    arc.panic("foo")
    ```
  }];
  let arguments = (ins OptionalAttr<StrAttr>:$msg);
  let results = (outs );

  let assemblyFormat = [{
    `(` ($msg^)? `)` attr-dict
  }];
}

def ArcReturnOp : Arc_Op<"return", [Terminator]> {
  let summary = "Terminator for Arc functions.";
  let description = [{
    A terminator operation for regions that appear in the body of  `rust.func`
    functions. The operands to the `arc.return` are the result values returned
    by an incovation of the `function`.
  }];

  let arguments = (ins Optional<AnyType>:$returnedValue);

  let hasVerifier = 1;
}

def ArcSpawnOp : Arc_Op<"spawn", [
      CallOpInterface,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Spawn a new task.";

  let description = [{ Spawn a new task. The new task starts to
    execute `fun` with the given parameters.  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Arc_FilterOp : Arc_Op<"filter", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Standard filter dataflow op";
  let description = [{
    Standard filter dataflow op.
  }];

  let arguments = (ins
      ArcSourceStream:$input,
      FlatSymbolRefAttr:$predicate,
      OptionalAttr<FlatSymbolRefAttr>:$predicate_env_thunk
  );
  let results = (outs ArcSourceStream:$output);
  let hasVerifier = 1;
}

def Arc_MapOp : Arc_Op<"map", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
  let summary = "Standard map dataflow op";
  let description = [{
    Standard map dataflow op.
  }];

  let arguments = (ins
      ArcSourceStream:$input,
      FlatSymbolRefAttr:$map_fun
  );
  let results = (outs ArcSourceStream:$output);
}

#endif // ARC_OPS
