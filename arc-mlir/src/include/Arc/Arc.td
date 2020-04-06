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
include "../../mlir/include/mlir/IR/OpBase.td"
include "../../mlir/include/mlir/Interfaces/SideEffects.td"
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

class Arc_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Arc_Dialect, mnemonic, traits>;

/* The structure for these unary operands is stolen from the standard dialect */
class Arc_UnaryOp<string mnemonic, list<OpTrait> traits = []>
    : Op<Arc_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
  let results = (outs AnyType);
}

class Arc_UnaryOpSameOperandAndResultType<string mnemonic,
                                         list<OpTrait> traits = []>
    : Arc_UnaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
  let parser =
      [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
}

class Arc_FloatUnaryOp<string mnemonic, list<OpTrait> traits = []>
    : Arc_UnaryOpSameOperandAndResultType<mnemonic, traits>,
      Arguments<(ins FloatLike:$operand)>;

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
// vectors of arc integers, tensors of arc integers.
def ArcIntegerLike : TypeConstraint<Or<[
        AnyArcInteger.predicate, Index.predicate,
        VectorOf<[AnyArcInteger]>.predicate,
        TensorOf<[AnyArcInteger]>.predicate]>,
    "arc-integer-like">;

def AnyBuilder : Type<CPred<"$_self.isa<BuilderType>()">, "any builder">;
def AnyAppender : Type<CPred<"$_self.isa<AppenderType>()">, "any appender">;

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

def MakeVectorOp
    : Arc_Op<"make_vector", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "create a vector from a sequence of values";
  let description = [{
    A pure operation which turns its arguments into a vector.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs 1DTensorOf<[AnyType]>);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
}

def ArcKeepOp : Arc_Op<"keep", []> {
  let summary = "Operation consuming anything and having side effects.";
  let description = [{
    An operation used for debugging and testing. Can be used to prevent
    optimization passes from removing dummy results.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
}

def SinOp : Arc_FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    The `sin` operation computes the sine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def TanOp : Arc_FloatUnaryOp<"tan"> {
  let summary = "tangent of the specified value";
  let description = [{
    The `tan` operation computes the tangent of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def AcosOp : Arc_FloatUnaryOp<"acos"> {
  let summary = "arcus cosine of the specified value";
  let description = [{
    The `sin` operation computes the arcus cosine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def AsinOp : Arc_FloatUnaryOp<"asin"> {
  let summary = "arcus sine of the specified value";
  let description = [{
    The `sin` operation computes the arcus sine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def AtanOp : Arc_FloatUnaryOp<"atan"> {
  let summary = "arcus tangent of the specified value";
  let description = [{
    The `tan` operation computes the arcus tangent of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def CoshOp : Arc_FloatUnaryOp<"cosh"> {
  let summary = "hyperbolic cosine of the specified value";
  let description = [{
    The `cosh` operation computes the hyperbolic cosine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def SinhOp : Arc_FloatUnaryOp<"sinh"> {
  let summary = "hyperbolic sine of the specified value";
  let description = [{
    The `sinh` operation computes the hyperbolic sine of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def ErfOp : Arc_FloatUnaryOp<"erf"> {
  let summary = "error function of the specified value";
  let description = [{
    The `erf' operation computes the Gauss error function of a given value.
    It takes one operand and returns one result of the same type. This type may
    be a float scalar type, a vector whose element type is float, or a tensor of
    floats. It has no standard attributes.
  }];
}

def MakeTupleOp : Arc_Op<"make_tuple", [NoSideEffect]> {
  let summary = "create a tuple from a sequence of values";
  let description = [{
    A pure operation which turns its arguments into a tuple.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyTuple);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
}

def IndexTupleOp : Arc_Op<"index_tuple", [NoSideEffect]> {
  let summary = "index into a tuple using a constant value";
  let description = [{
    A pure operation which reads a value from a tuple.
  }];
  let arguments = (ins AnyTuple:$values, Confined<I64Attr, [IntNonNegative]>:$index);
  let results = (outs AnyType);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
}

def ArcBlockResultOp :
    Arc_Op<"block.result", [HasParent<"IfOp">,
                            Terminator, SameOperandsAndResultType]> {
  let summary = "specifies the value of a block";
  let description = [{
    "arc.block.result" is a special terminator operation for the block inside
    "arc.if". It terminates the region. It should have the same type as the
    parent if.

    ```mlir
      "arc.block.result"(%b) : (f32) -> f64
    ```
  }];
  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);
}

def IfOp : Arc_Op<"if", [SingleBlockImplicitTerminator<"ArcBlockResultOp">]> {
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
  let results = (outs AnyType);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
  let hasCanonicalizer = 1;
}

def MakeAppenderOp : Arc_Op<"make_appender", [NoSideEffect, Linear]> {
  let summary = "instantiate a new appender";
  let description = [{
    The "arc.make_appender" operation instantiates and returns a new appender.
    The appender's initial storage `size` can optionally be given as an
    attribute. Note that the result of this operation must be used exactly once.

    ```mlir
    %a = arc.make_appender() {size = 10} : () -> !arc.appender<i32>
    ```
  }];
  let arguments = (Confined<I64Attr, [IntNonNegative]>:$size);
  let results = (outs AnyAppender);
}

def MergeOp : Arc_Op<"merge", [NoSideEffect, Linear]> {
  let summary = "merge a value into a builder";
  let description = [{
    The "arc.merge" operation takes a builder and a value, and returns a new
    builder with the value stored into the old builder. The old builder is
    consumed in the process. Note that the result of this operation must be
    used exactly once.

    ```mlir
    %a = "arc.make_appender"() : () -> !arc.appender<i32>
    %v = constant 5 : i32
    %b = "arc.merge"(%a, %v) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    ```
  }];
  let arguments = (ins AnyBuilder:$builder, AnyValue:$value);
  let results = (outs AnyBuilder);
  let verifier = [{ return customVerify(); }];
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
}

def ResultOp : Arc_Op<"result", [NoSideEffect]> {
  let summary = "materialize a builder into a value";
  let description = [{
    The "arc.result" operation takes a builder and materializes it into a value
    of its result type. The resulting value is returned, and the builder is
    consumed in the process.

    ```mlir
    %a = "arc.make_appender"() : () -> !arc.appender<i32>
    %v = constant 5 : i32
    %b = "arc.merge"(%a, %v) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
    %r = "arc.result"(%b) : (!arc.appender<i32>) -> i32
    ```
  }];
  let arguments = (ins AnyAppender:$builder);
  let results = (outs AnyType);
  let verifier = [{ return customVerify(); }];
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
}

def Arc_ConstantIntOp : Arc_Op<"constant", [ConstantLike, NoSideEffect]> {
  let summary = "The op that declares an Arc integer constant";

  let description = [{
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
    AnyArcInteger:$constant
  );

  let parser = [{ return ::parseConstantIntOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
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
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape,
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
    static StringRef getPredicateAttrName() { return "predicate"; }
    static Arc_CmpIPredicate getPredicateByName(StringRef name);

    Arc_CmpIPredicate getPredicate() {
      return (Arc_CmpIPredicate)
          getAttrOfType<IntegerAttr>(getPredicateAttrName())
              .getInt();
    }
  }];

  let verifier = [{ return success(); }];

  // TODO implement folder
  //let hasFolder = 1;

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Arc_SelectOp : Arc_Op<"select", [NoSideEffect, SameOperandsAndResultShape,
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
  let verifier = ?;

  let extraClassDeclaration = [{
      Value getCondition() { return condition(); }
      Value getTrueValue() { return true_value(); }
      Value getFalseValue() { return false_value(); }
  }];

  // TODO add fiolder
  //  let hasFolder = 1;

  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value attr-dict `:` type($result)
  }];
}

// Base class for standard arithmetic operations.  Requires operands
// and results to be of the same type, but does not constrain them to
// specific types. Individual classes will have `lhs` and `rhs`
// accessor to operands.  Stolen from standard dialect.
class ArcArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Arc_Dialect, mnemonic,
       !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {

  let results = (outs AnyType);

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printArcBinaryOp(this->getOperation(), p);
  }];
}

// Base class for standard arithmetic operations on integers, vectors
// and tensors thereof. This operation takes two operands and returns
// one result, each of these is required to be of the same type.  This
// type may be an integer scalar type, a vector whose element type is
// an integer type, or an integer tensor. The custom assembly form of
// the operation is as follows. Stolen from standard dialect.
//
//     <op>i %0, %1 : i32
class ArcIntArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    ArcArithmeticOp<mnemonic, traits>,
    Arguments<(ins ArcIntegerLike:$lhs, ArcIntegerLike:$rhs)>;

def Arc_AddIOp : ArcIntArithmeticOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    As std.addi but supporting signed/unsigned values.
  }];

  // TODO: create folder
  //let hasFolder = 1;
}

def ArcAndOp : ArcIntArithmeticOp<"and", [Commutative]> {
  let summary = "integer binary and";
  // TODO let hasFolder = 1;
}

def Arc_DivIOp : ArcIntArithmeticOp<"divi"> {
  let summary = "integer division operation";
  // TODO  let hasFolder = 1;
}

def Arc_OrOp : ArcIntArithmeticOp<"or", [Commutative]> {
  let summary = "integer binary or";
  // TODO  let hasFolder = 1;
}

def Arc_MulIOp : ArcIntArithmeticOp<"muli", [Commutative]> {
  let summary = "integer multiplication operation";
  // TODO let hasFolder = 1;
}

def Arc_SubIOp : ArcIntArithmeticOp<"subi"> {
  let summary = "integer subtraction operation";
  // TODO let hasFolder = 1;
}

def Arc_SignedRemIOp : ArcIntArithmeticOp<"remi"> {
  let summary = "integer division remainder operation";
  // TODO  let hasFolder = 1;
}

def Arc_XOrOp : ArcIntArithmeticOp<"xor", [Commutative]> {
  let summary = "integer binary xor";
  // TODO let hasFolder = 1;
}

#endif // ARC_OPS
