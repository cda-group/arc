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
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Arc Dialect
//===----------------------------------------------------------------------===//

def Arc_Dialect : Dialect {
  let name = "arc";
  let cppNamespace = "arc";
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

def AnyValue : Type<Or<[
  AnySignlessInteger.predicate,
  AnyFloat.predicate,
]>, "any value type">;

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
  let arguments = (ins AnyTuple:$values, NonNegativeI64Attr:$index);
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
  let arguments = (NonNegativeI64Attr:$size);
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

#endif // ARC_OPS
