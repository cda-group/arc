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
// Arc Operations
//===----------------------------------------------------------------------===//

def MakeVector
    : Arc_Op<"make_vector", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "create a vector from a sequence of values";
  let description = [{
    A pure operation which turns its arguments into a vector
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs 1DTensorOf<[AnyType]>);
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
}

def Keep : Arc_Op<"keep", []> {
  let summary = "Operation consuming anything and having side effects.";
  let description = [{
    An operation used for debugging and testing. Can be used
    to prevent optimization passes from removing dummy
    results.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
}

def Sin : Arc_FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    The `sin` operation computes the sine of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Tan : Arc_FloatUnaryOp<"tan"> {
  let summary = "tangent of the specified value";
  let description = [{
    The `tan` operation computes the tangent of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Acos : Arc_FloatUnaryOp<"acos"> {
  let summary = "arcus cosine of the specified value";
  let description = [{
    The `sin` operation computes the arcus cosine of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Asin : Arc_FloatUnaryOp<"asin"> {
  let summary = "arcus sine of the specified value";
  let description = [{
    The `sin` operation computes the arcus sine of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Atan : Arc_FloatUnaryOp<"atan"> {
  let summary = "arcus tangent of the specified value";
  let description = [{
    The `tan` operation computes the arcus tangent of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Cosh : Arc_FloatUnaryOp<"cosh"> {
  let summary = "hyperbolic cosine of the specified value";
  let description = [{
    The `cosh` operation computes the hyperbolic cosine of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Sinh : Arc_FloatUnaryOp<"sinh"> {
  let summary = "hyperbolic sine of the specified value";
  let description = [{
    The `sinh` operation computes the hyperbolic sine of a given
    value.It takes one operand and returns one result of the same
    type.This type may be a float scalar type,
    a vector whose element type is float,
    or a tensor of floats.It has no standard attributes.
  }];
}

def Erf : Arc_FloatUnaryOp<"erf"> {
  let summary = "error function of the specified value";
  let description = [{
    The `erf' operation computes the Gauss error function of a given
    value. It takes one operand and returns one result of the same
    type. This type may be a float scalar type, a vector whose element
    type is float, or a tensor of floats.It has no standard attributes.
  }];
}

def MakeTuple : Arc_Op<"make_tuple", [NoSideEffect]> {
  let summary = "create a tuple from a sequence of values";
  let description = [{
    A pure operation which turns its arguments into a tuple
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyTuple);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
}

def IndexTuple : Arc_Op<"index_tuple", [NoSideEffect]> {
  let summary = "index into a tuple using a constant value";
  let description = [{
    A pure operation which reads a value from a tuple
  }];
  let arguments = (ins AnyTuple:$values, NonNegativeI64Attr:$index);
  let results = (outs AnyType);
  let extraClassDeclaration = [{ LogicalResult customVerify(); }];
  let verifier = [{ return customVerify(); }];
}

def BlockResult :
    Arc_Op<"block.result", [HasParent<"If">,
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

def If : Arc_Op<"if", [SingleBlockImplicitTerminator<"BlockResult">]> {
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

#endif // ARC_OPS
