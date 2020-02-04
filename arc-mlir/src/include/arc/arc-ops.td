//===- Arc dialect operation definitions ----------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
// Copyright 2019 RISE AB.
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

def Arc_Dialect : Dialect {
  let name = "arc";
  let cppNamespace = "arc";
}

class Arc_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Arc_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Arc Operations
//===----------------------------------------------------------------------===//
def MakeVector
    : Arc_Op<"make_vector", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "create a vector from a sequence of values";
  let description =
      [{A pure operation which turns its arguments into a vector}];

  let arguments = (ins Variadic<AnyType> : $values);

  let results = (outs 1DTensorOf<[AnyType]>);

  let hasCanonicalizer = 1;

  let verifier = [{
    unsigned NumOperands = this->getOperation()->getNumOperands();
    auto ElemTy = this->getOperation()->getOperand(0).getType();
    auto TensorTy =
        this->getOperation()->getResult(0).getType().cast<TensorType>();
    if (!TensorTy.hasStaticShape())
      return emitOpError("result must have static shape, expected ")
             << RankedTensorType::get({NumOperands}, ElemTy);
    if (NumOperands != TensorTy.getNumElements())
      return emitOpError("result does not match the number of operands: found ")
             << NumOperands << " but expected " << TensorTy.getNumElements()
             << " operands";
    return mlir::success();
  }];
}

def ArcKeep : Arc_Op<"keep", []> {
  let summary = "Operation consuming anything and having side effects.";
  let description = [{An operation used for debugging and testing. Can be used
                      to prevent optimization passes from removing dummy
                      results.
  }];
  let arguments = (ins Variadic<AnyType> : $values);
  let results = (outs);
}

/* The structure for these unary operands is stolen from the standard dialect */
class ArcUnaryOp<string mnemonic, list<OpTrait> traits = []>
    : Op<Arc_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
  let results = (outs AnyType);
}

class ArcUnaryOpSameOperandAndResultType<string mnemonic,
                                         list<OpTrait> traits = []>
    : ArcUnaryOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])> {
  let parser =
      [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
}

class ArcFloatUnaryOp<string mnemonic, list<OpTrait> traits = []>
    : ArcUnaryOpSameOperandAndResultType<mnemonic, traits>,
      Arguments<(ins FloatLike
                 : $operand)>;

def SinOp : ArcFloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description =
      [{The `sin` operation computes the sine of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def TanOp : ArcFloatUnaryOp<"tan"> {
  let summary = "tangent of the specified value";
  let description =
      [{The `tan` operation computes the tangent of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def AcosOp : ArcFloatUnaryOp<"acos"> {
  let summary = "arcus cosine of the specified value";
  let description =
      [{The `sin` operation computes the arcus cosine of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def AsinOp : ArcFloatUnaryOp<"asin"> {
  let summary = "arcus sine of the specified value";
  let description =
      [{The `sin` operation computes the arcus sine of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def AtanOp : ArcFloatUnaryOp<"atan"> {
  let summary = "arcus tangent of the specified value";
  let description =
      [{The `tan` operation computes the arcus tangent of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def CoshOp : ArcFloatUnaryOp<"cosh"> {
  let summary = "hyperbolic cosine of the specified value";
  let description =
      [{The `cosh` operation computes the hyperbolic cosine of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def SinhOp : ArcFloatUnaryOp<"sinh"> {
  let summary = "hyperbolic sine of the specified value";
  let description =
      [{The `sinh` operation computes the hyperbolic sine of a given
        value.It takes one operand and returns one result of the same
        type.This type may be a float scalar type,
        a vector whose element type is float,
        or a tensor of floats.It has no standard attributes.}];
}

def ErfOp : ArcFloatUnaryOp<"erf"> {
  let summary = "error function of the specified value";
  let description =
      [{The `erf' operation computes the Gauss error function of a given
        value. It takes one operand and returns one result of the same
        type. This type may be a float scalar type, a vector whose element
        type is float, or a tensor of floats.It has no standard attributes.}];
}

def ArcSqrtOp : ArcFloatUnaryOp<"sqrt"> {
  let summary = "square root of the specified value";
  let description =
      [{The `sqrt' operation computes the square root of a given
        value. It takes one operand and returns one result of the same
        type. This type may be a float scalar type, a vector whose element
        type is float, or a tensor of floats.It has no standard attributes.}];
}

def MakeTuple
    : Arc_Op<"make_tuple", [NoSideEffect]> {
  let summary = "create a tuple from a sequence of values";
  let description =
      [{A pure operation which turns its arguments into a tuple}];

  let arguments = (ins Variadic<AnyType> : $values);

  let results = (outs AnyTuple);

  let verifier = [{
    auto Operation = this->getOperation();
    unsigned NumOperands = Operation->getNumOperands();
    auto TupleTy = Operation->getResult(0).getType().cast<TupleType>();
    auto ElemTys = TupleTy.getTypes();
    if (NumOperands != TupleTy.size())
      return emitOpError("result does not match the number of operands: found ")
             << NumOperands << " but expected " << TupleTy.size()
             << " operands";
    if (NumOperands == 0)
      return emitOpError("tuple must contain at least one element ");
    unsigned I = 0;
    for (const Type &ElemTy : Operation->getOperands().getTypes()) {
        if (ElemTys[I] != ElemTy)
           return emitOpError("operand types do not match, found ")
              << ElemTy << " but expected " << ElemTys[I];
        I++;
    }
    return mlir::success();
  }];
}

def IndexTuple
    : Arc_Op<"index_tuple", [NoSideEffect]> {
  let summary = "index into a tuple using a constant value";
  let description = [{A pure operation which reads a value from a tuple}];

  let arguments = (ins AnyTuple:$values, NonNegativeI64Attr:$index);

  let results = (outs AnyType);

  let verifier = [{
    auto Operation = this->getOperation();
    auto ResultTy = Operation->getResult(0).getType();
    auto TupleTy = Operation->getOperand(0).getType().cast<TupleType>();
    auto ElemTys = TupleTy.getTypes();
    auto Index = getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    auto NumElems = TupleTy.size();
    auto IndexTy = ElemTys[Index];
    if (Index >= NumElems)
      return emitOpError("index ") << Index
        << " is out-of-bounds for tuple with size " << NumElems;
    if (IndexTy != ResultTy)
      return emitOpError("element type at index ") << Index
        << " does not match result, found " << IndexTy
        << " but expected " << ResultTy;
    return mlir::success();
  }];
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

def IfOp : Arc_Op<"if",
      [SingleBlockImplicitTerminator<"ArcBlockResultOp">]> {
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
}

#endif // ARC_OPS
