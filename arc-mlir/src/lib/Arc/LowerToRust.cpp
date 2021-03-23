//====- LowerToRust.cpp - Lowering from Arc+MLIR to Rust --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a lowering of Arc and MLIR operations to
// Rust. Boilerplate stolen from the MLIR Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "Arc/Arc.h"
#include "Arc/Passes.h"
#include "Rust/Rust.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace arc;

namespace arc {
#define GEN_PASS_CLASSES
#include "Arc/Passes.h.inc"
} // namespace arc

//===----------------------------------------------------------------------===//
// ArcToRustLoweringPass
//===----------------------------------------------------------------------===//

/// This is a lowering of arc operations to the Rust dialect.
namespace {
struct ArcToRustLoweringPass : public LowerToRustBase<ArcToRustLoweringPass> {
  void runOnOperation() final;

  static void emitCrateDependency(StringRef crate, StringRef version,
                                  MLIRContext *ctx,
                                  ConversionPatternRewriter &rewriter);

  static void emitModuleDirective(StringRef key, StringRef str,
                                  MLIRContext *ctx,
                                  ConversionPatternRewriter &rewriter);

  static const std::string hexfCrate;
};
} // end anonymous namespace.

class RustTypeConverter : public TypeConverter {
  MLIRContext *Ctx;
  rust::RustDialect *Dialect;

public:
  RustTypeConverter(MLIRContext *ctx);

  FunctionType convertFunctionSignature(Type, SignatureConversion &);

protected:
  Type convertFloatType(FloatType type);
  Type convertFunctionType(FunctionType type);
  Type convertIntegerType(IntegerType type);
  Type convertTensorType(RankedTensorType type);
  Type convertTupleType(TupleType type);
  Type convertStructType(arc::types::StructType type);
};

struct ReturnOpLowering : public ConversionPattern {

  ReturnOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ReturnOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustReturnOp>(
        op, llvm::None, operands.size() ? operands[0] : Value());
    return success();
  };
};

struct StdCallOpLowering : public ConversionPattern {
  StdCallOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(CallOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    CallOp o = cast<CallOp>(op);
    SmallVector<Type, 4> resultTypes;
    for (auto r : o.getResultTypes())
      resultTypes.push_back(TypeConverter.convertType(r));
    rewriter.replaceOpWithNewOp<rust::RustCallOp>(op, o.getCallee(),
                                                  resultTypes, operands);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct StdConstantOpLowering : public ConversionPattern {

  StdConstantOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(mlir::ConstantOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter), Ctx(ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::ConstantOp cOp = cast<mlir::ConstantOp>(op);
    Attribute attr = cOp.getValue();
    if (attr.isa<FloatAttr>())
      return convertFloat(cOp, rewriter);
    if (attr.isa<IntegerAttr>())
      return convertInteger(cOp, rewriter);
    op->emitError("unhandled constant type");
    return failure();
  };

private:
  RustTypeConverter &TypeConverter;
  MLIRContext *Ctx;

  LogicalResult returnResult(Operation *op, Type ty, StringRef value,
                             ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<rust::RustConstantOp>(op, ty, value);
    return success();
  }

  LogicalResult convertFloat(mlir::ConstantOp op,
                             ConversionPatternRewriter &rewriter) const {
    FloatAttr attr = op.getValue().cast<FloatAttr>();
    Type ty = attr.getType();
    Type rustTy = TypeConverter.convertType(ty);
    APFloat f = attr.getValue();
    unsigned width = ty.getIntOrFloatBitWidth();

    Type f16 = rust::types::RustType::getFloat16Ty(
        Ctx->getOrLoadDialect<rust::RustDialect>());
    Type bf16 = rust::types::RustType::getBFloat16Ty(
        Ctx->getOrLoadDialect<rust::RustDialect>());

    std::string rustTyName = "f" + Twine(width).str();

    if (f.isInfinity()) {
      if (f.isNegative())
        rustTyName += "::NEG_INFINITY";
      else
        rustTyName += "::INFINITY";
      return returnResult(op, rustTy, rustTyName, rewriter);
    }
    if (f.isNaN())
      return returnResult(op, rustTy, rustTyName + "::NAN", rewriter);

    char hex[256];
    f.convertToHexString(hex, 0, false, llvm::APFloat::rmNearestTiesToEven);

    // APFloat::convertToHexString() omits the '.' if the mantissa is
    // 0 which hexf doesn't like. To keep hexf happy we patch the
    // string.
    for (size_t i = 0; hex[i] && hex[i] != '.'; i++) {
      if (hex[i] == 'p') {
        memmove(hex + i + 1, hex + i, strlen(hex + i));
        hex[i] = '.';
        break;
      }
    }

    Twine str = "hexf" + Twine(width) + "!(\"" + hex + "\")";
    std::string cst = str.str();

    // The 16 bit floats are printed as f32 hex and converted using
    // arcorn.
    if (rustTy == f16 || rustTy == bf16) {
      cst = "arcorn::";
      cst += (rustTy == bf16) ? "b" : "";
      cst += "f16::from_f32(hexf32!(\"";
      cst += hex;
      cst += "\"))";
    }

    std::string directive =
        "#[macro_use] extern crate " + ArcToRustLoweringPass::hexfCrate + ";";
    ArcToRustLoweringPass::emitCrateDependency(ArcToRustLoweringPass::hexfCrate,
                                               rust::CrateVersions::hexf,
                                               op.getContext(), rewriter);
    ArcToRustLoweringPass::emitModuleDirective(
        ArcToRustLoweringPass::hexfCrate, directive, op.getContext(), rewriter);
    return returnResult(op, rustTy, cst, rewriter);
  }

  LogicalResult convertInteger(mlir::ConstantOp op,
                               ConversionPatternRewriter &rewriter) const {
    IntegerAttr attr = op.getValue().cast<IntegerAttr>();
    IntegerType ty = attr.getType().cast<IntegerType>();
    Type rustTy = TypeConverter.convertType(ty);
    APInt v = attr.getValue();

    unsigned width = ty.getIntOrFloatBitWidth();
    switch (width) {
    case 1:
      return returnResult(op, rustTy, v.isNullValue() ? "false" : "true",
                          rewriter);
    default:
      op.emitError("unhandled constant integer width");
      return failure();
    }
  }
};

struct ConstantIntOpLowering : public ConversionPattern {

  ConstantIntOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::ConstantIntOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    arc::ConstantIntOp cOp = cast<arc::ConstantIntOp>(op);
    Attribute attr = cOp.getValue();
    if (attr.isa<IntegerAttr>())
      return convertInteger(cOp, rewriter);
    op->emitError("unhandled constant type");
    return failure();
  };

private:
  RustTypeConverter &TypeConverter;

  LogicalResult returnResult(Operation *op, Type ty, StringRef value,
                             ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<rust::RustConstantOp>(op, ty, value);
    return success();
  }

  LogicalResult convertInteger(arc::ConstantIntOp op,
                               ConversionPatternRewriter &rewriter) const {
    IntegerAttr attr = op.getValue().cast<IntegerAttr>();
    IntegerType ty = attr.getType().cast<IntegerType>();
    Type rustTy = TypeConverter.convertType(ty);
    APInt v = attr.getValue();

    unsigned width = ty.getIntOrFloatBitWidth();
    switch (width) {
    case 8:
    case 16:
    case 32:
    case 64:
      if (ty.isSignless()) {
        op.emitError("signless integers are not supported");
        return failure();
      }
      return returnResult(op, rustTy, v.toString(10, ty.isSigned()), rewriter);
    default:
      op.emitError("unhandled constant integer width");
      return failure();
    }
  }
};

struct BlockResultOpLowering : public ConversionPattern {
  BlockResultOpLowering(MLIRContext *ctx)
      : ConversionPattern(arc::ArcBlockResultOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustBlockResultOp>(op, operands[0]);
    return success();
  };
};

struct IfOpLowering : public ConversionPattern {
  IfOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::IfOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    arc::IfOp o = cast<arc::IfOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    auto newOp =
        rewriter.create<rust::RustIfOp>(op->getLoc(), retTy, operands[0]);

    rewriter.inlineRegionBefore(o.thenRegion(), newOp.thenRegion(),
                                newOp.thenRegion().end());
    rewriter.inlineRegionBefore(o.elseRegion(), newOp.elseRegion(),
                                newOp.elseRegion().end());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct IndexTupleOpLowering : public ConversionPattern {
  IndexTupleOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::IndexTupleOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    IndexTupleOp o = cast<IndexTupleOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustFieldAccessOp>(
        op, retTy, operands[0], std::to_string(o.index()));
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct MakeStructOpLowering : public ConversionPattern {
  MakeStructOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::MakeStructOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    MakeStructOp o = cast<MakeStructOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustMakeStructOp>(op, retTy, operands);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct MakeTensorOpLowering : public ConversionPattern {
  MakeTensorOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::MakeTensorOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    MakeTensorOp o = cast<MakeTensorOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustTensorOp>(op, retTy, operands);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct MakeTupleOpLowering : public ConversionPattern {
  MakeTupleOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::MakeTupleOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    MakeTupleOp o = cast<MakeTupleOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustTupleOp>(op, retTy, operands);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

namespace ArcIntArithmeticOp {
typedef enum {
  AddIOp = 0,
  AndOp,
  DivIOp,
  OrOp,
  MulIOp,
  SubIOp,
  RemIOp,
  XOrOp,
  LAST
} Op;
}

template <class T, ArcIntArithmeticOp::Op arithOp>
struct ArcIntArithmeticOpLowering : public ConversionPattern {
  ArcIntArithmeticOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(T::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    T o = cast<T>(op);
    Type rustTy = TypeConverter.convertType(o.getType());
    if (rustTy.isa<rust::types::RustTensorType>())
      rewriter.replaceOpWithNewOp<rust::RustBinaryRcOp>(
          op, rustTy, opStr[arithOp], operands[0], operands[1]);
    else
      rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
          op, rustTy, opStr[arithOp], operands[0], operands[1]);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[ArcIntArithmeticOp::LAST] = {"+", "&", "/", "|",
                                                 "*", "-", "%", "^"};
};

struct ArcCmpIOpLowering : public ConversionPattern {
  ArcCmpIOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::CmpIOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    arc::CmpIOp o = cast<arc::CmpIOp>(op);
    Type rustTy = TypeConverter.convertType(o.getType());
    const char *cmpOp = NULL;
    switch (o.getPredicate()) {
    case Arc_CmpIPredicate::eq:
      cmpOp = "==";
      break;
    case Arc_CmpIPredicate::ne:
      cmpOp = "!=";
      break;
    case Arc_CmpIPredicate::lt:
      cmpOp = "<";
      break;
    case Arc_CmpIPredicate::le:
      cmpOp = "<=";
      break;
    case Arc_CmpIPredicate::gt:
      cmpOp = ">";
      break;
    case Arc_CmpIPredicate::ge:
      cmpOp = ">=";
      break;
    }
    rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(op, rustTy, cmpOp,
                                                    operands[0], operands[1]);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

namespace StdArithmeticOp {
typedef enum { AddFOp = 0, DivFOp, MulFOp, SubFOp, RemFOp, LAST } Op;
}

template <class T, StdArithmeticOp::Op arithOp>
struct StdArithmeticOpLowering : public ConversionPattern {
  StdArithmeticOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(T::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    T o = cast<T>(op);
    Type rustTy = TypeConverter.convertType(o.getType());
    if (rustTy.isa<rust::types::RustTensorType>())
      rewriter.replaceOpWithNewOp<rust::RustBinaryRcOp>(
          op, rustTy, opStr[arithOp], operands[0], operands[1]);
    else
      rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
          op, rustTy, opStr[arithOp], operands[0], operands[1]);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[StdArithmeticOp::LAST] = {"+", "/", "*", "-", "%"};
};

namespace OpAsMethod {
typedef enum { PowFOp = 0, LAST } Op;
}

template <class T, OpAsMethod::Op theOp>
struct OpAsMethodLowering : public ConversionPattern {
  OpAsMethodLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(T::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    T o = cast<T>(op);
    Type rustTy = TypeConverter.convertType(o.getType());
    SmallVector<mlir::Value, 1> args;

    for (unsigned i = 1; i < operands.size(); i++)
      args.push_back(operands[i]);

    rewriter.replaceOpWithNewOp<rust::RustMethodCallOp>(
        op, rustTy, opStr[theOp], operands[0], args);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[StdArithmeticOp::LAST] = {"powf"};
};

struct StdCmpFOpLowering : public ConversionPattern {
  StdCmpFOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(mlir::CmpFOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::CmpFOp o = cast<mlir::CmpFOp>(op);
    Type rustTy = TypeConverter.convertType(o.getType());
    const char *cmpOp = NULL;
    switch (o.getPredicate()) {
    case CmpFPredicate::OEQ:
      cmpOp = "==";
      break;
    case CmpFPredicate::ONE:
      cmpOp = "!=";
      break;
    case CmpFPredicate::OLT:
      cmpOp = "<";
      break;
    case CmpFPredicate::OLE:
      cmpOp = "<=";
      break;
    case CmpFPredicate::OGT:
      cmpOp = ">";
      break;
    case CmpFPredicate::OGE:
      cmpOp = ">=";
      break;
    default:
      op->emitError("unhandled std.cmpf operation");
      return failure();
    }
    rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(op, rustTy, cmpOp,
                                                    operands[0], operands[1]);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

namespace ArcUnaryFloatOp {
typedef enum {
  sin = 0,
  cos,
  tan,
  asin,
  acos,
  atan,
  cosh,
  sinh,
  tanh,
  log,
  exp,
  sqrt,
  LAST
} Op;
}

template <class T, ArcUnaryFloatOp::Op arithOp>
struct ArcUnaryFloatOpLowering : public ConversionPattern {
  ArcUnaryFloatOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(T::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    T o = cast<T>(op);
    Type rustTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustMethodCallOp>(
        op, rustTy, opStr[arithOp], operands[0],
        SmallVector<mlir::Value, 1>({}));
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[ArcUnaryFloatOp::LAST] = {"sin",  "cos",  "tan",  "asin",
                                              "acos", "atan", "cosh", "sinh",
                                              "tanh", "ln",   "exp",  "sqrt"};
};

struct StructAccessOpLowering : public ConversionPattern {
  StructAccessOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::StructAccessOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    StructAccessOp o = cast<StructAccessOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    std::string idx_str(o.field());
    rewriter.replaceOpWithNewOp<rust::RustFieldAccessOp>(op, retTy, operands[0],
                                                         idx_str);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

RustTypeConverter::RustTypeConverter(MLIRContext *ctx)
    : Ctx(ctx), Dialect(ctx->getOrLoadDialect<rust::RustDialect>()) {
  addConversion([&](FloatType type) { return convertFloatType(type); });
  addConversion([&](FunctionType type) { return convertFunctionType(type); });
  addConversion([&](IntegerType type) { return convertIntegerType(type); });
  addConversion([&](RankedTensorType type) { return convertTensorType(type); });
  addConversion([&](TupleType type) { return convertTupleType(type); });
  addConversion(
      [&](arc::types::StructType type) { return convertStructType(type); });

  // RustType is legal, so add a pass-through conversion.
  addConversion([](rust::types::RustType type) { return type; });
  addConversion([](rust::types::RustStructType type) { return type; });
  addConversion([](rust::types::RustTensorType type) { return type; });
  addConversion([](rust::types::RustTupleType type) { return type; });
}

Type RustTypeConverter::convertFloatType(FloatType type) {
  if (type.isa<Float32Type>())
    return rust::types::RustType::getFloatTy(Dialect);
  if (type.isa<Float64Type>())
    return rust::types::RustType::getDoubleTy(Dialect);
  if (type.isa<Float16Type>())
    return rust::types::RustType::getFloat16Ty(Dialect);
  if (type.isa<BFloat16Type>())
    return rust::types::RustType::getBFloat16Ty(Dialect);
  if (type.isa<IntegerType>())
    return rust::types::RustType::getIntegerTy(Dialect,
                                               type.cast<IntegerType>());
  return emitError(UnknownLoc::get(Ctx), "unsupported type"), Type();
}

Type RustTypeConverter::convertFunctionType(FunctionType type) {
  SmallVector<mlir::Type, 4> inputs;
  SmallVector<mlir::Type, 1> results;

  if (failed(convertTypes(type.getInputs(), inputs)))
    emitError(UnknownLoc::get(Ctx), "failed to convert function input types");
  if (failed(convertTypes(type.getResults(), results)))
    emitError(UnknownLoc::get(Ctx), "failed to convert function result types");

  return FunctionType::get(Ctx, inputs, results);
}

Type RustTypeConverter::convertIntegerType(IntegerType type) {
  if (auto t = type.dyn_cast<IntegerType>())
    return rust::types::RustType::getIntegerTy(Dialect, t);
  return emitError(UnknownLoc::get(Ctx), "unsupported type"), Type();
}

Type RustTypeConverter::convertStructType(arc::types::StructType type) {
  SmallVector<rust::types::RustType::StructFieldTy, 4> fields;
  for (const auto &f : type.getFields()) {
    Type t = convertType(f.second);
    fields.push_back(std::make_pair(f.first, t));
  }
  return rust::types::RustStructType::get(Dialect, fields);
}

Type RustTypeConverter::convertTensorType(RankedTensorType type) {
  Type t = convertType(type.getElementType());
  return rust::types::RustTensorType::get(Dialect, t, type.getShape());
}

Type RustTypeConverter::convertTupleType(TupleType type) {
  SmallVector<Type, 4> elements;
  for (Type t : type)
    elements.push_back(convertType(t));
  return rust::types::RustTupleType::get(Dialect, elements);
}

FunctionType
RustTypeConverter::convertFunctionSignature(Type ty, SignatureConversion &SC) {
  mlir::FunctionType funcType =
      convertType(ty.cast<mlir::FunctionType>()).cast<mlir::FunctionType>();

  for (auto &en : llvm::enumerate(funcType.getInputs())) {
    Type type = en.value();
    SC.addInputs(en.index(), convertType(type));
  }

  return funcType;
}

struct FuncOpLowering : public ConversionPattern {

  FuncOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(mlir::FuncOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = op->getContext();
    mlir::FuncOp func = cast<mlir::FuncOp>(op);
    SmallVector<NamedAttribute, 4> attributes;

    TypeConverter::SignatureConversion sigConv(func.getNumArguments());
    mlir::FunctionType funcType =
        TypeConverter.convertFunctionSignature(func.getType(), sigConv);

    attributes.push_back(
        NamedAttribute(Identifier::get("type", ctx), TypeAttr::get(funcType)));
    attributes.push_back(NamedAttribute(Identifier::get("sym_name", ctx),
                                        StringAttr::get(ctx, func.getName())));

    if (func.isExternal())
      return buildExternalFun(rewriter, operands, attributes, func, op, sigConv,
                              ctx);
    return buildLocalFun(rewriter, operands, attributes, func, op, sigConv,
                         ctx);
  };

private:
  RustTypeConverter &TypeConverter;

  LogicalResult buildLocalFun(ConversionPatternRewriter &rewriter,
                              ArrayRef<Value> &operands,
                              SmallVector<NamedAttribute, 4> &attributes,
                              mlir::FuncOp &func, Operation *op,
                              TypeConverter::SignatureConversion &sigConv,
                              MLIRContext *ctx) const {

    auto newOp = rewriter.create<rust::RustFuncOp>(
        op->getLoc(), SmallVector<mlir::Type, 1>({}), operands, attributes);

    rewriter.inlineRegionBefore(func.getBody(), newOp.getBody(), newOp.end());
    rewriter.applySignatureConversion(&newOp.getBody(), sigConv);

    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), TypeConverter,
                                           &sigConv)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult buildExternalFun(ConversionPatternRewriter &rewriter,
                                 ArrayRef<Value> &operands,
                                 SmallVector<NamedAttribute, 4> &attributes,
                                 mlir::FuncOp &func, Operation *op,
                                 TypeConverter::SignatureConversion &sigConv,
                                 MLIRContext *ctx) const {
    StringAttr lang = func->getAttr("arc.foreign_language").cast<StringAttr>();
    auto prefix = (Twine("arc.foreign.") + lang.getValue() + ".").str();
    attributes.push_back(
        NamedAttribute(Identifier::get("language", ctx), lang));

    for (auto a : func->getDialectAttrs()) {
      auto name = a.first.strref();
      if (name.consume_front(prefix)) {
        attributes.push_back(
            NamedAttribute(Identifier::get(name, ctx), a.second));
      }
    }

    auto newOp = rewriter.create<rust::RustExtFuncOp>(
        op->getLoc(), SmallVector<mlir::Type, 1>({}), operands, attributes);
    rewriter.applySignatureConversion(&newOp.getBody(), sigConv);

    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), TypeConverter,
                                           &sigConv)))
      return failure();

    /* Move all module-level arc-dialect dependency attributes over to
       their target dialect names */
    prefix = prefix + "dependency.";
    auto parent = newOp->getParentOp();

    for (auto a : parent->getDialectAttrs()) {
      auto name = a.first.strref();
      if (name.consume_front(prefix)) {
        auto new_name = (lang.getValue() + ".dependency." + name).str();
        parent->removeAttr(a.first.strref());
        parent->setAttr(Identifier::get(new_name, ctx), a.second);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void ArcToRustLoweringPass::runOnOperation() {
  RustTypeConverter typeConverter(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<rust::RustDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  OwningRewritePatternList patterns(&getContext());
  patterns.insert<ReturnOpLowering>(&getContext());
  patterns.insert<FuncOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdConstantOpLowering>(&getContext(), typeConverter);
  patterns.insert<ConstantIntOpLowering>(&getContext(), typeConverter);

  patterns.insert<
      ArcIntArithmeticOpLowering<arc::AddIOp, ArcIntArithmeticOp::AddIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      ArcIntArithmeticOpLowering<arc::DivIOp, ArcIntArithmeticOp::DivIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      ArcIntArithmeticOpLowering<arc::MulIOp, ArcIntArithmeticOp::MulIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      ArcIntArithmeticOpLowering<arc::SubIOp, ArcIntArithmeticOp::SubIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      ArcIntArithmeticOpLowering<arc::RemIOp, ArcIntArithmeticOp::RemIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      ArcIntArithmeticOpLowering<arc::AndOp, ArcIntArithmeticOp::AndOp>>(
      &getContext(), typeConverter);
  patterns
      .insert<ArcIntArithmeticOpLowering<arc::OrOp, ArcIntArithmeticOp::OrOp>>(
          &getContext(), typeConverter);
  patterns.insert<
      ArcIntArithmeticOpLowering<arc::XOrOp, ArcIntArithmeticOp::XOrOp>>(
      &getContext(), typeConverter);

  patterns
      .insert<StdArithmeticOpLowering<mlir::AddFOp, StdArithmeticOp::AddFOp>>(
          &getContext(), typeConverter);
  patterns
      .insert<StdArithmeticOpLowering<mlir::DivFOp, StdArithmeticOp::DivFOp>>(
          &getContext(), typeConverter);
  patterns
      .insert<StdArithmeticOpLowering<mlir::MulFOp, StdArithmeticOp::MulFOp>>(
          &getContext(), typeConverter);
  patterns
      .insert<StdArithmeticOpLowering<mlir::SubFOp, StdArithmeticOp::SubFOp>>(
          &getContext(), typeConverter);
  patterns
      .insert<StdArithmeticOpLowering<mlir::RemFOp, StdArithmeticOp::RemFOp>>(
          &getContext(), typeConverter);
  patterns.insert<OpAsMethodLowering<math::PowFOp, OpAsMethod::PowFOp>>(
      &getContext(), typeConverter);

  patterns.insert<ArcCmpIOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCmpFOpLowering>(&getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<math::SinOp, ArcUnaryFloatOp::sin>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<math::CosOp, ArcUnaryFloatOp::cos>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::TanOp, ArcUnaryFloatOp::tan>>(
      &getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<arc::AsinOp, ArcUnaryFloatOp::asin>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::AcosOp, ArcUnaryFloatOp::acos>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<math::AtanOp, ArcUnaryFloatOp::atan>>(
      &getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<arc::SinhOp, ArcUnaryFloatOp::sinh>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::CoshOp, ArcUnaryFloatOp::cosh>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<math::TanhOp, ArcUnaryFloatOp::tanh>>(
      &getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<math::LogOp, ArcUnaryFloatOp::log>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<math::ExpOp, ArcUnaryFloatOp::exp>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<math::SqrtOp, ArcUnaryFloatOp::sqrt>>(
      &getContext(), typeConverter);

  patterns.insert<IfOpLowering>(&getContext(), typeConverter);
  patterns.insert<IndexTupleOpLowering>(&getContext(), typeConverter);
  patterns.insert<BlockResultOpLowering>(&getContext());
  patterns.insert<MakeTupleOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeTensorOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeStructOpLowering>(&getContext(), typeConverter);
  patterns.insert<StructAccessOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCallOpLowering>(&getContext(), typeConverter);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createLowerToRustPass() {
  return std::make_unique<ArcToRustLoweringPass>();
}

const std::string ArcToRustLoweringPass::hexfCrate = "hexf";

void ArcToRustLoweringPass::emitCrateDependency(
    StringRef crate, StringRef version, MLIRContext *ctx,
    ConversionPatternRewriter &rewriter) {
  rewriter.create<rust::RustDependencyOp>(UnknownLoc::get(ctx), crate, version);
}

void ArcToRustLoweringPass::emitModuleDirective(
    StringRef key, StringRef str, MLIRContext *ctx,
    ConversionPatternRewriter &rewriter) {
  rewriter.create<rust::RustModuleDirectiveOp>(UnknownLoc::get(ctx), key, str);
}
