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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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
  static const std::string hexfVersion;
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

struct StdConstantOpLowering : public ConversionPattern {

  StdConstantOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(mlir::ConstantOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    mlir::ConstantOp cOp = cast<mlir::ConstantOp>(op);
    Attribute attr = cOp.getValue();
    switch (attr.getKind()) {
    case StandardAttributes::Float:
      return convertFloat(cOp, rewriter);
    case StandardAttributes::Integer:
      return convertInteger(cOp, rewriter);
    default:
      op->emitError("unhandled constant type");
      return failure();
    }
  };

private:
  RustTypeConverter &TypeConverter;

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
    char hex[256];
    f.convertToHexString(hex, 0, false, llvm::APFloat::rmNearestTiesToEven);

    unsigned width = ty.getIntOrFloatBitWidth();
    Twine str = "hexf" + Twine(width) + "!(\"" + hex + "\")";
    std::string directive =
        "#[macro_use] extern crate " + ArcToRustLoweringPass::hexfCrate + ";";
    ArcToRustLoweringPass::emitCrateDependency(
        ArcToRustLoweringPass::hexfCrate, ArcToRustLoweringPass::hexfVersion,
        op.getContext(), rewriter);
    ArcToRustLoweringPass::emitModuleDirective(
        ArcToRustLoweringPass::hexfCrate, directive, op.getContext(), rewriter);
    return returnResult(op, rustTy, str.str(), rewriter);
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
    switch (attr.getKind()) {
    case StandardAttributes::Integer:
      return convertInteger(cOp, rewriter);
    default:
      op->emitError("unhandled constant type");
      return failure();
    }
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
  BlockResultOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : ConversionPattern(arc::ArcBlockResultOp::getOperationName(), 1, ctx),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    ArcBlockResultOp o = cast<ArcBlockResultOp>(op);
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustBlockResultOp>(op, retTy,
                                                         operands[0]);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
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
    APInt i = o.index();
    std::string idx_str = i.toString(10, false);
    rewriter.replaceOpWithNewOp<rust::RustFieldAccessOp>(op, retTy, operands[0],
                                                         idx_str);
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
};

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
    rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(op, rustTy, opStr[arithOp],
                                                    operands[0], operands[1]);
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
};

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

RustTypeConverter::RustTypeConverter(MLIRContext *ctx)
    : Ctx(ctx), Dialect(ctx->getRegisteredDialect<rust::RustDialect>()) {
  addConversion([&](FloatType type) { return convertFloatType(type); });
  addConversion([&](FunctionType type) { return convertFunctionType(type); });
  addConversion([&](IntegerType type) { return convertIntegerType(type); });
  addConversion([&](TupleType type) { return convertTupleType(type); });
  addConversion(
      [&](arc::types::StructType type) { return convertStructType(type); });

  // RustType is legal, so add a pass-through conversion.
  addConversion([](rust::types::RustType type) { return type; });
  addConversion([](rust::types::RustStructType type) { return type; });
}

Type RustTypeConverter::convertFloatType(FloatType type) {
  switch (type.getKind()) {
  case mlir::StandardTypes::F32:
    return rust::types::RustType::getFloatTy(Dialect);
  case mlir::StandardTypes::F64:
    return rust::types::RustType::getDoubleTy(Dialect);
  case mlir::StandardTypes::Integer:
    return rust::types::RustType::getIntegerTy(Dialect,
                                               type.cast<IntegerType>());
  default:
    return emitError(UnknownLoc::get(Ctx), "unsupported type"), Type();
  }
}

Type RustTypeConverter::convertFunctionType(FunctionType type) {
  SmallVector<mlir::Type, 4> inputs;
  SmallVector<mlir::Type, 1> results;

  if (failed(convertTypes(type.getInputs(), inputs)))
    emitError(UnknownLoc::get(Ctx), "failed to convert function input types");
  if (failed(convertTypes(type.getResults(), results)))
    emitError(UnknownLoc::get(Ctx), "failed to convert function result types");

  return FunctionType::get(inputs, results, Ctx);
}

Type RustTypeConverter::convertIntegerType(IntegerType type) {
  switch (type.getKind()) {
  case mlir::StandardTypes::Integer:
    return rust::types::RustType::getIntegerTy(Dialect,
                                               type.cast<IntegerType>());
  default:
    return emitError(UnknownLoc::get(Ctx), "unsupported type"), Type();
  }
}

Type RustTypeConverter::convertStructType(arc::types::StructType type) {
  SmallVector<rust::types::RustType::StructFieldTy, 4> fields;
  for (const auto &f : type.getFields()) {
    Type t = convertType(f.second);
    fields.push_back(std::make_pair(f.first, t));
  }
  return rust::types::RustStructType::get(Dialect, fields);
}

Type RustTypeConverter::convertTupleType(TupleType type) {
  SmallVector<rust::types::RustType, 4> elements;
  for (Type t : type) {
    rust::types::RustType rt = convertType(t).cast<rust::types::RustType>();
    elements.push_back(rt);
  }
  return rust::types::RustType::getTupleTy(Dialect, elements);
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
                                        StringAttr::get(func.getName(), ctx)));
    auto newOp = rewriter.create<rust::RustFuncOp>(
        op->getLoc(), SmallVector<mlir::Type, 1>({}), operands, attributes);

    rewriter.inlineRegionBefore(func.getBody(), newOp.getBody(), newOp.end());
    rewriter.applySignatureConversion(&newOp.getBody(), sigConv);

    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), TypeConverter,
                                           &sigConv)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

void ArcToRustLoweringPass::runOnOperation() {
  RustTypeConverter typeConverter(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<rust::RustDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  OwningRewritePatternList patterns;
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

  patterns.insert<ArcCmpIOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCmpFOpLowering>(&getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<mlir::SinOp, ArcUnaryFloatOp::sin>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<mlir::CosOp, ArcUnaryFloatOp::cos>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::TanOp, ArcUnaryFloatOp::tan>>(
      &getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<arc::AsinOp, ArcUnaryFloatOp::asin>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::AcosOp, ArcUnaryFloatOp::acos>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::AtanOp, ArcUnaryFloatOp::atan>>(
      &getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<arc::SinhOp, ArcUnaryFloatOp::sinh>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<arc::CoshOp, ArcUnaryFloatOp::cosh>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<mlir::TanhOp, ArcUnaryFloatOp::tanh>>(
      &getContext(), typeConverter);

  patterns.insert<ArcUnaryFloatOpLowering<mlir::LogOp, ArcUnaryFloatOp::log>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<mlir::ExpOp, ArcUnaryFloatOp::exp>>(
      &getContext(), typeConverter);
  patterns.insert<ArcUnaryFloatOpLowering<mlir::SqrtOp, ArcUnaryFloatOp::sqrt>>(
      &getContext(), typeConverter);

  patterns.insert<IfOpLowering>(&getContext(), typeConverter);
  patterns.insert<IndexTupleOpLowering>(&getContext(), typeConverter);
  patterns.insert<BlockResultOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeTupleOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeStructOpLowering>(&getContext(), typeConverter);

  if (failed(applyFullConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createLowerToRustPass() {
  return std::make_unique<ArcToRustLoweringPass>();
}

const std::string ArcToRustLoweringPass::hexfCrate = "hexf";
const std::string ArcToRustLoweringPass::hexfVersion = "0.1.0";

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
