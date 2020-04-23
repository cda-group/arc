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
  void runOnFunction() final;

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

protected:
  Type convertFloatType(FloatType type);
  Type convertFunctionType(FunctionType type);
};

struct ReturnOpLowering : public ConversionPattern {

  ReturnOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ReturnOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Can't use a replaceOpWithNewOp as the rust operator has a
    // result (to encapsulate the type).
    rewriter.create<rust::RustReturnOp>(op->getLoc(), operands[0]);
    rewriter.eraseOp(op);
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
};

RustTypeConverter::RustTypeConverter(MLIRContext *ctx)
    : Ctx(ctx), Dialect(ctx->getRegisteredDialect<rust::RustDialect>()) {
  addConversion([&](FloatType type) { return convertFloatType(type); });
  addConversion([&](FunctionType type) { return convertFunctionType(type); });

  // RustType is legal, so add a pass-through conversion.
  addConversion([](rust::types::RustType type) { return type; });
}

Type RustTypeConverter::convertFloatType(FloatType type) {
  switch (type.getKind()) {
  case mlir::StandardTypes::F32:
    return rust::types::RustType::getFloatTy(Dialect);
  case mlir::StandardTypes::F64:
    return rust::types::RustType::getDoubleTy(Dialect);
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
    mlir::FunctionType funcTy = func.getType().cast<mlir::FunctionType>();
    mlir::Type funcType = TypeConverter.convertType(funcTy);

    attributes.push_back(
        NamedAttribute(Identifier::get("type", ctx), TypeAttr::get(funcType)));
    attributes.push_back(NamedAttribute(Identifier::get("sym_name", ctx),
                                        StringAttr::get(func.getName(), ctx)));
    auto newOp = rewriter.create<rust::RustFuncOp>(
        op->getLoc(), SmallVector<mlir::Type, 1>({}), operands, attributes);

    rewriter.inlineRegionBefore(func.getBody(), newOp.getBody(), newOp.end());
    rewriter.eraseOp(op);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

void ArcToRustLoweringPass::runOnFunction() {
  RustTypeConverter typeConverter(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<rust::RustDialect>();

  OwningRewritePatternList patterns;
  patterns.insert<ReturnOpLowering>(&getContext());
  patterns.insert<FuncOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdConstantOpLowering>(&getContext(), typeConverter);

  if (failed(
          applyFullConversion(getFunction(), target, patterns, &typeConverter)))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> arc::createLowerToRustPass() {
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
