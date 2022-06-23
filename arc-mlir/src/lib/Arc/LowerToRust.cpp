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

#include "Arc/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace arc;

//===----------------------------------------------------------------------===//
// ArcToRustLoweringPass
//===----------------------------------------------------------------------===//

/// This is a lowering of arc operations to the Rust dialect.
namespace {
struct ArcToRustLoweringPass : public LowerToRustBase<ArcToRustLoweringPass> {
  void runOnOperation() final;
};
} // end anonymous namespace.

class RustTypeConverter : public TypeConverter {
  MLIRContext *Ctx;
  rust::RustDialect *Dialect;

public:
  RustTypeConverter(MLIRContext *ctx);

  FunctionType convertFunctionSignature(Type, SignatureConversion &);

protected:
  Type convertADTType(arc::types::ADTType type);
  Type convertADTTemplateType(arc::types::ADTGenericType type);
  Type convertEnumType(arc::types::EnumType type);
  Type convertFloatType(FloatType type);
  Type convertFunctionType(FunctionType type);
  Type convertIntegerType(IntegerType type);
  Type convertNoneType(NoneType type);
  Type convertTensorType(RankedTensorType type);
  Type convertTupleType(TupleType type);
  Type convertStreamType(arc::types::StreamType type);
  Type convertSinkStreamType(arc::types::SinkStreamType type);
  Type convertSourceStreamType(arc::types::SourceStreamType type);
  Type convertStructType(arc::types::StructType type);
};

struct ArcReturnOpLowering : public OpConversionPattern<ArcReturnOp> {

  ArcReturnOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<ArcReturnOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(ArcReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustReturnOp>(
        op, llvm::None,
        adaptor.getOperands().size() ? adaptor.getOperands()[0] : Value());
    return success();
  };
};

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {

  ReturnOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<func::ReturnOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustReturnOp>(
        op, llvm::None,
        adaptor.getOperands().size() ? adaptor.getOperands()[0] : Value());
    return success();
  };
};

struct SCFWhileOpLowering : public OpConversionPattern<scf::WhileOp> {
  SCFWhileOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<scf::WhileOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    scf::WhileOp o = cast<scf::WhileOp>(op);
    SmallVector<Type, 4> resTys;

    for (auto ty : op->getResultTypes())
      resTys.push_back(TypeConverter.convertType(ty));

    rust::RustLoopOp loop = rewriter.create<rust::RustLoopOp>(
        op->getLoc(), resTys, adaptor.getOperands());

    rewriter.inlineRegionBefore(o.getBefore(), loop.before(),
                                loop.before().end());
    rewriter.inlineRegionBefore(o.getAfter(), loop.after(), loop.after().end());

    if (failed(rewriter.convertRegionTypes(&loop.before(), TypeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&loop.after(), TypeConverter)))
      return failure();

    rewriter.replaceOp(op, loop.getResults());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct SCFLoopConditionOpLowering
    : public OpConversionPattern<scf::ConditionOp> {

  SCFLoopConditionOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<scf::ConditionOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustLoopConditionOp>(
        op, llvm::None, adaptor.getOperands());
    return success();
  };
};

struct SCFLoopYieldOpLowering : public OpConversionPattern<scf::YieldOp> {

  SCFLoopYieldOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<scf::YieldOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustLoopYieldOp>(op, llvm::None,
                                                       adaptor.getOperands());
    return success();
  };
};

struct StdCallOpLowering : public OpConversionPattern<func::CallOp> {
  StdCallOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<func::CallOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(func::CallOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 4> resultTypes;
    for (auto r : o.getResultTypes())
      resultTypes.push_back(TypeConverter.convertType(r));
    rewriter.replaceOpWithNewOp<rust::RustCallOp>(
        o, adaptor.getCallee(), resultTypes, adaptor.getOperands());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct StdCallIndirectOpLowering
    : public OpConversionPattern<func::CallIndirectOp> {
  StdCallIndirectOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<func::CallIndirectOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(func::CallIndirectOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 4> resultTypes;
    for (auto r : o.getResultTypes())
      resultTypes.push_back(TypeConverter.convertType(r));
    rewriter.replaceOpWithNewOp<rust::RustCallIndirectOp>(
        o, resultTypes, adaptor.getOperands());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct ArithConstantOpLowering : public OpConversionPattern<arith::ConstantOp> {

  ArithConstantOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arith::ConstantOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp cOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Attribute attr = cOp.getValue();
    if (attr.isa<FloatAttr>())
      return convertFloat(cOp, rewriter);
    if (attr.isa<IntegerAttr>())
      return convertInteger(cOp, rewriter);
    cOp->emitError("unhandled constant type");
    return failure();
  };

private:
  RustTypeConverter &TypeConverter;

  LogicalResult returnResult(Operation *op, Type ty, StringRef value,
                             ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<rust::RustConstantOp>(op, ty, value);
    return success();
  }

  LogicalResult convertFloat(arith::ConstantOp op,
                             ConversionPatternRewriter &rewriter) const {
    FloatAttr attr = op.getValue().cast<FloatAttr>();
    Type ty = attr.getType();
    Type rustTy = TypeConverter.convertType(ty);
    APFloat f = attr.getValue();
    unsigned width = ty.getIntOrFloatBitWidth();

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
        memmove(hex + i + 1, hex + i, strlen(hex + i) + 1);
        hex[i] = '.';
        break;
      }
    }

    Twine str = "hexf" + Twine(width) + "!(\"" + hex + "\")";
    std::string cst = str.str();

    return returnResult(op, rustTy, cst, rewriter);
  }

  LogicalResult convertInteger(arith::ConstantOp op,
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

struct StdConstantOpLowering : public OpConversionPattern<func::ConstantOp> {

  StdConstantOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<func::ConstantOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(func::ConstantOp cOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Attribute attr = cOp.getValueAttr();
    if (attr.isa<SymbolRefAttr>())
      return convertSymbolRef(cOp, rewriter);
    cOp->emitError("unhandled constant type");
    return failure();
  };

private:
  RustTypeConverter &TypeConverter;

  LogicalResult returnResult(Operation *op, Type ty, StringRef value,
                             ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<rust::RustConstantOp>(op, ty, value);
    return success();
  }

  LogicalResult convertSymbolRef(func::ConstantOp op,
                                 ConversionPatternRewriter &rewriter) const {
    SymbolRefAttr attr = op.getValueAttr().cast<SymbolRefAttr>();
    Operation *refOp =
        SymbolTable::lookupNearestSymbolFrom(op->getParentOp(), attr);

    if (rust::RustFuncOp o = dyn_cast<rust::RustFuncOp>(refOp)) {
      Type ty = o.getFunctionType();
      Type rustTy = TypeConverter.convertType(ty);

      rewriter.replaceOpWithNewOp<rust::RustConstantOp>(op, rustTy,
                                                        o.getName());
      return success();
    } else if (rust::RustExtFuncOp o = dyn_cast<rust::RustExtFuncOp>(refOp)) {
      Type ty = o.getFunctionType();
      Type rustTy = TypeConverter.convertType(ty);

      rewriter.replaceOpWithNewOp<rust::RustConstantOp>(op, rustTy,
                                                        o.getName());
      return success();
    }
    op.emitError("unhandled symbol ref");
    return failure();
  }
};

struct ConstantIntOpLowering : public OpConversionPattern<arc::ConstantIntOp> {

  ConstantIntOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::ConstantIntOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(arc::ConstantIntOp cOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Attribute attr = cOp.getValue();
    if (attr.isa<IntegerAttr>())
      return convertInteger(cOp, rewriter);
    cOp->emitError("unhandled constant type");
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
      {
        SmallVector<char> tmp;
        v.toString(tmp, 10, ty.isSigned());
        std::string str(tmp.begin(), tmp.end());
        return returnResult(op, rustTy, str, rewriter);
      }
    default:
      op.emitError("unhandled constant integer width");
      return failure();
    }
  }
};

struct BlockResultOpLowering
    : public OpConversionPattern<arc::ArcBlockResultOp> {
  BlockResultOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::ArcBlockResultOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(arc::ArcBlockResultOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value, 1> values;
    auto operands = adaptor.getOperands();
    if (operands.size() != 0)
      values.push_back(operands[0]);
    rewriter.replaceOpWithNewOp<rust::RustBlockResultOp>(op, values);
    return success();
  };
};

struct ConstantADTOpLowering : public OpConversionPattern<arc::ConstantADTOp> {
  ConstantADTOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::ConstantADTOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(arc::ConstantADTOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.result().getType());
    rewriter.replaceOpWithNewOp<rust::RustConstantOp>(o, rustTy,
                                                      adaptor.value());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct IfOpLowering : public OpConversionPattern<arc::IfOp> {
  IfOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::IfOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(arc::IfOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Type, 1> resTys;

    if (o->getNumResults()) {
      Type retTy = TypeConverter.convertType(o.getType(0));
      resTys.push_back(retTy);
    }
    auto newOp = rewriter.create<rust::RustIfOp>(o.getLoc(), resTys,
                                                 adaptor.condition());

    rewriter.inlineRegionBefore(o.thenRegion(), newOp.thenRegion(),
                                newOp.thenRegion().end());
    rewriter.inlineRegionBefore(o.elseRegion(), newOp.elseRegion(),
                                newOp.elseRegion().end());
    rewriter.replaceOp(o, newOp.getResults());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct IndexTupleOpLowering : public OpConversionPattern<arc::IndexTupleOp> {
  IndexTupleOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::IndexTupleOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(IndexTupleOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustFieldAccessOp>(
        o, retTy, adaptor.value(), std::to_string(o.index()));
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct LoopBreakOpLowering : public OpConversionPattern<arc::LoopBreakOp> {
  LoopBreakOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::LoopBreakOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(arc::LoopBreakOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustLoopBreakOp>(op,
                                                       adaptor.getOperands());
    return success();
  };
};

struct PanicOpLowering : public OpConversionPattern<arc::PanicOp> {
  PanicOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::PanicOp>(typeConverter, ctx, 1), Ctx(ctx) {}

  LogicalResult
  matchAndRewrite(PanicOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto msg = o.msg();
    if (msg.hasValue())
      rewriter.replaceOpWithNewOp<rust::RustPanicOp>(
          o, StringAttr::get(Ctx, msg.getValue()));
    else
      rewriter.replaceOpWithNewOp<rust::RustPanicOp>(o, nullptr);

    return success();
  };

private:
  MLIRContext *Ctx;
};

struct MakeStructOpLowering : public OpConversionPattern<arc::MakeStructOp> {
  MakeStructOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::MakeStructOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(MakeStructOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustMakeStructOp>(o, retTy,
                                                        adaptor.getOperands());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct MakeEnumOpLowering : public OpConversionPattern<MakeEnumOp> {
  MakeEnumOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<MakeEnumOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(MakeEnumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(op.getType());
    std::string variant_str(op.variant());
    SmallVector<Value, 1> values;
    if (op.values().size())
      values.push_back(adaptor.values()[0]);
    rewriter.replaceOpWithNewOp<rust::RustMakeEnumOp>(op, retTy, values,
                                                      variant_str);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct MakeTensorOpLowering : public OpConversionPattern<arc::MakeTensorOp> {
  MakeTensorOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::MakeTensorOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(MakeTensorOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustTensorOp>(o, retTy,
                                                    adaptor.getOperands());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct MakeTupleOpLowering : public OpConversionPattern<arc::MakeTupleOp> {
  MakeTupleOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::MakeTupleOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(MakeTupleOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustTupleOp>(o, retTy,
                                                   adaptor.getOperands());
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
struct ArcIntArithmeticOpLowering : public OpConversionPattern<T> {
  ArcIntArithmeticOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<T>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(T o, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.getType());
    if (rustTy.isa<rust::types::RustTensorType>())
      rewriter.replaceOpWithNewOp<rust::RustBinaryRcOp>(
          o, rustTy, opStr[arithOp], adaptor.lhs(), adaptor.rhs());
    else
      rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
          o, rustTy, opStr[arithOp], adaptor.lhs(), adaptor.rhs());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[ArcIntArithmeticOp::LAST] = {"+", "&", "/", "|",
                                                 "*", "-", "%", "^"};
};

struct ArcCmpIOpLowering : public OpConversionPattern<arc::CmpIOp> {
  ArcCmpIOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::CmpIOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(arc::CmpIOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
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
    rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
        o, rustTy, cmpOp, adaptor.lhs(), adaptor.rhs());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

namespace StdArithmeticOp {
typedef enum {
  AddFOp = 0,
  DivFOp,
  MulFOp,
  SubFOp,
  RemFOp,
  AndIOp,
  OrIOp,
  XOrIOp,
  LAST
} Op;
}

template <class T, StdArithmeticOp::Op arithOp>
struct StdArithmeticOpLowering : public OpConversionPattern<T> {
  StdArithmeticOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<T>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(T o, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.getType());
    if (rustTy.isa<rust::types::RustTensorType>())
      rewriter.replaceOpWithNewOp<rust::RustBinaryRcOp>(
          o, rustTy, opStr[arithOp], adaptor.getLhs(), adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
          o, rustTy, opStr[arithOp], adaptor.getLhs(), adaptor.getRhs());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[StdArithmeticOp::LAST] = {"+", "/", "*", "-",
                                              "%", "&", "|", "^"};
};

namespace OpAsMethod {
typedef enum { PowFOp = 0, LAST } Op;
}

template <class T, OpAsMethod::Op theOp>
struct OpAsMethodLowering : public OpConversionPattern<T> {
  OpAsMethodLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<T>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(T o, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.getType());
    SmallVector<mlir::Value, 1> args;
    auto operands = adaptor.getOperands();
    for (unsigned i = 1; i < operands.size(); i++)
      args.push_back(operands[i]);

    rewriter.replaceOpWithNewOp<rust::RustMethodCallOp>(o, rustTy, opStr[theOp],
                                                        operands[0], args);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[StdArithmeticOp::LAST] = {"powf"};
};

struct StdCmpFOpLowering : public OpConversionPattern<mlir::arith::CmpFOp> {
  StdCmpFOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<mlir::arith::CmpFOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(mlir::arith::CmpFOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.getType());
    const char *cmpOp = NULL;
    switch (o.getPredicate()) {
    case mlir::arith::CmpFPredicate::OEQ:
      cmpOp = "==";
      break;
    case mlir::arith::CmpFPredicate::ONE:
      cmpOp = "!=";
      break;
    case mlir::arith::CmpFPredicate::OLT:
      cmpOp = "<";
      break;
    case mlir::arith::CmpFPredicate::OLE:
      cmpOp = "<=";
      break;
    case mlir::arith::CmpFPredicate::OGT:
      cmpOp = ">";
      break;
    case mlir::arith::CmpFPredicate::OGE:
      cmpOp = ">=";
      break;
    default:
      o.emitError("unhandled std.cmpf operation");
      return failure();
    }
    rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
        o, rustTy, cmpOp, adaptor.getLhs(), adaptor.getRhs());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct StdCmpIOpLowering : public OpConversionPattern<mlir::arith::CmpIOp> {
  StdCmpIOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<mlir::arith::CmpIOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(mlir::arith::CmpIOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.getType());
    const char *cmpOp = NULL;
    switch (o.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      cmpOp = "==";
      break;
    case mlir::arith::CmpIPredicate::ne:
      cmpOp = "!=";
      break;
    default:
      o.emitError("unhandled std.cmpi operation");
      return failure();
    }
    rewriter.replaceOpWithNewOp<rust::RustBinaryOp>(
        o, rustTy, cmpOp, adaptor.getLhs(), adaptor.getRhs());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct ArithSelectOpLowering
    : public OpConversionPattern<mlir::arith::SelectOp> {
  ArithSelectOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<mlir::arith::SelectOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(mlir::arith::SelectOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // There is no ternary if in Rust, so just sink this to an
    // ordinary arc if.
    auto thisIf = rewriter.replaceOpWithNewOp<arc::IfOp>(
        o, o.getType(), adaptor.getCondition());
    Block *thenBB = rewriter.createBlock(&thisIf.thenRegion());
    rewriter.setInsertionPointToEnd(thenBB);
    rewriter.create<arc::ArcBlockResultOp>(o.getLoc(), adaptor.getTrueValue());

    Block *elseBB = rewriter.createBlock(&thisIf.elseRegion());
    rewriter.setInsertionPointToEnd(elseBB);
    rewriter.create<arc::ArcBlockResultOp>(o.getLoc(), adaptor.getFalseValue());

    return success();
  };
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
struct ArcUnaryFloatOpLowering : public OpConversionPattern<T> {
  ArcUnaryFloatOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<T>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(T o, typename T::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type rustTy = TypeConverter.convertType(o.getType());
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<rust::RustMethodCallOp>(
        o, rustTy, opStr[arithOp], operands[0],
        SmallVector<mlir::Value, 1>({}));
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
  const char *opStr[ArcUnaryFloatOp::LAST] = {"sin",  "cos",  "tan",  "asin",
                                              "acos", "atan", "cosh", "sinh",
                                              "tanh", "ln",   "exp",  "sqrt"};
};

struct EnumAccessOpLowering : public OpConversionPattern<arc::EnumAccessOp> {
  EnumAccessOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::EnumAccessOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(EnumAccessOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    std::string variant_str(o.variant());
    rewriter.replaceOpWithNewOp<rust::RustEnumAccessOp>(
        o, retTy, adaptor.value(), variant_str);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct StructAccessOpLowering
    : public OpConversionPattern<arc::StructAccessOp> {
  StructAccessOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::StructAccessOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(StructAccessOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    std::string idx_str(o.field());
    rewriter.replaceOpWithNewOp<rust::RustFieldAccessOp>(
        o, retTy, adaptor.value(), idx_str);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct EnumCheckOpLowering : public OpConversionPattern<arc::EnumCheckOp> {
  EnumCheckOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::EnumCheckOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(EnumCheckOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    std::string variant_str(o.variant());
    rewriter.replaceOpWithNewOp<rust::RustEnumCheckOp>(
        o, retTy, adaptor.value(), variant_str);
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct EmitOpLowering : public OpConversionPattern<arc::EmitOp> {
  EmitOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::EmitOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(EmitOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustEmitOp>(o, adaptor.value(),
                                                  adaptor.stream());
    return success();
  };
};

struct ReceiveOpLowering : public OpConversionPattern<arc::ReceiveOp> {
  ReceiveOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::ReceiveOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(ReceiveOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type retTy = TypeConverter.convertType(o.getType());
    rewriter.replaceOpWithNewOp<rust::RustReceiveOp>(o, retTy,
                                                     adaptor.source());
    return success();
  };

private:
  RustTypeConverter &TypeConverter;
};

struct SendOpLowering : public OpConversionPattern<arc::SendOp> {
  SendOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<arc::SendOp>(typeConverter, ctx, 1) {}

  LogicalResult
  matchAndRewrite(SendOp o, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<rust::RustSendOp>(o, adaptor.value(),
                                                  adaptor.sink());
    return success();
  };
};

RustTypeConverter::RustTypeConverter(MLIRContext *ctx)
    : Ctx(ctx), Dialect(ctx->getOrLoadDialect<rust::RustDialect>()) {
  addConversion([&](arc::types::ADTType type) { return convertADTType(type); });
  addConversion([&](arc::types::ADTGenericType type) {
    return convertADTTemplateType(type);
  });
  addConversion(
      [&](arc::types::EnumType type) { return convertEnumType(type); });
  addConversion([&](FloatType type) { return convertFloatType(type); });
  addConversion([&](FunctionType type) { return convertFunctionType(type); });
  addConversion([&](IntegerType type) { return convertIntegerType(type); });
  addConversion([&](RankedTensorType type) { return convertTensorType(type); });
  addConversion([&](TupleType type) { return convertTupleType(type); });
  addConversion(
      [&](arc::types::StreamType type) { return convertStreamType(type); });
  addConversion([&](arc::types::SinkStreamType type) {
    return convertSinkStreamType(type);
  });
  addConversion([&](arc::types::SourceStreamType type) {
    return convertSourceStreamType(type);
  });
  addConversion(
      [&](arc::types::StructType type) { return convertStructType(type); });
  addConversion([&](NoneType type) { return convertNoneType(type); });

  // RustType is legal, so add a pass-through conversion.
  addConversion([](rust::types::RustType type) { return type; });
  addConversion([](rust::types::RustEnumType type) { return type; });
  addConversion([](rust::types::RustStreamType type) { return type; });
  addConversion([](rust::types::RustSinkStreamType type) { return type; });
  addConversion([](rust::types::RustSourceStreamType type) { return type; });
  addConversion([](rust::types::RustStructType type) { return type; });
  addConversion([](rust::types::RustGenericADTType type) { return type; });
  addConversion([](rust::types::RustTensorType type) { return type; });
  addConversion([](rust::types::RustTupleType type) { return type; });
}

Type RustTypeConverter::convertADTType(arc::types::ADTType type) {
  return rust::types::RustGenericADTType::get(Dialect, type.getTypeName(), {});
}

Type RustTypeConverter::convertADTTemplateType(
    arc::types::ADTGenericType type) {
  SmallVector<Type, 4> parameters;

  for (const auto &t : type.getParameterTypes())
    parameters.push_back(convertType(t));
  return rust::types::RustGenericADTType::get(Dialect, type.getTemplateName(),
                                              parameters);
}

Type RustTypeConverter::convertEnumType(arc::types::EnumType type) {
  SmallVector<rust::types::RustType::EnumVariantTy, 4> variants;
  for (const auto &f : type.getVariants()) {
    Type t = convertType(f.second);
    variants.push_back(std::make_pair(f.first, t));
  }
  return rust::types::RustEnumType::get(Dialect, variants);
}

Type RustTypeConverter::convertFloatType(FloatType type) {
  if (type.isa<Float32Type>())
    return rust::types::RustType::getFloatTy(Dialect);
  if (type.isa<Float64Type>())
    return rust::types::RustType::getDoubleTy(Dialect);
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

Type RustTypeConverter::convertNoneType(NoneType type) {
  return rust::types::RustType::getNoneTy(Dialect);
}

Type RustTypeConverter::convertStreamType(arc::types::StreamType type) {
  return rust::types::RustStreamType::get(Dialect, convertType(type.getType()));
}

Type RustTypeConverter::convertSinkStreamType(arc::types::SinkStreamType type) {
  return rust::types::RustSinkStreamType::get(Dialect,
                                              convertType(type.getType()));
}

Type RustTypeConverter::convertSourceStreamType(
    arc::types::SourceStreamType type) {
  return rust::types::RustSourceStreamType::get(Dialect,
                                                convertType(type.getType()));
}

Type RustTypeConverter::convertStructType(arc::types::StructType type) {
  SmallVector<rust::types::RustType::StructFieldTy, 4> fields;
  for (const auto &f : type.getFields()) {
    Type t = convertType(f.second);
    fields.push_back(std::make_pair(f.first, t));
  }
  return rust::types::RustStructType::get(Dialect, type.isCompact(), fields);
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

struct FuncOpLowering : public OpConversionPattern<mlir::func::FuncOp> {

  FuncOpLowering(MLIRContext *ctx, RustTypeConverter &typeConverter)
      : OpConversionPattern<mlir::func::FuncOp>(typeConverter, ctx, 1),
        TypeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(mlir::func::FuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = func.getContext();
    SmallVector<NamedAttribute, 4> attributes;

    // Check that this function is contained in a named module
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module.getName()) {
      emitError(module.getLoc())
          << "Module is missing a name (is the module implicitly created?)";
      return failure();
    }

    if (func->hasAttr("arc.rust_name"))
      attributes.push_back(NamedAttribute(StringAttr::get(ctx, "arc.rust_name"),
                                          func->getAttr("arc.rust_name")));

    if (func->hasAttr("arc.mod_name"))
      attributes.push_back(NamedAttribute(StringAttr::get(ctx, "arc.mod_name"),
                                          func->getAttr("arc.mod_name")));

    if (func->hasAttr("arc.is_task"))
      attributes.push_back(NamedAttribute(StringAttr::get(ctx, "arc.is_task"),
                                          func->getAttr("arc.is_task")));
    if (func->hasAttr("rust.annotation"))
      attributes.push_back(
          NamedAttribute(StringAttr::get(ctx, "rust.annotation"),
                         func->getAttr("rust.annotation")));

    if (func->hasAttr("rust.declare"))
      attributes.push_back(NamedAttribute(StringAttr::get(ctx, "rust.declare"),
                                          func->getAttr("rust.declare")));

    if (func->hasAttr("rust.async"))
      attributes.push_back(NamedAttribute(StringAttr::get(ctx, "rust.async"),
                                          func->getAttr("rust.async")));

    TypeConverter::SignatureConversion sigConv(func.getNumArguments());
    mlir::FunctionType funcType =
        TypeConverter.convertFunctionSignature(func.getFunctionType(), sigConv);

    attributes.push_back(NamedAttribute(StringAttr::get(ctx, "function_type"),
                                        TypeAttr::get(funcType)));
    attributes.push_back(NamedAttribute(StringAttr::get(ctx, "sym_name"),
                                        StringAttr::get(ctx, func.getName())));

    SmallVector<Value, 4> operandsArray;
    for (auto i : adaptor.getOperands())
      operandsArray.push_back(i);
    ArrayRef<Value> operands = llvm::makeArrayRef(operandsArray);
    if (func.isExternal())
      return buildExternalFun(rewriter, operands, attributes, func, func,
                              sigConv, ctx);
    return buildLocalFun(rewriter, operands, attributes, func, func, sigConv,
                         ctx);
  };

private:
  RustTypeConverter &TypeConverter;

  LogicalResult buildLocalFun(ConversionPatternRewriter &rewriter,
                              ArrayRef<Value> &operands,
                              SmallVector<NamedAttribute, 4> &attributes,
                              mlir::func::FuncOp &func, Operation *op,
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
                                 mlir::func::FuncOp &func, Operation *op,
                                 TypeConverter::SignatureConversion &sigConv,
                                 MLIRContext *ctx) const {
    auto newOp = rewriter.create<rust::RustExtFuncOp>(
        op->getLoc(), SmallVector<mlir::Type, 1>({}), operands, attributes);
    rewriter.applySignatureConversion(&newOp.getBody(), sigConv);

    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), TypeConverter,
                                           &sigConv)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

void ArcToRustLoweringPass::runOnOperation() {
  RustTypeConverter typeConverter(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<rust::RustDialect>();
  target.addLegalOp<ModuleOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ReturnOpLowering>(&getContext(), typeConverter);
  patterns.insert<ArcReturnOpLowering>(&getContext(), typeConverter);
  patterns.insert<FuncOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdConstantOpLowering>(&getContext(), typeConverter);
  patterns.insert<ArithConstantOpLowering>(&getContext(), typeConverter);
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

  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::AddFOp, StdArithmeticOp::AddFOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::DivFOp, StdArithmeticOp::DivFOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::MulFOp, StdArithmeticOp::MulFOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::SubFOp, StdArithmeticOp::SubFOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::RemFOp, StdArithmeticOp::RemFOp>>(
      &getContext(), typeConverter);

  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::AndIOp, StdArithmeticOp::AndIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::OrIOp, StdArithmeticOp::OrIOp>>(
      &getContext(), typeConverter);
  patterns.insert<
      StdArithmeticOpLowering<mlir::arith::XOrIOp, StdArithmeticOp::XOrIOp>>(
      &getContext(), typeConverter);

  patterns.insert<OpAsMethodLowering<math::PowFOp, OpAsMethod::PowFOp>>(
      &getContext(), typeConverter);

  patterns.insert<ArcCmpIOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCmpFOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCmpIOpLowering>(&getContext(), typeConverter);

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
  patterns.insert<BlockResultOpLowering>(&getContext(), typeConverter);
  patterns.insert<ConstantADTOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeTupleOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeTensorOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeEnumOpLowering>(&getContext(), typeConverter);
  patterns.insert<MakeStructOpLowering>(&getContext(), typeConverter);
  patterns.insert<LoopBreakOpLowering>(&getContext(), typeConverter);
  patterns.insert<EnumAccessOpLowering>(&getContext(), typeConverter);
  patterns.insert<EnumCheckOpLowering>(&getContext(), typeConverter);
  patterns.insert<EmitOpLowering>(&getContext(), typeConverter);
  patterns.insert<PanicOpLowering>(&getContext(), typeConverter);
  patterns.insert<ArithSelectOpLowering>(&getContext(), typeConverter);
  patterns.insert<ReceiveOpLowering>(&getContext(), typeConverter);
  patterns.insert<SendOpLowering>(&getContext(), typeConverter);
  patterns.insert<StructAccessOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCallOpLowering>(&getContext(), typeConverter);
  patterns.insert<StdCallIndirectOpLowering>(&getContext(), typeConverter);
  patterns.insert<SCFWhileOpLowering>(&getContext(), typeConverter);
  patterns.insert<SCFLoopConditionOpLowering>(&getContext(), typeConverter);
  patterns.insert<SCFLoopYieldOpLowering>(&getContext(), typeConverter);

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> arc::createLowerToRustPass() {
  return std::make_unique<ArcToRustLoweringPass>();
}
