package se.kth.cda.arc.ast.printer

import java.io.PrintStream

import se.kth.cda.arc.ast.AST.ExprKind._
import se.kth.cda.arc.ast.AST.IterKind._
import se.kth.cda.arc.ast.AST._
import se.kth.cda.arc.ast.Type.Builder._
import se.kth.cda.arc.ast.Type._
import se.kth.cda.arc.ast._

object MLIRPrinter {

  implicit class ASTToMLIR(val self: ASTNode) extends AnyVal {

    def toMLIR: String = self match {
      case symbol: Symbol      => symbol.toMLIR
      case Program(_, expr, _) => expr.toMLIR
      case expr: Expr          => expr.toMLIR
    }
  }

  implicit class SymbolToMLIR(val self: Symbol) extends AnyVal {

    def toMLIR: String = {
      val Symbol(name, token, scope) = self
      s"${name}_$scope"
    }
  }

  implicit class ExprToMLIR(val self: Expr) extends AnyVal {

    def toMLIR: String = self.kind match {
      case Literal.I8(raw, value)              =>
        s"""
           |
           |""".stripMargin
      case Literal.I16(raw, value)             => s""
      case Literal.I32(raw, value)             => s""
      case Literal.I64(raw, value)             => s""
      case Literal.U8(raw, value)              => s""
      case Literal.U16(raw, value)             => s""
      case Literal.U32(raw, value)             => s""
      case Literal.U64(raw, value)             => s""
      case Literal.F32(raw, value)             => s""
      case Literal.F64(raw, value)             => s""
      case Literal.Bool(raw, value)            => s""
      case Literal.UnitL(raw, value)           => s""
      case Literal.StringL(raw, value)         => s""
      case Let(symbol, bindingTy, value, body) => s""
      case If(cond, onTrue, onFalse)           => s""
      case Ident(symbol)                       => s""
      case Merge(builder, value)               => s""
      case Result(expr)                        => s""
      case BinOp(kind, lhs, rhs)               => s""
      case Lambda(params, body)                => s""
      case NewBuilder(ty, args)                => s""
      case For(iterator, builder, body)        => s""
      case Cast(ty, expr)                      => s""
      case ToVec(expr)                         => s""
      case MakeStruct(elems)                   => s""
      case MakeVec(elems)                      => s""
      case Select(cond, onTrue, onFalse)       => s""
      case Iterate(initial, updateFunc)        => s""
      case Broadcast(expr)                     => s""
      case Serialize(expr)                     => s""
      case Deserialize(ty, expr)               => s""
      case CUDF(reference, args, returnTy)     => s""
      case Zip(params)                         => s""
      case Hash(params)                        => s""
      case Len(expr)                           => s""
      case Lookup(data, key)                   => s""
      case Slice(data, index, size)            => s""
      case Sort(data, keyFunc)                 => s""
      case Drain(source, sink)                 => s""
      case Negate(expr)                        => s""
      case Not(expr)                           => s""
      case UnaryOp(kind, expr)                 => s""
      case Application(expr, args)             => s""
      case Projection(expr, index)             => s""
      case Ascription(expr, ty)                => s""
    }
  }

  implicit class IteratorToMLIR(val self: IterKind) extends AnyVal {

    def toMLIR: String = self match {
      case ScalarIter  => s""
      case SimdIter    => s""
      case FringeIter  => s""
      case NdIter      => s""
      case RangeIter   => s""
      case NextIter    => s""
      case KeyByIter   => s""
      case UnknownIter => s""
    }
  }

  implicit class TypeToMLIR(val self: Type) extends AnyVal {

    def toMLIR: String = self match {
      case I8                                                               => s""
      case I16                                                              => s""
      case I32                                                              => s""
      case I64                                                              => s""
      case U8                                                               => s""
      case U16                                                              => s""
      case U32                                                              => s""
      case U64                                                              => s""
      case F32                                                              => s""
      case F64                                                              => s""
      case Bool                                                             => s""
      case UnitT                                                            => s""
      case StringT                                                          => s""
      case Appender(elemTy, annotations)                                    => s""
      case StreamAppender(elemTy, annotations)                              => s""
      case Merger(elemTy, opTy, annotations)                                => s""
      case DictMerger(keyTy, valueTy, opTy, annotations)                    => s""
      case VecMerger(elemTy, opTy, annotations)                             => s""
      case GroupMerger(keyTy, valueTy, annotations)                         => s""
      case Windower(discTy, aggrTy, aggrMergeTy, aggrResultTy, annotations) => s""
      case Vec(elemTy)                                                      => s""
      case Dict(keyTy, valueTy)                                             => s""
      case Struct(elemTys)                                                  => s""
      case Simd(elemTy)                                                     => s""
      case Stream(elemTy)                                                   => s""
      case Function(params, returnTy)                                       => s""
      case TypeVariable(id)                                                 => s""
    }
  }

  implicit class NewBuilderToMLIR(val self: NewBuilder) extends AnyVal {

    def toMLIR: String = {
      val NewBuilder(ty, args) = self
      s""
    }
  }

  implicit class AnnotationToMLIR(val self: Annotations) extends AnyVal {

    def toMLIR: String = {
      val Annotations(params) = self
      s""
    }
  }

}
