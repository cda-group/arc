package se.kth.cda.arc.ast.typer

import se.kth.cda.arc.ast.AST.ExprKind._
import se.kth.cda.arc.ast.AST.IterKind.IterKind
import se.kth.cda.arc.ast.AST.{Expr, ExprKind, Iter, IterKind}
import se.kth.cda.arc.ast.{CompoundType, ConcreteType, Type}

object PostProcess {
  implicit class PostProcessExpr(val self: Expr) extends AnyVal {

    def postProcess: Expr = {
      val newKind = self.kind match {
        case For(iterator, builder, body)        => For(iterator.postProcess, builder.postProcess, body.postProcess)
        case Let(symbol, bindingTy, value, body) => Let(symbol, bindingTy, value.postProcess, body.postProcess)
        case Lambda(params, body)                => Lambda(params, body.postProcess)
        case Cast(ty, expr)                      => Cast(ty, expr.postProcess)
        case ToVec(expr)                         => ToVec(expr.postProcess)
        case MakeStruct(elems)                   => MakeStruct(elems.map(_.postProcess))
        case MakeVec(elems)                      => MakeVec(elems.map(_.postProcess))
        case If(cond, onTrue, onFalse)           => If(cond.postProcess, onTrue.postProcess, onFalse.postProcess)
        case Select(cond, onTrue, onFalse)       => Select(cond.postProcess, onTrue.postProcess, onFalse.postProcess)
        case Iterate(initial, updateFunc)        => Iterate(initial.postProcess, updateFunc.postProcess)
        case Broadcast(expr)                     => Broadcast(expr.postProcess)
        case Serialize(expr)                     => Serialize(expr.postProcess)
        case Deserialize(ty, expr)               => Deserialize(ty, expr.postProcess)
        case Zip(params)                         => Zip(params.map(_.postProcess))
        case Hash(params)                        => Hash(params.map(_.postProcess))
        case Len(expr)                           => Len(expr.postProcess)
        case Lookup(data, key)                   => Lookup(data.postProcess, key.postProcess)
        case Slice(data, index, size)            => Slice(data.postProcess, index.postProcess, size.postProcess)
        case Sort(data, keyFunc)                 => Sort(data.postProcess, keyFunc.postProcess)
        case Drain(source, sink)                 => Drain(source.postProcess, sink.postProcess)
        case Negate(expr)                        => Negate(expr.postProcess)
        case Not(expr)                           => Not(expr.postProcess)
        case UnaryOp(kind, expr)                 => UnaryOp(kind, expr.postProcess)
        case Merge(builder, value)               => Merge(builder.postProcess, value.postProcess)
        case Result(expr)                        => Result(expr.postProcess)
        case NewBuilder(ty, args)                => NewBuilder(ty, args.map(_.postProcess))
        case BinOp(kind, lhs, rhs)               => BinOp(kind, lhs.postProcess, rhs.postProcess)
        case Application(expr, args)             => Application(expr.postProcess, args.map(_.postProcess))
        case Projection(expr, index)             => Projection(expr.postProcess, index)
        case Ascription(expr, ty)                => Ascription(expr.postProcess, ty)
        case CUDF(Right(reference), args, returnTy) =>
          CUDF(Right(reference.postProcess), args.map(_.postProcess), returnTy)
        case kind => kind
      }
      self.copy(kind = newKind)
    }
  }

  implicit class PostProcessIter(val self: Iter) extends AnyVal {

    import se.kth.cda.arc.ast.ASTUtils.TypeMethods

    def postProcess: Iter = {
      val newKind = self.kind match {
        case IterKind.UnknownIter =>
          if (self.data.ty.isArcType) {
            IterKind.NextIter
          } else {
            IterKind.ScalarIter
          }
        case kind => kind
      }
      self.copy(kind = newKind)
    }
  }
}
