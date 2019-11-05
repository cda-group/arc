package se.kth.cda.arc.ast.transformer

import se.kth.cda.arc.ast.AST._
import se.kth.cda.arc.Utils.TryVector
import se.kth.cda.arc.ast.{Builder, Type}

import scala.util.{Success, Try}

object Transformer {
  type ExprTransformer[Env] = (Expr, Env) => Try[(Option[Expr], Env)]
  type ExprKindTransformer[Env] = (ExprKind, Env) => Try[(Option[ExprKind], Env)]
  type TypeTransformer[Env] = (Type, Env) => Try[(Option[Type], Env)]

  def exprTransformer[Env](onExpr: ExprTransformer[Env]): Transformer[Env] = Transformer(onExpr, defaultType)

  def exprKindTransformer[Env](onExprKind: ExprKindTransformer[Env]): Transformer[Env] =
    Transformer(
    onExpr = (expr, env) =>
      for {
        (newKind, newEnv) <- onExprKind(expr.kind, env)
      } yield {
        (newKind.map(Expr(_, expr.ty, expr.ctx)), newEnv)
    },
    onType = defaultType
  )

  def typeTransformer[Env](onType: TypeTransformer[Env]): Transformer[Env] = Transformer(defaultExpr, onType)

  def biTransformer[Env](onExpr: ExprTransformer[Env], onType: Transformer.TypeTransformer[Env]): Transformer[Env] =
    Transformer(onExpr, onType)

  def defaultExpr[Env]: ExprTransformer[Env] = (_, env) => Success((None, env))

  def defaultType[Env]: TypeTransformer[Env] = (_, env) => Success((None, env))
}

final case class Transformer[Env](onExpr: Transformer.ExprTransformer[Env], onType: Transformer.TypeTransformer[Env]) {

  import ExprKind._

  def transform(initialExpr: Expr, initialEnv: Env): Try[Option[Expr]] =
    onExpr(initialExpr, initialEnv).flatMap { result =>
      val (didChange, changedExpr, env) = result match {
        case (Some(newExpr), newEnv) => (true, newExpr, newEnv)
        case (None, newEnv)          => (false, initialExpr, newEnv)
      }
      val newExprKind: Try[Option[ExprKind]] = changedExpr.kind match {
        case _: Ident                      => Success(None)
        case _: Literal[_]                 => Success(None)
        case ToVec(expr)                   => transformMap(expr, env)(ToVec)
        case Broadcast(expr)               => transformMap(expr, env)(Broadcast)
        case Serialize(expr)               => transformMap(expr, env)(Serialize)
        case Len(expr)                     => transformMap(expr, env)(Len)
        case Negate(expr)                  => transformMap(expr, env)(Negate)
        case Not(expr)                     => transformMap(expr, env)(Not)
        case UnaryOp(kind, expr)           => transformMap(expr, env)(UnaryOp(kind, _))
        case Result(expr)                  => transformMap(expr, env)(Result)
        case Lambda(params, body)          => transformMap(body, env)(Lambda(params, _))
        case Projection(expr, index)       => transformMap(expr, env)(Projection(_, index))
        case MakeStruct(elems)             => transformMap(elems, env)(MakeStruct)
        case MakeVec(elems)                => transformMap(elems, env)(MakeVec)
        case Zip(params)                   => transformMap(params, env)(Zip)
        case Hash(params)                  => transformMap(params, env)(Hash)
        case If(cond, onTrue, onFalse)     => transformMap((cond, onTrue, onFalse), env)(If)
        case Select(cond, onTrue, onFalse) => transformMap((cond, onTrue, onFalse), env)(Select)
        case Iterate(initial, updateFunc)  => transformMap((initial, updateFunc), env)(Iterate)
        case Lookup(data, key)             => transformMap((data, key), env)(Lookup)
        case Slice(data, index, size)      => transformMap((data, index, size), env)(Slice)
        case Sort(data, keyFunc)           => transformMap((data, keyFunc), env)(Sort)
        case Drain(source, sink)           => transformMap((source, sink), env)(Drain)
        case Merge(builder, value)         => transformMap((builder, value), env)(Merge)
        case BinOp(kind, left, right)      => transformMap((left, right), env)(BinOp(kind, _, _))
        case Let(symbol, bindingTy, value, body) =>
          for {
            (newBindingTy, newEnv) <- onType(bindingTy, env)
            newValue <- transform(value, newEnv)
            newBody <- transform(body, newEnv)
          } yield {
            if (newBindingTy.isEmpty && newValue.isEmpty && newBody.isEmpty) {
              None
            } else {
              Some(
                Let(
                  symbol,
                  bindingTy = newBindingTy.getOrElse(bindingTy),
                  value = newValue.getOrElse(value),
                  body = newBody.getOrElse(body)
                )
              )
            }
          }
        case Cast(ty, expr) =>
          for {
            (newTy, newEnv) <- onType(ty, env)
            newExpr <- transform(expr, newEnv)
          } yield {
            if (newTy.isEmpty && newExpr.isEmpty) {
              None
            } else {
              Some(Cast(ty, expr = newExpr.getOrElse(expr)))
            }
          }
        case Deserialize(ty, expr) =>
          for {
            (newTy, newEnv) <- onType(ty, env)
            newExpr <- transform(expr, newEnv)
          } yield {
            if (newTy.isEmpty && newExpr.isEmpty) {
              None
            } else {
              Some(
                Deserialize(
                  ty = newTy.getOrElse(ty),
                  expr = newExpr.getOrElse(expr)
                )
              )
            }
          }
        case CUDF(Left(symbol), args, returnTy) =>
          for {
            (newReturnTy, newEnv) <- onType(returnTy, env)
            newArgs <- transform(args, newEnv)
          } yield {
            if (newReturnTy.isEmpty && newArgs.isEmpty) {
              None
            } else {
              Some(
                CUDF(
                  reference = Left(symbol),
                  args = newArgs.getOrElse(args),
                  returnTy = newReturnTy.getOrElse(returnTy)
                )
              )
            }
          }
        case CUDF(Right(pointer), args, returnTy) =>
          for {
            (newReturnTy, newEnv) <- onType(returnTy, env)
            newPointer <- transform(pointer, newEnv)
            newArgs <- transform(args, newEnv)
          } yield {
            if (newReturnTy.isEmpty && newPointer.isEmpty && newArgs.isEmpty) {
              None
            } else {
              Some(
                CUDF(
                  reference = Right(newPointer.getOrElse(pointer)),
                  args = newArgs.getOrElse(args),
                  returnTy = newReturnTy.getOrElse(returnTy)
                )
              )
            }
          }
        case For(iterator, builder, body) =>
          transformMap((iterator.data, builder, body), env) { (newData, newBuilder, newBody) =>
            For(iterator.copy(data = newData), newBuilder, newBody)
          }
        case NewBuilder(ty, args) =>
          for {
            (newTy, newEnv) <- onType(ty, env)
            newArgs <- transform(args, newEnv)
          } yield {
            if (newTy.isEmpty && newArgs.isEmpty) {
              None
            } else {
              Some(
                NewBuilder(
                  ty = newTy.getOrElse(ty).asInstanceOf[Builder],
                  args = newArgs.getOrElse(args)
                )
              )
            }
          }
        case Application(expr, args) =>
          for {
            newExpr <- transform(expr, env)
            newArgs <- transform(args, env)
          } yield {
            if (newExpr.isEmpty && newArgs.isEmpty) {
              None
            } else {
              Some(
                Application(
                  expr = newExpr.getOrElse(expr),
                  args = newArgs.getOrElse(args)
                )
              )
            }
          }
        case Ascription(expr, ty) =>
          for {
            (newTy, newEnv) <- onType(ty, env)
            newExpr <- transform(expr, newEnv)
          } yield {
            if (newTy.isEmpty && newExpr.isEmpty) {
              None
            } else {
              Some(
                Ascription(
                  expr = newExpr.getOrElse(expr),
                  ty = newTy.getOrElse(ty)
                )
              )
            }
          }
      }
      newExprKind.map {
        case Some(kind) => Some(Expr(kind, changedExpr.ty, changedExpr.ctx));
        case None       => if (didChange) Some(changedExpr) else None;
      }
    }

  def transformMap(expr: Expr, env: Env)(placer: Expr => ExprKind): Try[Option[ExprKind]] = {
    transform(expr, env).map(_.map(placer))
  }

  def transformMap(exprs: (Expr, Expr), env: Env)(placer: (Expr, Expr) => ExprKind): Try[Option[ExprKind]] = {
    for {
      newExpr1 <- transform(exprs._1, env)
      newExpr2 <- transform(exprs._2, env)
    } yield {
      if (newExpr1.isEmpty && newExpr2.isEmpty) {
        None
      } else {
        Some(placer(newExpr1.getOrElse(exprs._1), newExpr2.getOrElse(exprs._2)))
      }
    }
  }

  def transformMap(exprs: (Expr, Expr, Expr), env: Env)(
      placer: (Expr, Expr, Expr) => ExprKind): Try[Option[ExprKind]] = {
    for {
      newExpr1 <- transform(exprs._1, env)
      newExpr2 <- transform(exprs._2, env)
      newExpr3 <- transform(exprs._3, env)
    } yield {
      if (newExpr1.isEmpty && newExpr2.isEmpty && newExpr3.isEmpty) {
        None
      } else {
        Some(placer(newExpr1.getOrElse(exprs._1), newExpr2.getOrElse(exprs._2), newExpr3.getOrElse(exprs._3)))
      }
    }
  }

  def transform(exprs: Vector[Expr], env: Env): Try[Option[Vector[Expr]]] = {
    for {
      newExprs <- exprs.map(transform(_, env)).sequence
    } yield {
      if (newExprs.forall(_.isEmpty)) {
        None
      } else {
        Some(exprs.zip(newExprs).map {
          case (_, Some(newExpr)) => newExpr
          case (oldExpr, None)    => oldExpr
        })
      }
    }

  }

  def transformMap(elems: Vector[Expr], env: Env)(placer: Vector[Expr] => ExprKind): Try[Option[ExprKind]] =
    transform(elems, env).map(_.map(placer))
}
