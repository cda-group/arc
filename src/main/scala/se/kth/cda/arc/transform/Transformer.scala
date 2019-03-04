package se.kth.cda.arc.transform

import se.kth.cda.arc.AST._
import se.kth.cda.arc.Utils.TryVector
import se.kth.cda.arc.{BuilderType, Type}

import scala.util.{Success, Try}

object Transformer {
  type ExprTransformer[Env] = (Expr, Env) => Try[(Option[Expr], Env)]
  type ExprKindTransformer[Env] = (ExprKind, Env) => Try[(Option[ExprKind], Env)]
  type TypeTransformer[Env] = (Type, Env) => Try[(Option[Type], Env)]

  def exprTransformer[Env](onExpr: ExprTransformer[Env]): Transformer[Env] = {
    new Transformer(onExpr, defaultType)
  }

  def exprKindTransformer[Env](onExprKind: ExprKindTransformer[Env]): Transformer[Env] = {
    val onExpr: ExprTransformer[Env] = (expr, env) =>
      for {
        (newKindO, env1) <- onExprKind(expr.kind, env)
      } yield {
        val newExprO = newKindO.map(newKind => Expr(newKind, expr.ty, expr.ctx))
        (newExprO, env1)
    }
    new Transformer(onExpr, defaultType)
  }

  def typeTransformer[Env](onType: TypeTransformer[Env]): Transformer[Env] = {
    new Transformer(defaultExpr, onType)
  }

  def biTransformer[Env](onExpr: ExprTransformer[Env], onType: Transformer.TypeTransformer[Env]): Transformer[Env] = {
    new Transformer(onExpr, onType)
  }

  def defaultExpr[Env]: ExprTransformer[Env] = (_, env) => Success(None, env)

  def defaultType[Env]: TypeTransformer[Env] = (_, env) => Success(None, env)
}

class Transformer[Env](val onExpr: Transformer.ExprTransformer[Env], val onType: Transformer.TypeTransformer[Env]) {

  def transform(e: Expr, env: Env): Try[Option[Expr]] = {
    import ExprKind._

    onExpr(e, env).flatMap { r =>
      val (changed, newE, env0) = r match {
        case (Some(newE), newEnv) => (true, newE, newEnv)
        case (None, newEnv)       => (false, e, newEnv)
      }
      val newExprO: Try[Option[ExprKind]] = newE.kind match {
        case Let(name, bindingTy, value, body) =>
          for {
            (newBTyO, env1) <- onType(bindingTy, env0)
            newValueO <- transform(value, env1)
            newBodyO <- transform(body, env1)
          } yield {
            if (newBTyO.isEmpty && newValueO.isEmpty && newBodyO.isEmpty) {
              None
            } else {
              val newBTy = newBTyO.getOrElse(bindingTy)
              val newValue = newValueO.getOrElse(value)
              val newBody = newBodyO.getOrElse(body)
              Some(Let(name, newBTy, newValue, newBody))
            }
          }
        case Lambda(params, body) => transformMap(body, env0)(newBody => Lambda(params, newBody))
        case _: Literal[_]        => Success(None)
        case Cast(ty, expr) =>
          for {
            (newTyO, env1) <- onType(ty, env0)
            newExprO <- transform(expr, env1)
          } yield {
            if (newTyO.isEmpty && newExprO.isEmpty) {
              None
            } else {
              val newExpr = newExprO.getOrElse(expr)
              Some(Cast(ty, expr))
            }
          }
        case ToVec(expr)       => transformMap(expr, env0)(newExpr => ToVec(newExpr))
        case _: Ident          => Success(None)
        case MakeStruct(elems) => transform(elems, env0, newElems => MakeStruct(newElems))
        case MakeVec(elems)    => transform(elems, env0, newElems => MakeVec(newElems))
        case If(cond, onTrue, onFalse) =>
          transform((cond, onTrue, onFalse), env0)((newCond, newOnTrue, newOnFalse) =>
            If(newCond, newOnTrue, newOnFalse))
        case Select(cond, onTrue, onFalse) =>
          transform((cond, onTrue, onFalse), env0)((newCond, newOnTrue, newOnFalse) =>
            Select(newCond, newOnTrue, newOnFalse))
        case Iterate(initial, updateFunc) =>
          transform((initial, updateFunc), env0)((newInitial, newUpdateFunc) => Iterate(newInitial, newUpdateFunc))
        case Broadcast(expr) => transformMap(expr, env0)(newExpr => Broadcast(newExpr))
        case Serialize(expr) => transformMap(expr, env0)(newExpr => Serialize(newExpr))
        case Deserialize(ty, expr) =>
          for {
            (newTyO, env1) <- onType(ty, env0)
            newExprO <- transform(expr, env1)
          } yield {
            if (newTyO.isEmpty && newExprO.isEmpty) {
              None
            } else {
              val newTy = newTyO.getOrElse(ty)
              val newExpr = newExprO.getOrElse(expr)
              Some(Deserialize(newTy, newExpr))
            }
          }
        case CUDF(Left(name), args, returnTy) =>
          for {
            (newReturnTyO, env1) <- onType(returnTy, env0)
            newArgsO <- transform(args, env1)
          } yield {
            if (newReturnTyO.isEmpty && newArgsO.isEmpty) {
              None
            } else {
              val newReturnTy = newReturnTyO.getOrElse(returnTy)
              val newArgs = newArgsO.getOrElse(args)
              Some(CUDF(Left(name), newArgs, newReturnTy))
            }
          }
        case CUDF(Right(pointer), args, returnTy) =>
          for {
            (newReturnTyO, env1) <- onType(returnTy, env0)
            newPointerO <- transform(pointer, env1)
            newArgsO <- transform(args, env1)
          } yield {
            if (newReturnTyO.isEmpty && newPointerO.isEmpty && newArgsO.isEmpty) {
              None
            } else {
              val newReturnTy = newReturnTyO.getOrElse(returnTy)
              val newPointer = newPointerO.getOrElse(pointer)
              val newArgs = newArgsO.getOrElse(args)
              Some(CUDF(Right(newPointer), newArgs, newReturnTy))
            }
          }
        case Zip(params)  => transform(params, env0, newParams => Zip(newParams))
        case Hash(params) => transform(params, env0, newParams => Hash(newParams))
        case For(iterator, builder, body) =>
          transform((iterator.data, builder, body), env0)((newData, newBuilder, newBody) =>
            For(iterator.copy(data = newData), newBuilder, newBody))
        case Len(expr)         => transformMap(expr, env0)(newExpr => Len(newExpr))
        case Lookup(data, key) => transform((data, key), env0)((newData, newKey) => Lookup(newData, newKey))
        case Slice(data, index, size) =>
          transform((data, index, size), env0)((newData, newIndex, newSize) => Slice(newData, newIndex, newSize))
        case Sort(data, keyFunc) => transform((data, keyFunc), env0)((newData, newKeyFunc) => Sort(newData, newKeyFunc))
        case Negate(expr)        => transformMap(expr, env0)(newExpr => Negate(newExpr))
        case Not(expr)           => transformMap(expr, env0)(newExpr => Not(newExpr))
        case UnaryOp(kind, expr) => transformMap(expr, env0)(newExpr => UnaryOp(kind, newExpr))
        case Merge(builder, value) =>
          transform((builder, value), env0)((newBuilder, newValue) => Merge(newBuilder, newValue))
        case Result(expr) => transformMap(expr, env0)(newExpr => Result(newExpr))
        case NewBuilder(ty, Some(arg)) =>
          for {
            (newTyO, env1) <- onType(ty, env0)
            newArgO <- transform(arg, env1)
          } yield {
            if (newTyO.isEmpty && newArgO.isEmpty) {
              None
            } else {
              val newTy = newTyO.getOrElse(ty)
              val newArg = newArgO.getOrElse(arg)
              Some(NewBuilder(newTy.asInstanceOf[BuilderType], Some(newArg)))
            }
          }
        case NewBuilder(ty, None) =>
          for {
            (newTyO, _) <- onType(ty, env0)
          } yield {
            newTyO.map(newTy => NewBuilder(newTy.asInstanceOf[BuilderType], None))
          }
        case BinOp(kind, left, right) =>
          transform((left, right), env0)((newLeft, newRight) => BinOp(kind, newLeft, newRight))
        case Application(funcExpr, args) =>
          for {
            newFuncExprO <- transform(funcExpr, env0)
            newArgsO <- transform(args, env0)
          } yield {
            if (newFuncExprO.isEmpty && newArgsO.isEmpty) {
              None
            } else {
              val newFuncExpr = newFuncExprO.getOrElse(funcExpr)
              val newArgs = newArgsO.getOrElse(args)
              Some(Application(newFuncExpr, newArgs))
            }
          }
        case Projection(structExpr, index) =>
          transformMap(structExpr, env0)(newStructExpr => Projection(newStructExpr, index))
        case Ascription(expr, ty) =>
          for {
            (newTyO, env1) <- onType(ty, env0)
            newExprO <- transform(expr, env1)
          } yield {
            if (newTyO.isEmpty && newExprO.isEmpty) {
              None
            } else {
              val newTy = newTyO.getOrElse(ty)
              val newExpr = newExprO.getOrElse(expr)
              Some(Ascription(newExpr, newTy))
            }
          }
      }
      newExprO.map {
        case Some(ek) => Some(Expr(ek, newE.ty, newE.ctx));
        case None     => if (changed) Some(newE) else None;
      }
    }

  }

  def transformMap(e: Expr, env: Env)(placer: Expr => ExprKind): Try[Option[ExprKind]] = {
    transform(e, env).map(_.map(placer))
  }

  def transform(t: (Expr, Expr), env: Env)(placer: (Expr, Expr) => ExprKind): Try[Option[ExprKind]] = {
    for {
      newE1 <- transform(t._1, env)
      newE2 <- transform(t._2, env)
    } yield {
      if (newE1.isEmpty && newE2.isEmpty) {
        None
      } else {
        Some(placer(newE1.getOrElse(t._1), newE2.getOrElse(t._2)))
      }
    }
  }

  def transform(t: (Expr, Expr, Expr), env: Env)(placer: (Expr, Expr, Expr) => ExprKind): Try[Option[ExprKind]] = {
    for {
      newE1 <- transform(t._1, env)
      newE2 <- transform(t._2, env)
      newE3 <- transform(t._3, env)
    } yield {
      if (newE1.isEmpty && newE2.isEmpty && newE3.isEmpty) {
        None
      } else {
        Some(placer(newE1.getOrElse(t._1), newE2.getOrElse(t._2), newE3.getOrElse(t._3)))
      }
    }
  }

  def transform(elems: Vector[Expr], env: Env): Try[Option[Vector[Expr]]] = {
    for {
      newElems <- elems.map(transform(_, env)).sequence
    } yield {
      if (newElems.forall(_.isEmpty)) {
        None
      } else {
        val newElemsFull = elems.zip(newElems).map {
          case (_, Some(e)) => e
          case (oldE, None) => oldE
        }
        Some(newElemsFull)
      }
    }

  }

  def transform(elems: Vector[Expr], env: Env, placer: Vector[Expr] => ExprKind): Try[Option[ExprKind]] =
    transform(elems, env).map(_.map(placer))
}
