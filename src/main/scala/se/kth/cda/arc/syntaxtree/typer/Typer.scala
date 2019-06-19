package se.kth.cda.arc.syntaxtree.typer

import se.kth.cda.arc.syntaxtree.AST._
import se.kth.cda.arc.Utils.{OptionTry, TryVector}
import se.kth.cda.arc._
import se.kth.cda.arc.syntaxtree.BuilderType

import scala.util.{Success, Try}

object Typer {

  def applyTypes(expr: Expr, solution: ConstraintSolver.Result): Try[Expr] = {
    //println(s"Solution:\n${solution.describe}");
    applyTypes(expr, solution.typeSubstitutions).flatMap(appliedExpr =>
      solution match {
        case _: ConstraintSolver.Solution              => Success(appliedExpr)
        case partial: ConstraintSolver.PartialSolution => partial.describeUnresolvedConstraints(appliedExpr)
    })
  }

  def applyTypes(expr: Expr, substitute: Substituter): Try[Expr] = {
    substituteTypes(expr, substitute)
  }

  private def substituteTypes(expr: Expr, substitute: Substituter): Try[Expr] = {
    for {
      newTy <- substitute(expr.ty)
      newKind <- substituteTypes(expr.kind, substitute)
    } yield Expr(newKind, newTy, expr.ctx)

  }

  private def substituteTypes[T](elems: Vector[Expr], substitute: Substituter)(placer: Vector[Expr] => T): Try[T] = {
    for {
      newElems <- elems.map(substituteTypes(_, substitute)).sequence
    } yield placer(newElems)
  }

  private def substituteTypes(exprKind: ExprKind, substitute: Substituter): Try[ExprKind] = {
    import ExprKind._

    exprKind match {
      case Let(name, bindingTy, value, body) =>
        for {
          newBindingTy <- substitute(bindingTy)
          newValue <- substituteTypes(value, substitute)
          newBody <- substituteTypes(body, substitute)
        } yield Let(name, newBindingTy, newValue, newBody)
      case Lambda(params, body) =>
        for {
          newParams <- params.map(p => substitute(p.ty).map(Parameter(p.symbol, _))).sequence
          newBody <- substituteTypes(body, substitute)
        } yield Lambda(newParams, newBody)
      case Cast(ty, expr) =>
        for { // ignore ty since it must be a concrete scalar
          newExpr <- substituteTypes(expr, substitute)
        } yield Cast(ty, newExpr)
      case ToVec(expr) =>
        for {
          newExpr <- substituteTypes(expr, substitute)
        } yield ToVec(newExpr)
      case i: Ident          => Success(i)
      case MakeStruct(elems) => substituteTypes(elems, substitute)(MakeStruct)
      case MakeVec(elems)    => substituteTypes(elems, substitute)(MakeVec)
      case If(cond, onTrue, onFalse) =>
        substituteTypes(Vector(cond, onTrue, onFalse), substitute) {
          case Vector(newCond, newOnTrue, newOnFalse) => If(newCond, newOnTrue, newOnFalse)
        }
      case Select(cond, onTrue, onFalse) =>
        substituteTypes(Vector(cond, onTrue, onFalse), substitute) {
          case Vector(newCond, newOnTrue, newOnFalse) => Select(newCond, newOnTrue, newOnFalse)
        }
      case Iterate(initial, updateFunc) =>
        for {
          newInitial <- substituteTypes(initial, substitute)
          newUpdateFunc <- substituteTypes(updateFunc, substitute)
        } yield Iterate(newInitial, newUpdateFunc)
      case Broadcast(expr: Expr) =>
        for {
          newExpr <- substituteTypes(expr, substitute)
        } yield Broadcast(newExpr)
      case Serialize(expr) =>
        for {
          newExpr <- substituteTypes(expr, substitute)
        } yield Serialize(newExpr)
      case Deserialize(ty, expr) =>
        for {
          newTy <- substitute(ty)
          newExpr <- substituteTypes(expr, substitute)
        } yield Deserialize(newTy, newExpr)
      case CUDF(Left(name), args, returnType) =>
        for {
          newArgs <- substituteTypes(args, substitute)(identity)
          newReturnType <- substitute(returnType)
        } yield CUDF(Left(name), newArgs, newReturnType)
      case CUDF(Right(pointer), args, returnType) =>
        for {
          newPointer <- substituteTypes(pointer, substitute)
          newArgs <- substituteTypes(args, substitute)(identity)
          newReturnType <- substitute(returnType)
        } yield CUDF(Right(newPointer), newArgs, newReturnType)
      case Zip(elems)  => substituteTypes(elems, substitute)(Zip)
      case Hash(elems) => substituteTypes(elems, substitute)(Hash)
      case For(iterator, builder, body) =>
        for {
          newIterator <- substituteTypes(iterator, substitute)
          newBuilder <- substituteTypes(builder, substitute)
          newBody <- substituteTypes(body, substitute)
        } yield For(newIterator, newBuilder, newBody)
      case Len(expr) =>
        for {
          newExpr <- substituteTypes(expr, substitute)
        } yield Len(newExpr)
      case Lookup(data, key) =>
        for {
          newData <- substituteTypes(data, substitute)
          newKey <- substituteTypes(key, substitute)
        } yield Lookup(newData, newKey)
      case Slice(data, index, size) =>
        substituteTypes(Vector(data, index, size), substitute) {
          case Vector(newData, newIndex, newSize) => Slice(newData, newIndex, newSize)
        }
      case Sort(data, keyFunc) =>
        for {
          newData <- substituteTypes(data, substitute)
          newKeyFunc <- substituteTypes(keyFunc, substitute)
        } yield Sort(newData, newKeyFunc)
      case Negate(inner) =>
        for {
          newInner <- substituteTypes(inner, substitute)
        } yield Negate(newInner)
      case Not(inner) =>
        for {
          newInner <- substituteTypes(inner, substitute)
        } yield Not(newInner)
      case UnaryOp(kind, expr) =>
        for {
          newExpr <- substituteTypes(expr, substitute)
        } yield UnaryOp(kind, newExpr)
      case Merge(builder, value) =>
        for {
          newBuilder <- substituteTypes(builder, substitute)
          newValue <- substituteTypes(value, substitute)
        } yield Merge(newBuilder, newValue)
      case Result(expr) =>
        for {
          newExpr <- substituteTypes(expr, substitute)
        } yield Result(newExpr)
      case NewBuilder(ty, args) =>
        for {
          newTy <- substitute(ty)
          newArgs <- args.map(substituteTypes(_, substitute)).sequence
        } yield NewBuilder(newTy.asInstanceOf[BuilderType], newArgs)
      case BinOp(kind, left, right) =>
        for {
          newLeft <- substituteTypes(left, substitute)
          newRight <- substituteTypes(right, substitute)
        } yield BinOp(kind, newLeft, newRight)
      case Application(expr, params) =>
        for {
          newFunc <- substituteTypes(expr, substitute)
          newParams <- substituteTypes(params, substitute)(identity)
        } yield Application(newFunc, newParams)
      case Projection(expr: Expr, index: Int) =>
        for {
          newStruct <- substituteTypes(expr, substitute)
        } yield Projection(newStruct, index)
      case Ascription(inner, ty) =>
        for {
          newInner <- substituteTypes(inner, substitute)
        } yield Ascription(newInner, ty)
      case l: Literal[_] => Success(l)
    }
  }

  private def substituteTypes(iter: Iter, substitute: Substituter): Try[Iter] = {
    for {
      newData <- substituteTypes(iter.data, substitute)
      newStart <- iter.start.map(substituteTypes(_, substitute)).invert
      newEnd <- iter.end.map(substituteTypes(_, substitute)).invert
      newStride <- iter.stride.map(substituteTypes(_, substitute)).invert
      newStrides <- iter.strides.map(substituteTypes(_, substitute)).invert
      newShape <- iter.shape.map(substituteTypes(_, substitute)).invert
      newKeyFunc <- iter.keyFunc.map(substituteTypes(_, substitute)).invert
    } yield Iter(iter.kind, newData, newStart, newEnd, newStride, newStrides, newShape, newKeyFunc)
  }
}
