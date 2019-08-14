package se.kth.cda.arc.syntaxtree.typer

import se.kth.cda.arc.syntaxtree.AST._
import se.kth.cda.arc.syntaxtree.Type._
import se.kth.cda.arc.Utils.TryVector
import se.kth.cda.arc.syntaxtree.Type

import scala.util.{Failure, Success, Try}

class ConstraintGenerator(val rootExpr: Expr) {
  // top level conjuction
  private var constraints = List.empty[TypeConstraint]

  def generate(): Try[(List[TypeConstraint], Expr)] = {
    discoverConstraints(rootExpr).map((constraints, _))
  }

  def discoverConstraints(e: Expr): Try[Expr] = {
    discoverDown(e, TypingStore.empty()) match {
      case Success(Some(newE)) => Success(newE)
      case Success(None)       => Success(e)
      case Failure(f)          => Failure(f)
    }
  }

  private def discoverDown(expr: Expr, env: TypingStore): Try[Option[Expr]] = {
    import ExprKind._

    val newKindOT: Try[Option[ExprKind]] = expr.kind match {
      case Let(name, bindingTy, value, body) =>
        constrainEq(expr.ty, body.ty)
        constrainEq(bindingTy, value.ty)
        for {
          newValue <- discoverDown(value, env)
          newBody <- discoverDown(body, env + (name -> bindingTy))
        } yield {
          if (newValue.isEmpty && newBody.isEmpty) {
            None
          } else {
            Some(Let(name, bindingTy, newValue.getOrElse(value), newBody.getOrElse(body)))
          }
        }
      case Lambda(params, body) =>
        constrainEq(expr.ty, Type.Function(params.map(_.ty), body.ty))
        discoverDownMap(body, env ++ params.map(p => p.symbol -> p.ty).toList)(Lambda(params, _))
      case Cast(ty, inner) =>
        constrainEq(expr.ty, ty)
        discoverDownMap(inner, env)(Cast(ty, _))
      case ToVec(inner) =>
        val keyTy = Type.unknown
        val valueTy = Type.unknown
        constrainEq(expr.ty, Vec(Struct(Vector(keyTy, valueTy))))
        constrainEq(inner.ty, Dict(keyTy, valueTy))
        discoverDownMap(inner, env)(newInner => ToVec(newInner))
      case MakeStruct(elems) =>
        constrainEq(expr.ty, Struct(elems.map(_.ty)))
        discoverDown(elems, env).map(_.map(MakeStruct))
      case MakeVec(elems) =>
        val elemTy = Type.unknown
        val vecTy = Vec(elemTy)
        val elemTys = elems.map(_.ty)
        constrainEq(expr.ty, vecTy)
        constrainEq(elemTy +: elemTys: _*)
        discoverDown(elems, env).map(_.map(MakeVec))
      case If(cond: Expr, onTrue: Expr, onFalse: Expr) =>
        constrainEq(cond.ty, Type.Bool)
        constrainEq(expr.ty, onTrue.ty, onFalse.ty)
        discoverDown(Vector(cond, onTrue, onFalse), env).map(_.map {
          case Vector(newCond, newOnTrue, newOnFalse) => If(newCond, newOnTrue, newOnFalse)
        })
      case Select(cond: Expr, onTrue: Expr, onFalse: Expr) =>
        constrainEq(cond.ty, Type.Bool)
        constrainEq(expr.ty, onTrue.ty, onFalse.ty)
        discoverDown(Vector(cond, onTrue, onFalse), env).map(_.map {
          case Vector(newCond, newOnTrue, newOnFalse) => Select(newCond, newOnTrue, newOnFalse)
        })
      case Iterate(initial: Expr, updateFunc: Expr) =>
        val typeParam = Type.unknown
        val funTy = Function(Vector(typeParam), Struct(Vector(typeParam, Type.Bool)))
        constrainEq(expr.ty, initial.ty, typeParam)
        constrainEq(updateFunc.ty, funTy)
        discoverDown(Vector(initial, updateFunc), env).map(_.map {
          case Vector(newInitial, newUpdateFunc) => Iterate(newInitial, newUpdateFunc)
        })
      case Broadcast(inner) =>
        constrainEq(expr.ty, Simd(inner.ty))
        constrainScalar(inner.ty)
        discoverDownMap(inner, env)(Broadcast)
      case Serialize(inner) =>
        constrainEq(expr.ty, Vec(U8))
        discoverDownMap(inner, env)(Serialize)
      case Deserialize(ty, inner) =>
        constrainEq(expr.ty, ty)
        constrainEq(inner.ty, Vec(U8))
        discoverDownMap(inner, env)(Deserialize(ty, _))
      case CUDF(Left(name), args, returnType) =>
        constrainEq(expr.ty, returnType)
        // TODO: Add constraint for name to be function of certain type
        discoverDown(args, env).map(_.map { newArgs =>
          CUDF(Left(name), newArgs, returnType)
        })
      case CUDF(Right(pointer), args, returnType) =>
        val argTypes = args.map(_.ty)
        val pointerTy = Function(argTypes, returnType)
        constrainEq(expr.ty, returnType)
        constrainEq(pointer.ty, pointerTy)
        for {
          newPointer <- discoverDown(pointer, env)
          newArgs <- discoverDown(args, env)
        } yield {
          if (newPointer.isEmpty && newArgs.isEmpty) {
            None
          } else {
            Some(CUDF(Right(newPointer.getOrElse(pointer)), newArgs.getOrElse(args), returnType))
          }
        }
      case Zip(params) =>
        val elem_tys = params.map(_ => Type.unknown)
        params.zip(elem_tys).foreach {
          case (param, elem_ty) => constrainEq(param.ty, Vec(elem_ty));
        }
        constrainEq(expr.ty, Vec(Struct(elem_tys)))
        discoverDown(params, env).map(_.map { newParams =>
          Zip(newParams)
        })
      case Hash(params) =>
        params.foreach(param => constrainEq(param.ty, Type.unknown))
        constrainEq(expr.ty, I64)
        discoverDown(params, env).map(_.map(Hash))
      case For(iter, builder, body) =>
        val elemTy = Type.unknown
        constrainNorm(TypeConstraints.IterableKind(iter.data.ty, elemTy))
        if (iter.kind == IterKind.RangeIter) {
          constrainEq(elemTy, I64)
        }
        val newIter = for {
          sest <- if (iter.start.isDefined) { // start end and stride must be defined together
            val start = iter.start.get
            val end = iter.end.get
            val stride = iter.stride.get
            constrainEq(start.ty, end.ty, stride.ty, I64)
            discoverDown(Vector(start, end, stride), env).map(_.map { case Vector(s, e, st) => (s, e, st) })
          } else {
            Success(None)
          }
          shsts <- if (iter.shape.isDefined) { // shape and strides must be defined together
            val shape = iter.shape.get
            val strides = iter.strides.get
            constrainEq(shape.ty, strides.ty, Vec(I64))
            discoverDown(Vector(shape, strides), env).map(_.map { case Vector(sh, sts) => (sh, sts) })
          } else {
            Success(None)
          }
          keyFunc <- if (iter.keyFunc.isDefined) { // data and keyFunc must be defined together
            constrainEq(iter.data.ty, Stream(elemTy))
            val keyFunc = iter.keyFunc.get
            constrainEq(keyFunc.ty, Function(Vector(elemTy), Vec(I32)))
            discoverDown(keyFunc, env)
          } else {
            Success(None)
          }
          data <- discoverDown(iter.data, env)
        } yield {
          if (sest.isEmpty && shsts.isEmpty && keyFunc.isEmpty && data.isEmpty) {
            None
          } else {
            val (newStart, newEnd, newStride) = sest match {
              case Some((s, e, st)) => (Some(s), Some(e), Some(st))
              case None             => (iter.start, iter.end, iter.stride)
            }
            val (newShape, newStrides) = shsts match {
              case Some((sh, sts)) => (Some(sh), Some(sts))
              case None            => (iter.shape, iter.strides)
            }
            val newKeyFunc = keyFunc.orElse(iter.keyFunc)
            val newData = data.getOrElse(iter.data)
            Some(Iter(iter.kind, newData, newStart, newEnd, newStride, newShape, newStrides, newKeyFunc))
          }
        }
        val builderTy = Type.unknown
        val funcTy = Function(Vector(builderTy, I64, elemTy), builderTy)
        constrainBuilder(builderTy)
        constrainEq(builder.ty, builderTy)
        constrainEq(body.ty, funcTy)
        // TODO Weld also checks iter kinds once the types are resolved
        for {
          newIter <- newIter
          newBuilder <- discoverDown(builder, env)
          newBody <- discoverDown(body, env)
        } yield {
          if (newIter.isEmpty && newBuilder.isEmpty && newBody.isEmpty) {
            None
          } else {
            Some(For(newIter.getOrElse(iter), newBuilder.getOrElse(builder), newBody.getOrElse(body)))
          }
        }
      case Len(inner) =>
        constrainEq(inner.ty, Vec(Type.unknown))
        constrainEq(expr.ty, I64)
        discoverDownMap(inner, env)(Len)
      case Lookup(data, index) =>
        val resultTy = Type.unknown
        constrainLookup(data.ty, index.ty, resultTy)
        constrainEq(expr.ty, resultTy)
        discoverDown(Vector(data, index), env).map(_.map {
          case Vector(newData, newKey) => Lookup(newData, newKey)
        })
      case Slice(data, index, size) =>
        constrainEq(index.ty, size.ty, I64)
        constrainEq(expr.ty, data.ty, Vec(Type.unknown))
        discoverDown(Vector(data, index, size), env).map(_.map {
          case Vector(newData, newIndex, newSize) => Slice(newData, newIndex, newSize)
        })
      // TODO: Drain might be more hassle than what it is worth
      //case Drain(source, sink) =>
      //  val elemTy = Type.unknown
      //  constrainEq(Stream(elemTy), source.ty)
      //  constrainEq(expr.ty, sink.ty)
      //  constrainBuilder(sink.ty, mergeType = Stream(sink.ty))
      //  for {
      //    newBuilder <- discoverDown(builder, env)
      //    newValue <- discoverDown(value, env)
      //  } yield {
      //    if (newBuilder.isEmpty && newValue.isEmpty) {
      //      None
      //    } else {
      //      Some(Merge(newBuilder.getOrElse(builder), newValue.getOrElse(value)))
      //    }
      //  }
      case Sort(data, keyFunc) =>
        val elemTy = Type.unknown
        constrainEq(expr.ty, data.ty, Vec(elemTy))
        val funTy = Function(Vector(elemTy, elemTy), Type.I32)
        constrainEq(keyFunc.ty, funTy)
        discoverDown(Vector(data, keyFunc), env).map(_.map {
          case Vector(newData, newKeyFunc) => Sort(newData, newKeyFunc)
        })
      case Negate(inner) =>
        constrainEq(expr.ty, inner.ty)
        constrainNumeric(inner.ty, signed = true)
        discoverDownMap(inner, env)(Negate)
      case Not(inner) =>
        constrainEq(expr.ty, Type.Bool)
        constrainEq(inner.ty, Type.Bool)
        discoverDownMap(inner, env)(Not)
      case UnaryOp(kind, inner) =>
        constrainFloat(inner.ty)
        constrainEq(expr.ty, inner.ty)
        discoverDownMap(inner, env)(UnaryOp(kind, _))
      case Merge(builder, value) =>
        constrainEq(expr.ty, builder.ty)
        constrainBuilder(builder.ty, mergeType = value.ty)
        for {
          newBuilder <- discoverDown(builder, env)
          newValue <- discoverDown(value, env)
        } yield {
          if (newBuilder.isEmpty && newValue.isEmpty) {
            None
          } else {
            Some(Merge(newBuilder.getOrElse(builder), newValue.getOrElse(value)))
          }
        }
      case Result(builder) =>
        constrainBuilder(builder.ty, resultType = expr.ty)
        discoverDownMap(builder, env)(Result)
      case NewBuilder(ty, args) =>
        constrainEq(expr.ty, ty)
        constrainBuilder(ty, argTypes = args.map(_.ty))
        for {
          newArgs <- discoverDown(args, env)
        } yield {
          if (newArgs.isEmpty) {
            None
          } else {
            Some(NewBuilder(ty, newArgs.getOrElse(args)))
          }
        }
      case BinOp(kind, left, right) =>
        import BinOpKind._
        kind match {
          case Or | And =>
            constrainEq(expr.ty, left.ty, right.ty, Bool)
          case Eq | NEq =>
            constrainEq(left.ty, right.ty)
            constrainEq(expr.ty, Bool)
          case Lt | Gt | LEq | GEq =>
            constrainEq(left.ty, right.ty)
            constrainNumeric(left.ty)
            constrainEq(expr.ty, Bool)
          case Pow =>
            constrainEq(expr.ty, left.ty)
            constrainNumeric(left.ty)
            constrainNumeric(right.ty)
          case _ => // numeric
            constrainEq(expr.ty, left.ty, right.ty)
            constrainNumeric(left.ty)
        }
        for {
          newLeft <- discoverDown(left, env)
          newRight <- discoverDown(right, env)
        } yield {
          if (newLeft.isEmpty && newRight.isEmpty) {
            None
          } else {
            Some(BinOp(kind, newLeft.getOrElse(left), newRight.getOrElse(right)))
          }
        }
      case Application(funcExpr, args) =>
        val returnTy = Type.unknown
        constrainEq(expr.ty, returnTy)
        constrainEq(funcExpr.ty, Function(args.map(_.ty), returnTy))
        for {
          newFunc <- discoverDown(funcExpr, env)
          newArgs <- discoverDown(args, env)
        } yield {
          if (newFunc.isEmpty && newArgs.isEmpty) {
            None
          } else {
            Some(Application(newFunc.getOrElse(funcExpr), newArgs.getOrElse(args)))
          }
        }
      case Projection(structExpr, index) =>
        val fieldTy = Type.unknown
        constrainNorm(TypeConstraints.ProjectableKind(structExpr.ty, fieldTy, index))
        constrainEq(expr.ty, fieldTy)
        discoverDownMap(structExpr, env)(Projection(_, index))
      case Ascription(inner, ty) =>
        (inner.kind, ty) match {
          case (l: Literal[_], s: Scalar) =>
            constrainEq(expr.ty, ty)
            convertLiteral(l, s)
          case _ =>
            constrainEq(inner.ty, ty)
            constrainEq(expr.ty, ty)
            discoverDownMap(inner, env)(Ascription(_, ty))
        }
      case Ident(s) =>
        env.lookup(s) match {
          case Some(ty) =>
            constrainEq(expr.ty, ty)
            Success(None)
          case None => Failure(new TypingException(s"Identifier $s wasn't bound in environment!"))
        }
      case _ => Success(None)
    }
    newKindOT.map(_.map(Expr(_, expr.ty, expr.ctx)))
  }

  private def convertLiteral[T](l: ExprKind.Literal[T], ty: Scalar): Try[Option[ExprKind]] = {
    import ExprKind.Literal
    val res: Try[ExprKind] = (l, ty) match {
      case (Literal.I8(raw, value), U8)     => Literal.tryU8(raw + ":u8", value)
      case (Literal.I8(raw, value), U16)    => Literal.tryU16(raw + ":u16", value)
      case (Literal.I8(raw, value), U32)    => Literal.tryU32(raw + ":u32", value.toLong)
      case (Literal.I8(raw, value), U64)    => Literal.tryU64(raw + ":u64", value)
      case (Literal.I8(_, _), I8)           => Success(l)
      case (Literal.I16(raw, value), U8)    => Literal.tryU8(raw + ":u8", value)
      case (Literal.I16(raw, value), U16)   => Literal.tryU16(raw + ":u16", value)
      case (Literal.I16(raw, value), U32)   => Literal.tryU32(raw + ":u32", value.toLong)
      case (Literal.I16(raw, value), U64)   => Literal.tryU64(raw + ":u64", value)
      case (Literal.I16(_, _), I16)         => Success(l)
      case (Literal.I32(raw, value), U8)    => Literal.tryU8(raw + ":u8", value)
      case (Literal.I32(raw, value), U16)   => Literal.tryU16(raw + ":u16", value)
      case (Literal.I32(raw, value), U32)   => Literal.tryU32(raw + ":u32", value.toLong)
      case (Literal.I32(raw, value), U64)   => Literal.tryU64(raw + ":u64", value)
      case (Literal.I32(_, _), I32)         => Success(l)
      case (Literal.I64(raw, value), U8)    => Try(value.toInt).flatMap(v => Literal.tryU8(raw + ":u8", v))
      case (Literal.I64(raw, value), U16)   => Try(value.toInt).flatMap(v => Literal.tryU16(raw + ":u16", v))
      case (Literal.I64(raw, value), U32)   => Literal.tryU32(raw + ":u32", value.toLong)
      case (Literal.I64(raw, value), U64)   => Literal.tryU64(raw + ":u64", value)
      case (Literal.I64(_, _), I64)         => Success(l)
      case (Literal.U8(_, _), U8)           => Success(l)
      case (Literal.U16(_, _), U16)         => Success(l)
      case (Literal.U32(_, _), U32)         => Success(l)
      case (Literal.U64(_, _), U64)         => Success(l)
      case (Literal.F32(_, _), F32)         => Success(l)
      case (Literal.F64(_, _), F64)         => Success(l)
      case (Literal.Bool(_, _), Bool)       => Success(l)
      case (Literal.UnitL(_, _), UnitT)     => Success(l)
      case (Literal.StringL(_, _), StringT) => Success(l)
      case _ =>
        Failure(
          new TypingException(
            s"Invalid literal ascription ${l.raw}:${ty.render}! Maybe a cast is what you are looking for?"))
    }
    res.map(Some(_))
  }

  private def discoverDownMap[T](e: Expr, env: TypingStore)(placer: Expr => T): Try[Option[T]] = {
    discoverDown(e, env).map(_.map(placer))
  }

  private def discoverDown(elems: Vector[Expr], env: TypingStore): Try[Option[Vector[Expr]]] = {
    for {
      newElems <- elems.map(discoverDown(_, env)).sequence
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

  private def constrainEq(t: Type*): Unit = {
    constrainNorm(TypeConstraints.MultiEquality(t.toList))
  }
  private def constrainNumeric(t: Type, signed: Boolean = false): Unit = {
    constrainNorm(TypeConstraints.IsNumeric(t, signed))
  }
  private def constrainScalar(t: Type): Unit = {
    constrainNorm(TypeConstraints.IsScalar(t))
  }
  private def constrainFloat(t: Type): Unit = {
    constrainNorm(TypeConstraints.IsFloat(t))
  }
  private def constrainBuilder(
                                t: Type,
                                mergeType: Type = Type.unknown,
                                resultType: Type = Type.unknown,
                                argTypes: Vector[Type] = Vector(Type.unknown)): Unit = {
    constrainNorm(TypeConstraints.BuilderKind(t, mergeType, resultType, argTypes))
  }
  private def constrainLookup(t: Type, indexType: Type, resultType: Type): Unit = {
    constrainNorm(TypeConstraints.LookupKind(t, indexType, resultType))
  }
  private def constrainNorm(c: TypeConstraint): Unit = {
    c.normalise() match {
      case Some(norm) if norm != TypeConstraints.Tautology =>
        //println(s"Introduced ${c.describe} normalised to ${norm.describe}");
        constraints ::= norm
      case None if c != TypeConstraints.Tautology => constraints ::= c
      case _                                      => // ignore
    }
  }
}
