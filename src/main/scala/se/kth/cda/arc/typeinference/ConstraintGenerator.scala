package se.kth.cda.arc.typeinference

import se.kth.cda.arc.AST._
import se.kth.cda.arc.Types._
import se.kth.cda.arc.Utils.TryVector
import se.kth.cda.arc._

import scala.util.{Failure, Success, Try}

class ConstraintGenerator(val rootExpr: Expr) {
  // top level conjuction
  private var constraints = List.empty[TypeConstraint]

  def generate(): Try[(List[TypeConstraint], Expr)] = {
    discoverConstraints(rootExpr).map(c => (constraints, c))
  }

  def discoverConstraints(e: Expr): Try[Expr] = {
    discoverDown(e, TypingStore.empty()) match {
      case Success(Some(newE)) => Success(newE)
      case Success(None)       => Success(e)
      case Failure(f)          => Failure(f)
    }
  }

  private def discoverDown(e: Expr, env: TypingStore): Try[Option[Expr]] = {
    import ExprKind._

    val newKindOT: Try[Option[ExprKind]] = e.kind match {
      case Let(name, bindingTy, value, body) => {
        constrainEq(e.ty, body.ty)
        constrainEq(bindingTy, value.ty)
        for {
          newValueO <- discoverDown(value, env)
          newBodyO <- discoverDown(body, env + (name -> bindingTy))
        } yield {
          if (newValueO.isEmpty && newBodyO.isEmpty) {
            None
          } else {
            val newValue = newValueO.getOrElse(value)
            val newBody = newBodyO.getOrElse(body)
            Some(Let(name, bindingTy, newValue, newBody))
          }
        }
      }
      case Lambda(params, body) => {
        val paramTypes = params.map(_.ty).toVector
        val bodyTy = body.ty
        val lambdaTy = Types.Function(paramTypes, bodyTy)
        constrainEq(e.ty, lambdaTy)
        discoverDownMap(body, env ++ params.map(p => p.name -> p.ty).toList)(newBody => Lambda(params, newBody))
      }
      case Cast(ty, inner) => {
        constrainEq(e.ty, ty)
        discoverDownMap(inner, env)(newInner => Cast(ty, newInner))
      }
      case ToVec(inner) => {
        val keyTy = Types.unknown
        val valueTy = Types.unknown
        constrainEq(e.ty, Vec(Struct(Vector(keyTy, valueTy))))
        constrainEq(inner.ty, Dict(keyTy, valueTy))
        discoverDownMap(inner, env)(newInner => ToVec(newInner))
      }
      case MakeStruct(elems) => {
        val elemTys = elems.map(e => e.ty).toVector
        constrainEq(e.ty, Struct(elemTys))
        discoverDown(elems, env).map(_.map(newElems => MakeStruct(newElems)))
      }
      case MakeVec(elems) => {
        val elemTy = Types.unknown
        val vecTy = Vec(elemTy)
        constrainEq(e.ty, vecTy)
        val elemTys = elems.map(e => e.ty).toVector
        constrainEq((elemTy +: elemTys): _*)
        discoverDown(elems, env).map(_.map(newElems => MakeVec(newElems)))
      }
      case If(cond: Expr, onTrue: Expr, onFalse: Expr) => {
        constrainEq(cond.ty, Types.Bool)
        constrainEq(e.ty, onTrue.ty, onFalse.ty)
        discoverDown(Vector(cond, onTrue, onFalse), env).map(_.map {
          case Vector(newCond, newOnTrue, newOnFalse) => If(newCond, newOnTrue, newOnFalse)
        })
      }
      case Select(cond: Expr, onTrue: Expr, onFalse: Expr) => {
        constrainEq(cond.ty, Types.Bool)
        constrainEq(e.ty, onTrue.ty, onFalse.ty)
        discoverDown(Vector(cond, onTrue, onFalse), env).map(_.map {
          case Vector(newCond, newOnTrue, newOnFalse) => Select(newCond, newOnTrue, newOnFalse)
        })
      }
      case Iterate(initial: Expr, updateFunc: Expr) => {
        val typeParam = Types.unknown
        val funTy = Function(Vector(typeParam), Struct(Vector(typeParam, Types.Bool)))
        constrainEq(e.ty, initial.ty, typeParam)
        constrainEq(updateFunc.ty, funTy)
        discoverDown(Vector(initial, updateFunc), env).map(_.map {
          case Vector(newInitial, newUpdateFunc) => Iterate(newInitial, newUpdateFunc)
        })
      }
      case Broadcast(inner) => {
        constrainEq(e.ty, Simd(inner.ty))
        constrainScalar(inner.ty)
        discoverDownMap(inner, env)(newInner => Broadcast(newInner))
      }
      case Serialize(inner) => {
        constrainEq(e.ty, Vec(I8))
        discoverDownMap(inner, env)(newInner => Serialize(newInner))
      }
      case Deserialize(ty, inner) => {
        constrainEq(e.ty, ty)
        constrainEq(inner.ty, Vec(I8))
        discoverDownMap(inner, env)(newInner => Deserialize(ty, newInner))
      }
      case CUDF(Left(name), args, returnType) => {
        constrainEq(e.ty, returnType)
        discoverDown(args, env).map(_.map { newArgs =>
          CUDF(Left(name), newArgs, returnType)
        })
      }
      case CUDF(Right(pointer), args, returnType) => {
        val argTypes = args.map(e => e.ty)
        val pointerTy = Function(argTypes, returnType)
        constrainEq(e.ty, returnType)
        constrainEq(pointer.ty, pointerTy)
        for {
          newPointerO <- discoverDown(pointer, env)
          newArgsO <- discoverDown(args, env)
        } yield {
          if (newPointerO.isEmpty && newArgsO.isEmpty) {
            None
          } else {
            val newPointer = newPointerO.getOrElse(pointer)
            val newArgs = newArgsO.getOrElse(args)
            Some(CUDF(Right(newPointer), newArgs, returnType))
          }
        }
      }
      case Zip(params) => {
        val structParamTys = params.map(_ => Types.unknown)
        params.zip(structParamTys).foreach {
          case (p, spTy) => constrainEq(p.ty, Vec(spTy));
        }
        constrainEq(e.ty, Vec(Struct(structParamTys)))
        discoverDown(params, env).map(_.map { newParams =>
          Zip(newParams)
        })
      }
      case Hash(params) => {
        val paramTy = Types.unknown
        params.foreach {
          case p => constrainEq(p.ty, paramTy);
        }
        constrainEq(e.ty, I64)
        discoverDown(params, env).map(_.map { newParams =>
          Hash(newParams)
        })
      }
      case For(iterator, builder, body) => {
        val elemTy = Types.unknown
        constrainNorm(TypeConstraints.IterableKind(iterator.data.ty, elemTy))
        if (iterator.kind == IterKind.RangeIter) {
          constrainEq(elemTy, I64)
        }
        val sest = if (iterator.start.isDefined) { // start end and stride must be defined together
          val start = iterator.start.get
          val end = iterator.end.get
          val stride = iterator.stride.get
          constrainEq(start.ty, end.ty, stride.ty, I64)
          discoverDown(Vector(start, end, stride), env).map(_.map { case Vector(s, e, st) => (s, e, st) })
        } else Success(None)
        val shsts = if (iterator.shape.isDefined) { // shape and strides must be defined together
          val shape = iterator.shape.get
          val strides = iterator.strides.get
          constrainEq(shape.ty, strides.ty, Vec(I64))
          discoverDown(Vector(shape, strides), env).map(_.map { case Vector(sh, sts) => (sh, sts) })
        } else Success(None)
        val keyby = if (iterator.keyFunc.isDefined) {
          val keyFunc = iterator.keyFunc.get
          constrainEq(keyFunc.ty, Function(Vector(elemTy), U64))
          constrainEq(iterator.data.ty, Stream(elemTy))
          discoverDown(keyFunc, env)
        } else Success(None)
        val iterO = for {
          sestO <- sest
          shstsO <- shsts
          keyby0 <- keyby
          dataO <- discoverDown(iterator.data, env)
        } yield {
          if (sestO.isEmpty && shstsO.isEmpty && keyby0.isEmpty && dataO.isEmpty) {
            None
          } else {
            val (start, end, stride) = sestO match {
              case Some((s, e, st)) => (Some(s), Some(e), Some(st))
              case None             => (iterator.start, iterator.end, iterator.stride)
            }
            val (shape, strides) = shstsO match {
              case Some((sh, sts)) => (Some(sh), Some(sts))
              case None            => (iterator.shape, iterator.strides)
            }
            val keyFunc = keyby0.orElse(iterator.keyFunc)
            val data = dataO.getOrElse(iterator.data)
            Some(Iter(iterator.kind, data, start, end, stride, shape, strides, keyFunc))
          }
        }
        val builderTy = Types.unknown
        constrainBuilder(builderTy)
        val funcTy = Function(Vector(builderTy, I64, elemTy), builderTy)
        constrainEq(builder.ty, builderTy)
        constrainEq(body.ty, funcTy)
        // TODO Weld also checks iter kinds once the types are resolved
        for {
          newIterO <- iterO
          newBuilderO <- discoverDown(builder, env)
          newBodyO <- discoverDown(body, env)
        } yield {
          if (newIterO.isEmpty && newBuilderO.isEmpty && newBodyO.isEmpty) {
            None
          } else {
            val newIter = newIterO.getOrElse(iterator)
            val newBuilder = newBuilderO.getOrElse(builder)
            val newBody = newBodyO.getOrElse(body)
            Some(For(newIter, newBuilder, newBody))
          }
        }
      }
      case Len(inner) => {
        val vTy = Types.unknown
        constrainEq(inner.ty, Vec(vTy))
        constrainEq(e.ty, I64)
        discoverDownMap(inner, env)(newInner => Len(newInner))
      }
      case Lookup(data, index) => {
        val resultTy = Types.unknown
        constrainLookup(data.ty, index.ty, resultTy)
        constrainEq(e.ty, resultTy)
        discoverDown(Vector(data, index), env).map(_.map {
          case Vector(newData, newKey) => Lookup(newData, newKey)
        })
      }
      case Slice(data, index, size) => {
        constrainEq(index.ty, size.ty, I64)
        val dataTy = Vec(Types.unknown)
        constrainEq(e.ty, data.ty, dataTy)
        discoverDown(Vector(data, index, size), env).map(_.map {
          case Vector(newData, newIndex, newSize) => Slice(newData, newIndex, newSize)
        })
      }
      case Sort(data, keyFunc) => {
        val elemTy = Types.unknown
        val dataTy = Vec(elemTy)
        constrainEq(e.ty, data.ty, dataTy)
        val funTy = Function(Vector(elemTy), Types.unknown)
        constrainEq(keyFunc.ty, funTy)
        discoverDown(Vector(data, keyFunc), env).map(_.map {
          case Vector(newData, newKeyFunc) => Sort(newData, newKeyFunc)
        })
      }
      case Negate(inner) => {
        constrainEq(e.ty, inner.ty)
        constrainNumeric(inner.ty, signed = true)
        discoverDownMap(inner, env)(newInner => Negate(newInner))
      }
      case Not(inner) => {
        constrainEq(e.ty, Types.Bool)
        constrainEq(inner.ty, Types.Bool)
        discoverDownMap(inner, env)(newInner => Not(newInner))
      }
      case UnaryOp(kind, inner) => {
        constrainFloat(inner.ty)
        constrainEq(e.ty, inner.ty)
        discoverDownMap(inner, env)(newInner => UnaryOp(kind, newInner))
      }
      case Merge(builder, value) => {
        constrainEq(e.ty, builder.ty)
        constrainBuilder(builder.ty, mergeType = value.ty)
        for {
          newBuilderO <- discoverDown(builder, env)
          newValueO <- discoverDown(value, env)
        } yield {
          if (newBuilderO.isEmpty && newValueO.isEmpty) {
            None
          } else {
            val newBuilder = newBuilderO.getOrElse(builder)
            val newValue = newValueO.getOrElse(value)
            Some(Merge(newBuilder, newValue))
          }
        }
      }
      case Result(builder) => {
        constrainBuilder(builder.ty, resultType = e.ty)
        discoverDownMap(builder, env)(newBuilder => Result(newBuilder))
      }
      case NewBuilder(ty, argO) => {
        constrainEq(e.ty, ty)
        argO match {
          case Some(arg) => {
            constrainBuilder(ty, argType = arg.ty)
            discoverDownMap(arg, env)(newArg => NewBuilder(ty, Some(newArg)))
          }
          case None => {
            constrainBuilder(ty)
            Success(None)
          }
        }

      }
      case BinOp(kind, left, right) => {
        import BinOpKind._
        kind match {

          case LessThan | GreaterThan | LEq | GEq => {
            constrainEq(left.ty, right.ty)
            constrainNumeric(left.ty)
            // constrainNumeric(right.ty); // kinda redundant...
            constrainEq(e.ty, Bool)
          }
          case Equals | NEq => {
            constrainEq(left.ty, right.ty)
            constrainEq(e.ty, Bool)
          }
          case LogicalAnd | LogicalOr => {
            constrainEq(e.ty, left.ty, right.ty, Bool)
          }
          case _ => { // numeric
            constrainEq(e.ty, left.ty, right.ty)
            constrainNumeric(left.ty)
          }
        }
        for {
          leftO <- discoverDown(left, env)
          rightO <- discoverDown(right, env)
        } yield {
          if (leftO.isEmpty && rightO.isEmpty) {
            None
          } else {
            val newLeft = leftO.getOrElse(left)
            val newRight = rightO.getOrElse(right)
            Some(BinOp(kind, newLeft, newRight))
          }
        }
      }
      case Application(funcExpr, args) => {
        val argTypes = args.map(e => e.ty)
        val returnTy = Types.unknown
        val funcTy = Function(argTypes, returnTy)
        constrainEq(e.ty, returnTy)
        constrainEq(funcExpr.ty, funcTy)
        for {
          newFuncO <- discoverDown(funcExpr, env)
          newArgsO <- discoverDown(args, env)
        } yield {
          if (newFuncO.isEmpty && newArgsO.isEmpty) {
            None
          } else {
            val newFunc = newFuncO.getOrElse(funcExpr)
            val newArgs = newArgsO.getOrElse(args)
            Some(Application(newFunc, newArgs))
          }
        }
      }
      case Projection(structExpr, index) => {
        val fieldTy = Types.unknown
        constrainNorm(TypeConstraints.ProjectableKind(structExpr.ty, fieldTy, index))
        constrainEq(e.ty, fieldTy)
        discoverDownMap(structExpr, env)(newStructExpr => Projection(newStructExpr, index))
      }
      case Ascription(inner, ty) => {
        (inner.kind, ty) match {
          case (l: Literal[_], s: Scalar) => {
            constrainEq(e.ty, ty)
            convertLiteral(l, s)
          }
          case _ => {
            constrainEq(inner.ty, ty)
            constrainEq(e.ty, ty)
            discoverDownMap(inner, env)(newInner => Ascription(newInner, ty))
          }
        }

      }
      case Ident(s) => {
        env.lookup(s) match {
          case Some(ty) => {
            constrainEq(e.ty, ty)
            Success(None)
          }
          case None => Failure(new TypingException(s"Identifier $s wasn't bound in environment!"))
        }
      }
      case _ => Success(None)
    }
    newKindOT.map(_.map(newKind => Expr(newKind, e.ty, e.ctx)))
  }

  private def convertLiteral[T](l: ExprKind.Literal[T], ty: Scalar): Try[Option[ExprKind]] = {
    import ExprKind.Literal
    val res: Try[ExprKind] = (l, ty) match {
      case (Literal.I8(raw, value), U8)     => Literal.tryU8(raw + ":u8", value)
      case (Literal.I8(raw, value), U16)    => Literal.tryU16(raw + ":u16", value)
      case (Literal.I8(raw, value), U32)    => Literal.tryU32(raw + ":u32", value)
      case (Literal.I8(raw, value), U64)    => Literal.tryU64(raw + ":u64", value)
      case (Literal.I8(_, _), I8)           => Success(l)
      case (Literal.I16(raw, value), U8)    => Literal.tryU8(raw + ":u8", value)
      case (Literal.I16(raw, value), U16)   => Literal.tryU16(raw + ":u16", value)
      case (Literal.I16(raw, value), U32)   => Literal.tryU32(raw + ":u32", value)
      case (Literal.I16(raw, value), U64)   => Literal.tryU64(raw + ":u64", value)
      case (Literal.I16(_, _), I16)         => Success(l)
      case (Literal.I32(raw, value), U8)    => Literal.tryU8(raw + ":u8", value)
      case (Literal.I32(raw, value), U16)   => Literal.tryU16(raw + ":u16", value)
      case (Literal.I32(raw, value), U32)   => Literal.tryU32(raw + ":u32", value)
      case (Literal.I32(raw, value), U64)   => Literal.tryU64(raw + ":u64", value)
      case (Literal.I32(_, _), I32)         => Success(l)
      case (Literal.I64(raw, value), U8)    => Try(value.toInt).flatMap(v => Literal.tryU8(raw + ":u8", v))
      case (Literal.I64(raw, value), U16)   => Try(value.toInt).flatMap(v => Literal.tryU16(raw + ":u16", v))
      case (Literal.I64(raw, value), U32)   => Literal.tryU32(raw + ":u32", value)
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
    res.map(l => Some(l))
  }

  private def discoverDownMap[T](e: Expr, env: TypingStore)(placer: (Expr) => T): Try[Option[T]] = {
    discoverDown(e, env).map(_.map(placer))
  }

  private def discoverDown(elems: Vector[Expr], env: TypingStore): Try[Option[Vector[Expr]]] = {
    for {
      newElems <- elems.map(e => discoverDown(e, env)).sequence
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
      mergeType: Type = Types.unknown,
      resultType: Type = Types.unknown,
      argType: Type = Types.unknown): Unit = {
    constrainNorm(TypeConstraints.BuilderKind(t, mergeType, resultType, argType))
  }
  private def constrainLookup(t: Type, indexType: Type, resultType: Type): Unit = {
    constrainNorm(TypeConstraints.LookupKind(t, indexType, resultType))
  }
  private def constrainNorm(c: TypeConstraint): Unit = {
    c.normalise() match {
      case Some(norm) if norm != TypeConstraints.Tautology => {
        //println(s"Introduced ${c.describe} normalised to ${norm.describe}");
        constraints ::= norm
      }
      case None if c != TypeConstraints.Tautology => constraints ::= c
      case _                                      => // ignore
    }
  }
}
