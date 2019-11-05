package se.kth.cda.arc.ast

import se.kth.cda.arc.ast.AST.Expr
import se.kth.cda.arc.ast.AST.ExprKind._

object ASTUtils {

  // Helper methods for AST nodes

  implicit class ExprMethods(val self: Expr) extends AnyVal {

    def children(): Vector[Expr] = {
      self.kind match {
        case Application(expr, args) => Vector(expr) ++ args
        case Ascription(expr, _) => Vector(expr)
        case BinOp(_, left, right) => Vector(left, right)
        case Broadcast(expr) => Vector(expr)
        case CUDF(Left(_), args, _) => args
        case CUDF(Right(pointer), args, _) => Vector(pointer) ++ args
        case Cast(_, expr) => Vector(expr)
        case Deserialize(_, expr) => Vector(expr)
        case For(iterator, builder, body) =>
          Vector(iterator.data) ++
            iterator.start.toVector ++
            iterator.end.toVector ++
            iterator.strides.toVector ++
            iterator.shape.toVector ++
            iterator.keyFunc.toVector ++
            Vector(builder, body)
        case Hash(params) => params
        case If(cond, onTrue, onFalse) => Vector(cond, onTrue, onFalse)
        case Iterate(initial, updateFunc) => Vector(initial, updateFunc)
        case Lambda(_, body) => Vector(body)
        case Len(expr) => Vector(expr)
        case Let(_, _, value, body) => Vector(value, body)
        case Lookup(data, key) => Vector(data, key)
        case MakeStruct(elems) => elems
        case MakeVec(elems) => elems
        case Merge(builder, value) => Vector(builder, value)
        case Negate(expr) => Vector(expr)
        case NewBuilder(_, args) => args
        case Not(expr) => Vector(expr)
        case Projection(expr, _) => Vector(expr)
        case Result(expr) => Vector(expr)
        case Select(cond, onTrue, onFalse) => Vector(cond, onTrue, onFalse)
        case Serialize(expr) => Vector(expr)
        case Slice(data, index, size) => Vector(data, index, size)
        case Sort(data, keyFunc) => Vector(data, keyFunc)
        case ToVec(expr) => Vector(expr)
        case UnaryOp(_, expr) => Vector(expr)
        case Zip(params) => params
        case Drain(source, sink) => Vector(source, sink)
        case _: Ident => Vector()
        case _: Literal[_] => Vector()
      }
    }
  }

  implicit class TypeMethods(val self: Type) extends AnyVal {

    import Type.Builder._
    import Type._

    def isBuilderType: Boolean = self match {
      case Struct(types) => types.forall(_.isBuilderType)
      case Function(_, returnTy) => returnTy.isBuilderType
      case _ => self.isInstanceOf[Builder]
    }

    def isValueType: Boolean = self match {
      case Struct(types) => types.forall(_.isValueType)
      case _ => self.isInstanceOf[Scalar] ||
        self.isInstanceOf[Vec] ||
        self.isInstanceOf[Dict] ||
        self.isInstanceOf[Stream]
    }

    def isWeldType: Boolean = self match {
      case Struct(types) => types.forall(_.isWeldType)
      case _ => self.isInstanceOf[Appender] ||
        self.isInstanceOf[Merger] ||
        self.isInstanceOf[VecMerger] ||
        self.isInstanceOf[DictMerger] ||
        self.isInstanceOf[GroupMerger] ||
        self.isInstanceOf[Vec] ||
        self.isInstanceOf[Dict] ||
        self.isInstanceOf[Struct]
    }

    def isArcType: Boolean = self match {
      case Struct(types) => types.forall(_.isArcType)
      case Function(_, returnTy) => returnTy.isArcType
      case _ => self.isInstanceOf[StreamAppender] ||
        self.isInstanceOf[Windower] ||
        self.isInstanceOf[Stream]
    }

    def children: Vector[Type] = self match {
      case _: Scalar => Vector()
      case _: TypeVariable => Vector()
      case builder: Builder => Vector(builder.resultType, builder.mergeType) ++ builder.argTypes
      case Vec(elemTy) => Vector(elemTy)
      case Dict(keyTy, valueTy) => Vector(keyTy, valueTy)
      case Struct(elemTys) => elemTys
      case Simd(elemTy) => Vector(elemTy)
      case Stream(elemTy) => Vector(elemTy)
      case Function(params, returnTy) => params ++ Vector(returnTy)
    }
  }

  //implicit class ExprKindMethods(val self: ExprKind) extends AnyVal {
  //  def toExpr(ty: Type = null): Expr = {
  //    if (ty == null) {
  //      val ty = self match {
  //        case Let(_, _, _, body) => body.ty
  //        case Lambda(params, body) => Type.Function(params.map(_.ty), body.ty)
  //        case Literal.I8(_, _) => Type.I8
  //        case Literal.I16(_, _) => Type.I16
  //        case Literal.I32(_, _) => Type.I32
  //        case Literal.I64(_, _) => Type.I64
  //        case Literal.U8(_, _) => Type.U8
  //        case Literal.U16(_, _) => Type.U16
  //        case Literal.U32(_, _) => Type.U32
  //        case Literal.U64(_, _) => Type.U64
  //        case Literal.F32(_, _) => Type.F32
  //        case Literal.F64(_, _) => Type.F64
  //        case Literal.Bool(_, _) => Type.Bool
  //        case Literal.UnitL(_, _) => Type.Unit
  //        case Literal.StringL(_, _) => Type.String
  //        case Cast(ty, _) => ty
  //        case ToVec(expr) =>
  //        case Ident(symbol) =>
  //        case MakeStruct(elems) =>
  //        case MakeVec(elems) =>
  //        case If(cond, onTrue, onFalse) =>
  //        case Select(cond, onTrue, onFalse) =>
  //        case Iterate(initial, updateFunc) =>
  //        case Broadcast(expr) =>
  //        case Serialize(expr) =>
  //        case Deserialize(ty, expr) =>
  //        case CUDF(reference, args, returnTy) =>
  //        case Zip(params) =>
  //        case Hash(params) =>
  //        case For(iterator, builder, body) =>
  //        case Len(expr) =>
  //        case Lookup(data, key) =>
  //        case Slice(data, index, size) =>
  //        case Sort(data, keyFunc) =>
  //        case Negate(expr) =>
  //        case Not(expr) =>
  //        case UnaryOp(kind, expr) =>
  //        case Merge(builder, value) =>
  //        case Result(expr) =>
  //        case NewBuilder(ty, args) =>
  //        case BinOp(kind, lhs, rhs) =>
  //        case Application(expr, args) =>
  //        case Projection(expr, index) =>
  //        case Ascription(expr, ty) =>
  //      }
  //    }
  //  }
  //}

}
