package se.kth.cda.arc.syntaxtree

object ASTUtils {

  // Helper methods for AST nodes

  implicit class ExprMethods(val self: AST.Expr) extends AnyVal {

    import AST.ExprKind._

    def children(): Vector[AST.Expr] = {
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
        case _: Ident => Vector()
        case _: Literal[_] => Vector()
      }
    }
  }

  implicit class TypeMethods(val self: Type) extends AnyVal {

    import Type._
    import Type.Builder._

    def isBuilderType: Boolean = self match {
      case Struct(types) => types.forall(_.isBuilderType)
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
      case _ => self.isInstanceOf[StreamAppender] ||
        self.isInstanceOf[Windower]
    }

    def children: Vector[Type] = self match {
      case _: ConcreteType => Vector()
      case TypeVariable(_) => Vector()
      case compoundType: CompoundType => compoundType match {
        case builder: Builder => Vector(builder.resultType, builder.mergeType) ++ builder.argTypes
        case Vec(elemTy) => Vector(elemTy)
        case Dict(keyTy, valueTy) => Vector(keyTy, valueTy)
        case Struct(elemTys) => elemTys
        case Simd(elemTy) => Vector(elemTy)
        case Stream(elemTy) => Vector(elemTy)
        case Function(params, returnTy) => params ++ Vector(returnTy)
      }
    }
  }

}
