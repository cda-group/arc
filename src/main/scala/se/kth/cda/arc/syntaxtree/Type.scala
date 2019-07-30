package se.kth.cda.arc.syntaxtree

sealed trait Type extends Equals {
  def render: String

  def isComplete: Boolean
}
sealed trait ConcreteType extends Type {
  override def isComplete: Boolean = true
}
sealed trait CompoundType extends Type

sealed trait Builder extends CompoundType {
  def annotations: Option[AST.Annotations]

  def resultType: Type

  def mergeType: Type

  def argTypes: Vector[Type]
}

object Type {
  private var variableCounter: Int = 0

  def unknown: TypeVariable = {
    val t = TypeVariable(variableCounter)
    variableCounter += 1
    t
  }

  final case class TypeVariable(id: Int) extends Type {
    override def render: String = s"?$id"

    override def isComplete: Boolean = false
  }

  sealed trait Scalar extends ConcreteType {
    //override def isAssignableTo(t: Type): Boolean = this == t // no subtyping among scalars
    //override def isAssignableFrom(t: Type): Boolean = this == t // no subtyping among scalars
  }
  case object I8 extends Scalar {
    override def render: String = "i8"
  }
  case object I16 extends Scalar {
    override def render: String = "i16"
  }
  case object I32 extends Scalar {
    override def render: String = "i32"
  }
  case object I64 extends Scalar {
    override def render: String = "i64"
  }
  case object U8 extends Scalar {
    override def render: String = "u8"
  }
  case object U16 extends Scalar {
    override def render: String = "u16"
  }
  case object U32 extends Scalar {
    override def render: String = "u32"
  }
  case object U64 extends Scalar {
    override def render: String = "u64"
  }
  case object F32 extends Scalar {
    override def render: String = "f32"
  }
  case object F64 extends Scalar {
    override def render: String = "f64"
  }
  case object Bool extends Scalar {
    override def render: String = "bool"
  }
  case object UnitT extends Scalar {
    override def render: String = "unit"
  }
  case object StringT extends Scalar {
    override def render: String = "string"
  }
  final case class Vec(elemTy: Type) extends CompoundType {
    override def render: String = s"vec[${elemTy.render}]"

    override def isComplete: Boolean = elemTy.isComplete
  }
  final case class Dict(keyTy: Type, valueTy: Type) extends CompoundType {
    override def render: String = s"dict[${keyTy.render}, ${valueTy.render}]"

    override def isComplete: Boolean = keyTy.isComplete && valueTy.isComplete
  }
  final case class Struct(elemTys: Vector[Type]) extends CompoundType {
    override def render: String = elemTys.map(_.render).mkString("{", ",", "}")

    override def isComplete: Boolean = elemTys.forall(_.isComplete)
  }
  final case class Simd(elemTy: Type) extends CompoundType {
    override def render: String = s"simd[${elemTy.render}]"

    override def isComplete: Boolean = elemTy.isComplete
  }
  final case class Stream(elemTy: Type) extends CompoundType {
    override def render: String = s"stream[${elemTy.render}]"

    override def isComplete: Boolean = elemTy.isComplete
  }
  final case class Function(params: Vector[Type], returnTy: Type) extends CompoundType {
    override def render: String = s"|${params.map(_.render).mkString(",")}|(${returnTy.render})"

    override def isComplete: Boolean = params.forall(_.isComplete) && returnTy.isComplete
  }

  object Builder {
    final case class Appender(elemTy: Type, annotations: Option[AST.Annotations]) extends Builder {
      override def render: String = s"appender[${elemTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = Vec(elemTy)

      override def mergeType: Type = elemTy

      override def argTypes: Vector[Type] = Vector(UnitT) // TODO optionally allows an index?
    }
    final case class StreamAppender(elemTy: Type, annotations: Option[AST.Annotations]) extends Builder {
      override def render: String = s"streamappender[${elemTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = Stream(elemTy) // TODO technically a channel
      override def mergeType: Type = elemTy

      override def argTypes: Vector[Type] = Vector(UnitT)
    }
    final case class Merger(elemTy: Type, opTy: MergeOp, annotations: Option[AST.Annotations]) extends Builder {
      override def render: String = s"merger[${elemTy.render},${opTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = elemTy

      override def mergeType: Type = elemTy

      override def argTypes: Vector[Type] = Vector(UnitT) // TODO optionally allows elemTy
    }
    final case class DictMerger(keyTy: Type, valueTy: Type, opTy: MergeOp, annotations: Option[AST.Annotations])
        extends Builder {
      override def render: String = s"merger[${keyTy.render},${valueTy.render},${opTy.render}]"

      override def isComplete: Boolean = keyTy.isComplete && valueTy.isComplete

      override def resultType: Type = Dict(keyTy, valueTy)

      override def mergeType: Type = Struct(Vector(keyTy, valueTy))

      override def argTypes: Vector[Type] = Vector(UnitT)
    }
    final case class VecMerger(elemTy: Type, opTy: MergeOp, annotations: Option[AST.Annotations]) extends Builder {
      override def render: String = s"vecmerger[${elemTy.render},${opTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = Vec(elemTy)

      override def mergeType: Type = Struct(Vector(I64, elemTy))

      override def argTypes: Vector[Type] = Vector(Vec(elemTy))
    }
    final case class GroupMerger(keyTy: Type, valueTy: Type, annotations: Option[AST.Annotations]) extends Builder {
      override def render: String = s"groupmerger[${keyTy.render},${valueTy.render}]"

      override def isComplete: Boolean = keyTy.isComplete && valueTy.isComplete

      override def resultType: Type = Dict(keyTy, Vec(valueTy))

      override def mergeType: Type = Struct(Vector(keyTy, valueTy))

      override def argTypes: Vector[Type] = Vector(UnitT)
    }

    final case class Windower(
        discTy: Type,
        aggrTy: Type,
        aggrMergeTy: Type,
        aggrResultTy: Type,
        annotations: Option[AST.Annotations])
        extends Builder {
      override def render: String = s"windower[${discTy.render},${aggrTy.render}]"

      override def isComplete: Boolean = discTy.isComplete && aggrTy.isComplete

      override def resultType: Type = Stream(Struct(Vector(U64, aggrResultTy)))

      override def mergeType: Type = aggrMergeTy

      override def argTypes: Vector[Type] =
        Vector(
          // Assign
          Function(params = Vector(mergeType, Vec(U64), discTy), returnTy = Struct(Vector(Vec(U64), discTy))),
          // Trigger
          Function(params = Vector(U64, Vec(U64), discTy), returnTy = Struct(Vector(Vec(U64), discTy))),
          // Lower
          Function(params = Vector(U64, aggrTy), returnTy = Struct(Vector(U64, resultType)))
        )
    }
  }

}

sealed trait MergeOp {
  def render: String
}

object MergeOp {
  case object Sum extends MergeOp {
    override def render: String = "+"
  }
  case object Product extends MergeOp {
    override def render: String = "*"
  }
  case object Max extends MergeOp {
    override def render: String = "max"
  }
  case object Min extends MergeOp {
    override def render: String = "min"
  }
}
