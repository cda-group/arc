package se.kth.cda.arc

sealed trait Type extends Equals {
  def render: String

  def isComplete: Boolean
}
sealed trait ConcreteType extends Type {
  override def isComplete: Boolean = true
}
sealed trait CompositeType extends Type

sealed trait BuilderType extends CompositeType {
  def annotations: Option[AST.Annotations]

  def resultType: Type

  def mergeType: Type

  def argType: Type
}

object Types {

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
    //override def isAssignableTo(t: Type): Boolean = this == t; // no subtyping among scalars
    //override def isAssignableFrom(t: Type): Boolean = this == t; // no subtyping among scalars
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
  final case class Vec(elemTy: Type) extends CompositeType {
    override def render: String = s"vec[${elemTy.render}]"

    override def isComplete: Boolean = elemTy.isComplete
  }
  final case class Dict(keyTy: Type, valueTy: Type) extends CompositeType {
    override def render: String = s"dict[${keyTy.render}, ${valueTy.render}]"

    override def isComplete: Boolean = keyTy.isComplete && valueTy.isComplete
  }
  final case class Struct(elemTys: Vector[Type]) extends CompositeType {
    override def render: String = elemTys.map(_.render).mkString("{", ",", "}")

    override def isComplete: Boolean = elemTys.forall(_.isComplete)
  }
  final case class Simd(elemTy: Type) extends CompositeType {
    override def render: String = s"simd[${elemTy.render}]"

    override def isComplete: Boolean = elemTy.isComplete
  }
  final case class Stream(elemTy: Type) extends CompositeType {
    override def render: String = s"stream[${elemTy.render}]"

    override def isComplete: Boolean = elemTy.isComplete
  }
  final case class Function(params: Vector[Type], returnTy: Type) extends CompositeType {
    override def render: String = s"|${params.map(_.render).mkString(",")}|(${returnTy.render})"

    override def isComplete: Boolean = params.forall(_.isComplete) && returnTy.isComplete
  }

  object Builders {
    final case class Appender(elemTy: Type, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"appender[${elemTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = Vec(elemTy)

      override def mergeType: Type = elemTy

      override def argType: Type = UnitT; // TODO optionally allows an index?
    }
    final case class StreamAppender(elemTy: Type, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"streamappender[${elemTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = Stream(elemTy); // TODO technically a channel
      override def mergeType: Type = elemTy

      override def argType: Type = UnitT
    }
    final case class Merger(elemTy: Type, opTy: OpType, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"merger[${elemTy.render},${opTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = elemTy

      override def mergeType: Type = elemTy

      override def argType: Type = UnitT; // TODO optionally allows elemTy
    }
    final case class DictMerger(keyTy: Type, valueTy: Type, opTy: OpType, annotations: Option[AST.Annotations])
        extends BuilderType {
      override def render: String = s"merger[${keyTy.render},${valueTy.render},${opTy.render}]"

      override def isComplete: Boolean = keyTy.isComplete && valueTy.isComplete

      override def resultType: Type = Dict(keyTy, valueTy)

      override def mergeType: Type = Struct(Vector(keyTy, valueTy))

      override def argType: Type = UnitT
    }
    final case class VecMerger(elemTy: Type, opTy: OpType, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"vecmerger[${elemTy.render},${opTy.render}]"

      override def isComplete: Boolean = elemTy.isComplete

      override def resultType: Type = Vec(elemTy)

      override def mergeType: Type = Struct(Vector(I64, elemTy))

      override def argType: Type = Vec(elemTy)
    }
    final case class GroupMerger(keyTy: Type, valueTy: Type, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"groupmerger[${keyTy.render},${valueTy.render}]"

      override def isComplete: Boolean = keyTy.isComplete && valueTy.isComplete

      override def resultType: Type = Dict(keyTy, Vec(valueTy))

      override def mergeType: Type = Struct(Vector(keyTy, valueTy))

      override def argType: Type = UnitT
    }
  }
}

sealed trait OpType {
  def render: String
}

object OpTypes {
  case object Sum extends OpType {
    override def render: String = "+"
  }
  case object Product extends OpType {
    override def render: String = "*"
  }
  case object Max extends OpType {
    override def render: String = "max"
  }
  case object Min extends OpType {
    override def render: String = "min"
  }
}
