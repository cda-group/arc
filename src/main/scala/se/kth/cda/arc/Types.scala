package se.kth.cda.arc

sealed trait Type {
  def render: String;
  def upperBound(t: Type, reason: String): Unit;
  def lowerBound(t: Type, reason: String): Unit;
}
sealed trait ConcreteType extends Type {
  def isAssignableTo(t: Type): Boolean;
  def isAssignableFrom(t: Type): Boolean;
  override def upperBound(t: Type, reason: String): Unit = {
    assert(this.isAssignableTo(t), s"Type ${this.render} can not be assigned to ${t.render}! Caused by '$reason'.");
  }

  override def lowerBound(t: Type, reason: String): Unit = {
    assert(this.isAssignableFrom(t), s"Type ${t.render} can not be assigned to ${this.render}! Caused by '$reason'.");
  }
}
sealed trait TypeBound extends Type;
sealed trait CompositeType extends Type;
sealed trait BuilderType extends CompositeType {
  def annotations: Option[AST.Annotations];
}
object Types {
  def unknown: TypeBound = new LUBound();

  class LUBound extends TypeBound {
    private var lower: Option[ConcreteType] = None;
    private var upper: Option[ConcreteType] = None;
    override def upperBound(t: Type, reason: String): Unit = {
      println(s"Got upper bound $t");
      ???
    }
    override def lowerBound(t: Type, reason: String): Unit = {
      println(s"Got lower bound $t");
      ???
    }
    override def toString(): String = {
      (lower, upper) match {
        case (None, None)                   => "TypeBound(?)"
        case (None, Some(t))                => s"TypeBound(?<:${t.render})"
        case (Some(t), None)                => s"TypeBound(${t.render}<:?)"
        case (Some(t), Some(t2)) if t == t2 => s"TypeBound(${t.render})"
        case (Some(t), Some(t2))            => s"TypeBound(${t.render}<:?<:${t2.render})"
      }
    }
    override def render: String = "?";
    def isResolved: Boolean = !lower.isEmpty && (lower == upper);
    def toConcrete: Option[ConcreteType] = {
      (lower, upper) match {
        case (None, None)                   => None
        case (None, Some(t))                => Some(t) // resolve to upper bound
        case (Some(t), None)                => None // TODO Check how to infer lower bound
        case (Some(t), Some(t2)) if t == t2 => Some(t)
        case (Some(t), Some(t2))            => Some(t2) // resolve to upper bound
      }
    }
  }

  sealed trait Scalar extends ConcreteType with Equals {
    override def isAssignableTo(t: Type): Boolean = this == t; // no subtyping among scalars
    override def isAssignableFrom(t: Type): Boolean = this == t; // no subtyping among scalars
  }
  case object I8 extends Scalar {
    override def render: String = "i8";
  }
  case object I16 extends Scalar {
    override def render: String = "i16";
  }
  case object I32 extends Scalar {
    override def render: String = "i32";
  }
  case object I64 extends Scalar {
    override def render: String = "i64";
  }
  case object U8 extends Scalar {
    override def render: String = "u8";
  }
  case object U16 extends Scalar {
    override def render: String = "u16";
  }
  case object U32 extends Scalar {
    override def render: String = "u32";
  }
  case object U64 extends Scalar {
    override def render: String = "u64";
  }
  case object F32 extends Scalar {
    override def render: String = "f32";
  }
  case object F64 extends Scalar {
    override def render: String = "f64";
  }
  case object Bool extends Scalar {
    override def render: String = "bool";
  }
  case object UnitT extends Scalar {
    override def render: String = "unit";
  }
  case object StringT extends Scalar {
    override def render: String = "string";
  }
  case class Vec(elemType: Type) extends CompositeType {
    override def render: String = s"vec[${elemType.render}]";
    override def upperBound(t: Type, reason: String): Unit = {
      //      t match {
      //        case Vector(otherET) => otherET.upperBound(elemType, reason)
      //        case _               => fail(s"Type ${t.render} can not be bounded by ${this.render}! Caused by '$reason'.");
      //      }
      ???
    }
    override def lowerBound(t: Type, reason: String): Unit = ???;
  }
  case class Struct(elemTypes: Vector[Type]) extends CompositeType {
    override def render: String = elemTypes.map(_.render).mkString("{", ",", "}");
    override def upperBound(t: Type, reason: String): Unit = ???;
    override def lowerBound(t: Type, reason: String): Unit = ???;
  }
  case class Simd(elemType: Type) extends CompositeType {
    override def render: String = s"simd[${elemType.render}]";
    override def upperBound(t: Type, reason: String): Unit = ???;
    override def lowerBound(t: Type, reason: String): Unit = ???;
  }
  case class Function(params: Vector[Type], returnTy: Type) extends CompositeType {
    override def render: String = s"|${params.map(_.render).mkString(",")}|(${returnTy.render})";
    override def upperBound(t: Type, reason: String): Unit = ???;
    override def lowerBound(t: Type, reason: String): Unit = ???;
  }
  object Builders {
    case class Appender(elemTy: Type, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"appender[${elemTy.render}]";
      override def upperBound(t: Type, reason: String): Unit = ???;
      override def lowerBound(t: Type, reason: String): Unit = ???;
    }
    case class StreamAppender(elemTy: Type, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"streamappender[${elemTy.render}]";
      override def upperBound(t: Type, reason: String): Unit = ???;
      override def lowerBound(t: Type, reason: String): Unit = ???;
    }
    case class Merger(elemTy: Type, opTy: OpType, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"merger[${elemTy.render},${opTy.render}]";
      override def upperBound(t: Type, reason: String): Unit = ???;
      override def lowerBound(t: Type, reason: String): Unit = ???;
    }
    case class DictMerger(keyTy: Type, valueTy: Type, opTy: OpType, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"merger[${keyTy.render},${valueTy.render},${opTy.render}]";
      override def upperBound(t: Type, reason: String): Unit = ???;
      override def lowerBound(t: Type, reason: String): Unit = ???;
    }
    case class VecMerger(elemTy: Type, opTy: OpType, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"vecmerger[${elemTy.render},${opTy.render}]";
      override def upperBound(t: Type, reason: String): Unit = ???;
      override def lowerBound(t: Type, reason: String): Unit = ???;
    }
    case class GroupMerger(keyTy: Type, valueTy: Type, annotations: Option[AST.Annotations]) extends BuilderType {
      override def render: String = s"groupmerger[${keyTy.render},${valueTy.render}]";
      override def upperBound(t: Type, reason: String): Unit = ???;
      override def lowerBound(t: Type, reason: String): Unit = ???;
    }
  }
}

sealed trait OpType {
  def render: String;
}
object OpTypes {
  case object Sum extends OpType {
    override def render: String = "+";
  }
  case object Product extends OpType {
    override def render: String = "*";
  }
  case object Max extends OpType {
    override def render: String = "max";
  }
  case object Min extends OpType {
    override def render: String = "min";
  }
}
