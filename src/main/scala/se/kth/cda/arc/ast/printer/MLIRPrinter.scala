package se.kth.cda.arc.ast.printer

import java.io.PrintStream

import se.kth.cda.arc.ast.AST.ExprKind._
import se.kth.cda.arc.ast.AST.IterKind._
import se.kth.cda.arc.ast.AST._
import se.kth.cda.arc.ast.Type.Builder._
import se.kth.cda.arc.ast.Type._
import se.kth.cda.arc.ast._
import se.kth.cda.arc.Utils

object MLIRPrinter {

  type SymbolMap = Map[String, String]
  def SymbolMap(): SymbolMap = Map[String, String]()

  private var tmpCounter: Long = 0

  def newTmp: String = {
    val t = s"%${tmpCounter}"
    tmpCounter += 1
    t
  }

  private var out: PrintStream = null

  def addSymbol(map: SymbolMap, sym: Symbol, string: String): SymbolMap = {
    map + (sym.name -> string)
  }

  def symbolAsString(sym: Symbol): String = {
    val Symbol(name, token, scope) = sym
    s"%${name}_$scope"
  }

  implicit class ASTToMLIR(val self: ASTNode) extends AnyVal {

    def toMLIR: String = {
      val sb = new Utils.StringBuilderStream()
      out = sb.asPrintStream()
      val identifiers: SymbolMap = SymbolMap()

      out.print("module @toplevel {\n")
      out.print("func @main() {\n")
      val body =
        self match {
          case symbol: Symbol =>
            symbol.toMLIR(identifiers: SymbolMap)
          case Program(_, expr, _) =>
            expr.toMLIR(identifiers: SymbolMap)
          case expr: Expr =>
            expr.toMLIR(identifiers: SymbolMap)
        }
      out.print("return\n}\n")
      out.print("}\n")
      s"${sb.result()}\n"
    }
  }

  implicit class SymbolToMLIR(val self: Symbol) extends AnyVal {

    def toMLIR(identifiers: SymbolMap): String = {
      identifiers(self.name)
    }
  }

  implicit class ExprToMLIR(val self: Expr) extends AnyVal {

    def dumpLet(identifiers: SymbolMap, symbol: Symbol, bindingTy: Type, value: Expr, body: Expr): String = {
      out.print(s"// let ${symbolAsString(symbol)}\n")
      val letValue = value.toMLIR(identifiers)
      val bodyEnv = addSymbol(identifiers, symbol, letValue)
      val bodyValue = body.toMLIR(bodyEnv)
      bodyValue
    }

    def dumpBinOp(identifiers: SymbolMap, kind: BinOpKind, lhs: Expr, rhs: Expr): String = {
      val operator = kind match {
        case BinOpKind.Add => s"addi"
        case _             => "/* unknown binop */"
      }

      val lhsValue = lhs.toMLIR(identifiers)
      val rhsValue = rhs.toMLIR(identifiers)
      val tmp = newTmp
      out.print(s"${tmp} = ${operator} ${lhsValue}, ${rhsValue} : ${self.ty.toMLIR}\n")
      tmp
    }

    def dumpVec(identifiers: SymbolMap, elems: Vector[Expr], ty: Type): String = {
      val Vec(elemTy) = ty
      val es = elems.map(_.toMLIR(identifiers))
      val elementTypes = es.map(_ => elemTy.toMLIR).mkString(", ")
      val tmp = newTmp
      out.print(s"""${tmp} = "arc.make_vector"(${es
        .mkString(", ")}) : (${elementTypes}) -> tensor<${es.length}x${elemTy.toMLIR}>\n""");
      s"${tmp}"
      tmp
    }

    def toMLIR(identifiers: SymbolMap): String = {
      self.kind match {
        case Literal.I8(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.I16(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.I32(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.I64(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.U8(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.U16(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.U32(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.U64(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.F32(raw, _) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${raw.dropRight(3)} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.F64(raw, _) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${raw} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.Bool(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${if (value) 1 else 0} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.UnitL(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Literal.StringL(raw, value) => {
          val tmp = newTmp
          out.print(s"${tmp} = constant ${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Let(symbol, bindingTy, value, body) => dumpLet(identifiers, symbol, bindingTy, value, body)
        case Lambda(params, body)                => s""
        case Cast(ty, expr)                      => s""
        case ToVec(expr)                         => s""
        case Ident(symbol)                       => s"${identifiers(symbol.name)}"
        case MakeStruct(elems)                   => s""
        case MakeVec(elems)                      => dumpVec(identifiers, elems, self.ty)
        case If(cond, onTrue, onFalse)           => s""
        case Select(cond, onTrue, onFalse)       => s""
        case Iterate(initial, updateFunc)        => s""
        case Broadcast(expr)                     => s""
        case Serialize(expr)                     => s""
        case Deserialize(ty, expr)               => s""
        case CUDF(reference, args, returnTy)     => s""
        case Zip(params)                         => s""
        case Hash(params)                        => s""
        case For(iterator, builder, body)        => s""
        case Len(expr)                           => s""
        case Lookup(data, key)                   => s""
        case Slice(data, index, size)            => s""
        case Sort(data, keyFunc)                 => s""
        case Drain(source, sink)                 => s""
        case Negate(Expr(Literal.I8(_, value), _, _, _)) => {
          val tmp = newTmp; out.print(s"${tmp} = constant -${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Negate(Expr(Literal.I16(_, value), _, _, _)) => {
          val tmp = newTmp; out.print(s"${tmp} = constant -${value} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        /* The AST represents negative literals as a Scala signed integer, as
           an i32 can be -2147483648, and 2147483648 cannot be
           represented, we use the raw string. We do the same for
           i64 */
        case Negate(Expr(Literal.I32(raw, _), _, _, _)) => {
          val tmp = newTmp; out.print(s"${tmp} = constant -${raw} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Negate(Expr(Literal.I64(raw, _), _, _, _)) => { // We have to drop the L/l suffix
          val tmp = newTmp; out.print(s"${tmp} = constant -${raw.dropRight(3)} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Negate(Expr(Literal.F32(raw, _), _, _, _)) => {
          val tmp = newTmp; out.print(s"${tmp} = constant -${raw.dropRight(3)} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Negate(Expr(Literal.F64(raw, _), _, _, _)) => {
          val tmp = newTmp; out.print(s"${tmp} = constant -${raw} : ${self.ty.toMLIR}\n"); s"${tmp}"
        }
        case Negate(expr)            => s""
        case Not(expr)               => s""
        case UnaryOp(kind, expr)     => s""
        case Merge(builder, value)   => s""
        case Result(expr)            => s""
        case NewBuilder(ty, args)    => s""
        case BinOp(kind, lhs, rhs)   => dumpBinOp(identifiers, kind, lhs, rhs)
        case Application(expr, args) => s""
        case Projection(expr, index) => s""
        case Ascription(expr, ty)    => s""
      }
    }
  }

  implicit class IteratorToMLIR(val self: IterKind) extends AnyVal {

    def toMLIR: String = self match {
      case ScalarIter  => s"x"
      case SimdIter    => s"y"
      case FringeIter  => s"z"
      case NdIter      => s"å"
      case RangeIter   => s"ä"
      case NextIter    => s"ö"
      case KeyByIter   => s"A"
      case UnknownIter => s"B"
    }
  }

  implicit class TypeToMLIR(val self: Type) extends AnyVal {

    def toMLIR: String = self match {
      case I8                                                               => s"i8"
      case I16                                                              => s"i16"
      case I32                                                              => s"i32"
      case I64                                                              => s"i64"
      case U8                                                               => s"u8"
      case U16                                                              => s"u16"
      case U32                                                              => s"u32"
      case U64                                                              => s"u64"
      case F32                                                              => s"f32"
      case F64                                                              => s"f64"
      case Bool                                                             => s"i1"
      case UnitT                                                            => s"none"
      case StringT                                                          => s""
      case Appender(elemTy, annotations)                                    => s""
      case StreamAppender(elemTy, annotations)                              => s""
      case Merger(elemTy, opTy, annotations)                                => s""
      case DictMerger(keyTy, valueTy, opTy, annotations)                    => s""
      case VecMerger(elemTy, opTy, annotations)                             => s""
      case GroupMerger(keyTy, valueTy, annotations)                         => s""
      case Windower(discTy, aggrTy, aggrMergeTy, aggrResultTy, annotations) => s""
      case Vec(elemTy)                                                      => s"tensor<?x${elemTy.toMLIR}>"
      case Dict(keyTy, valueTy)                                             => s""
      case Struct(elemTys)                                                  => s""
      case Simd(elemTy)                                                     => s""
      case Stream(elemTy)                                                   => s""
      case Function(params, returnTy)                                       => s""
      case TypeVariable(id)                                                 => s""
    }
  }

  implicit class NewBuilderToMLIR(val self: NewBuilder) extends AnyVal {

    def toMLIR: String = {
      val NewBuilder(ty, args) = self
      s""
    }
  }

  implicit class AnnotationToMLIR(val self: Annotations) extends AnyVal {

    def toMLIR: String = {
      val Annotations(params) = self
      s""
    }
  }

}
