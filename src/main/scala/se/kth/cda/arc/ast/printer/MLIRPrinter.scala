package se.kth.cda.arc.ast.printer

import java.io.PrintStream

import se.kth.cda.arc.ast.AST.ExprKind._
import se.kth.cda.arc.ast.AST.IterKind._
import se.kth.cda.arc.ast.AST._
import se.kth.cda.arc.ast.Type.Builder._
import se.kth.cda.arc.ast.Type._
import se.kth.cda.arc.ast._
import se.kth.cda.arc.Utils
import se.kth.cda.arc.ast.AST.UnaryOpKind.UnaryOpKind

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

    def expandLtCmp(lhs: String, rhs: String, ty: Type): String = {
      val tmp = newTmp
      val cmpOp = ty match {
        case I8  => """cmpi "slt""""
        case I16 => """cmpi "slt""""
        case I32 => """cmpi "slt""""
        case I64 => """cmpi "slt""""
        case U8  => """cmpi "ult""""
        case U16 => """cmpi "ult""""
        case U32 => """cmpi "ult""""
        case U64 => """cmpi "ult""""
        case F32 => """cmpf "olt""""
        case F64 => """cmpf "olt""""
        case _   => "/* Unsupported type: ${ty} */"
      }
      out.print(s"""${tmp} = ${cmpOp}, ${lhs}, ${rhs} : ${ty.toMLIR}\n""")
      tmp
    }

    def expandMin(result: String, lhs: String, rhs: String, ty: Type): Type = {
      out.print(s"${result} = select ${expandLtCmp(lhs, rhs, ty)}, ${lhs}, ${rhs} : ${ty.toMLIR}\n")
      ty
    }

    def expandMax(result: String, lhs: String, rhs: String, ty: Type): Type = {
      out.print(s"${result} = select ${expandLtCmp(lhs, rhs, ty)}, ${rhs}, ${lhs} : ${ty.toMLIR}\n")
      ty
    }

    def getMulOperand(ty: Type): String = {
      ty match {
        case I8  => "muli"
        case I16 => "muli"
        case I32 => "muli"
        case I64 => "muli"
        case U8  => "muli"
        case U16 => "muli"
        case U32 => "muli"
        case U64 => "muli"
        case F32 => "mulf"
        case F64 => "mulf"
      }
    }

    def expandPow(result: String, lhs: String, rhs: String, ty: Type): Type = {
      val base = newTmp
      val prod = newTmp
      out.print(s"${base} = log ${lhs} : ${ty.toMLIR}\n")
      out.print(s"${prod} = ${getMulOperand(ty)} ${base}, ${rhs} : ${ty.toMLIR}\n")
      out.print(s"${result} = exp ${prod} : ${ty.toMLIR}\n")
      ty
    }

    def dumpBinOp(identifiers: SymbolMap, kind: BinOpKind, lhs: Expr, rhs: Expr, ty: Type): String = {
      val lhsValue = lhs.toMLIR(identifiers)
      val rhsValue = rhs.toMLIR(identifiers)
      val tmp = newTmp

      val (operator: String, typeTag: Type) = (kind, ty, lhs.ty) match {
        case (BinOpKind.Add, I8, _)      => ("addi", ty)
        case (BinOpKind.Add, I16, _)     => ("addi", ty)
        case (BinOpKind.Add, I32, _)     => ("addi", ty)
        case (BinOpKind.Add, I64, _)     => ("addi", ty)
        case (BinOpKind.Add, U8, _)      => ("addi", ty)
        case (BinOpKind.Add, U16, _)     => ("addi", ty)
        case (BinOpKind.Add, U32, _)     => ("addi", ty)
        case (BinOpKind.Add, U64, _)     => ("addi", ty)
        case (BinOpKind.Add, F32, _)     => ("addf", ty)
        case (BinOpKind.Add, F64, _)     => ("addf", ty)
        case (BinOpKind.Sub, I8, _)      => ("subi", ty)
        case (BinOpKind.Sub, I16, _)     => ("subi", ty)
        case (BinOpKind.Sub, I32, _)     => ("subi", ty)
        case (BinOpKind.Sub, I64, _)     => ("subi", ty)
        case (BinOpKind.Sub, U8, _)      => ("subi", ty)
        case (BinOpKind.Sub, U16, _)     => ("subi", ty)
        case (BinOpKind.Sub, U32, _)     => ("subi", ty)
        case (BinOpKind.Sub, U64, _)     => ("subi", ty)
        case (BinOpKind.Sub, F32, _)     => ("subf", ty)
        case (BinOpKind.Sub, F64, _)     => ("subf", ty)
        case (BinOpKind.Mul, _, _)       => (getMulOperand(ty), ty)
        case (BinOpKind.Div, I8, _)      => ("divi_signed", ty)
        case (BinOpKind.Div, I16, _)     => ("divi_signed", ty)
        case (BinOpKind.Div, I32, _)     => ("divi_signed", ty)
        case (BinOpKind.Div, I64, _)     => ("divi_signed", ty)
        case (BinOpKind.Div, U8, _)      => ("divi_unsigned", ty)
        case (BinOpKind.Div, U16, _)     => ("divi_unsigned", ty)
        case (BinOpKind.Div, U32, _)     => ("divi_unsigned", ty)
        case (BinOpKind.Div, U64, _)     => ("divi_unsigned", ty)
        case (BinOpKind.Div, F32, _)     => ("divf", ty)
        case (BinOpKind.Div, F64, _)     => ("divf", ty)
        case (BinOpKind.Mod, I8, _)      => ("remi_signed", ty)
        case (BinOpKind.Mod, I16, _)     => ("remi_signed", ty)
        case (BinOpKind.Mod, I32, _)     => ("remi_signed", ty)
        case (BinOpKind.Mod, I64, _)     => ("remi_signed", ty)
        case (BinOpKind.Mod, U8, _)      => ("remi_unsigned", ty)
        case (BinOpKind.Mod, U16, _)     => ("remi_unsigned", ty)
        case (BinOpKind.Mod, U32, _)     => ("remi_unsigned", ty)
        case (BinOpKind.Mod, U64, _)     => ("remi_unsigned", ty)
        case (BinOpKind.Mod, F32, _)     => ("remf", ty)
        case (BinOpKind.Mod, F64, _)     => ("remf", ty)
        case (BinOpKind.Lt, Bool, I8)    => ("""cmpi "slt", """, lhs.ty)
        case (BinOpKind.Lt, Bool, I16)   => ("""cmpi "slt", """, lhs.ty)
        case (BinOpKind.Lt, Bool, I32)   => ("""cmpi "slt", """, lhs.ty)
        case (BinOpKind.Lt, Bool, I64)   => ("""cmpi "slt", """, lhs.ty)
        case (BinOpKind.Lt, Bool, U8)    => ("""cmpi "ult", """, lhs.ty)
        case (BinOpKind.Lt, Bool, U16)   => ("""cmpi "ult", """, lhs.ty)
        case (BinOpKind.Lt, Bool, U32)   => ("""cmpi "ult", """, lhs.ty)
        case (BinOpKind.Lt, Bool, U64)   => ("""cmpi "ult", """, lhs.ty)
        case (BinOpKind.Lt, Bool, F32)   => ("""cmpf "olt", """, lhs.ty)
        case (BinOpKind.Lt, Bool, F64)   => ("""cmpf "olt", """, lhs.ty)
        case (BinOpKind.LEq, Bool, I8)   => ("""cmpi "sle", """, lhs.ty)
        case (BinOpKind.LEq, Bool, I16)  => ("""cmpi "sle", """, lhs.ty)
        case (BinOpKind.LEq, Bool, I32)  => ("""cmpi "sle", """, lhs.ty)
        case (BinOpKind.LEq, Bool, I64)  => ("""cmpi "sle", """, lhs.ty)
        case (BinOpKind.LEq, Bool, U8)   => ("""cmpi "ule", """, lhs.ty)
        case (BinOpKind.LEq, Bool, U16)  => ("""cmpi "ule", """, lhs.ty)
        case (BinOpKind.LEq, Bool, U32)  => ("""cmpi "ule", """, lhs.ty)
        case (BinOpKind.LEq, Bool, U64)  => ("""cmpi "ule", """, lhs.ty)
        case (BinOpKind.LEq, Bool, F32)  => ("""cmpf "ole", """, lhs.ty)
        case (BinOpKind.LEq, Bool, F64)  => ("""cmpf "ole", """, lhs.ty)
        case (BinOpKind.Gt, Bool, I8)    => ("""cmpi "sgt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, I16)   => ("""cmpi "sgt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, I32)   => ("""cmpi "sgt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, I64)   => ("""cmpi "sgt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, U8)    => ("""cmpi "ugt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, U16)   => ("""cmpi "ugt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, U32)   => ("""cmpi "ugt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, U64)   => ("""cmpi "ugt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, F32)   => ("""cmpf "ogt", """, lhs.ty)
        case (BinOpKind.Gt, Bool, F64)   => ("""cmpf "ogt", """, lhs.ty)
        case (BinOpKind.GEq, Bool, I8)   => ("""cmpi "sge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, I16)  => ("""cmpi "sge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, I32)  => ("""cmpi "sge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, I64)  => ("""cmpi "sge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, U8)   => ("""cmpi "uge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, U16)  => ("""cmpi "uge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, U32)  => ("""cmpi "uge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, U64)  => ("""cmpi "uge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, F32)  => ("""cmpf "oge", """, lhs.ty)
        case (BinOpKind.GEq, Bool, F64)  => ("""cmpf "oge", """, lhs.ty)
        case (BinOpKind.Eq, Bool, Bool)  => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, I8)    => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, I16)   => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, I32)   => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, I64)   => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, U8)    => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, U16)   => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, U32)   => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, U64)   => ("""cmpi "eq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, F32)   => ("""cmpf "oeq", """, lhs.ty)
        case (BinOpKind.Eq, Bool, F64)   => ("""cmpf "oeq", """, lhs.ty)
        case (BinOpKind.NEq, Bool, Bool) => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, I8)   => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, I16)  => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, I32)  => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, I64)  => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, U8)   => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, U16)  => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, U32)  => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, U64)  => ("""cmpi "ne", """, lhs.ty)
        case (BinOpKind.NEq, Bool, F32)  => ("""cmpf "one", """, lhs.ty)
        case (BinOpKind.NEq, Bool, F64)  => ("""cmpf "one", """, lhs.ty)
        case (BinOpKind.And, Bool, Bool) => ("and", lhs.ty)
        case (BinOpKind.Or, Bool, Bool)  => ("or", lhs.ty)
        case (BinOpKind.BwAnd, I8, _)    => ("and", ty)
        case (BinOpKind.BwAnd, I16, _)   => ("and", ty)
        case (BinOpKind.BwAnd, I32, _)   => ("and", ty)
        case (BinOpKind.BwAnd, I64, _)   => ("and", ty)
        case (BinOpKind.BwAnd, U8, _)    => ("and", ty)
        case (BinOpKind.BwAnd, U16, _)   => ("and", ty)
        case (BinOpKind.BwAnd, U32, _)   => ("and", ty)
        case (BinOpKind.BwAnd, U64, _)   => ("and", ty)
        case (BinOpKind.BwOr, I8, _)     => ("or", ty)
        case (BinOpKind.BwOr, I16, _)    => ("or", ty)
        case (BinOpKind.BwOr, I32, _)    => ("or", ty)
        case (BinOpKind.BwOr, I64, _)    => ("or", ty)
        case (BinOpKind.BwOr, U8, _)     => ("or", ty)
        case (BinOpKind.BwOr, U16, _)    => ("or", ty)
        case (BinOpKind.BwOr, U32, _)    => ("or", ty)
        case (BinOpKind.BwOr, U64, _)    => ("or", ty)
        case (BinOpKind.BwXor, I8, _)    => ("xor", ty)
        case (BinOpKind.BwXor, I16, _)   => ("xor", ty)
        case (BinOpKind.BwXor, I32, _)   => ("xor", ty)
        case (BinOpKind.BwXor, I64, _)   => ("xor", ty)
        case (BinOpKind.BwXor, U8, _)    => ("xor", ty)
        case (BinOpKind.BwXor, U16, _)   => ("xor", ty)
        case (BinOpKind.BwXor, U32, _)   => ("xor", ty)
        case (BinOpKind.BwXor, U64, _)   => ("xor", ty)
        case (BinOpKind.Min, _, _)       => ("", expandMin(tmp, lhsValue, rhsValue, lhs.ty))
        case (BinOpKind.Max, _, _)       => ("", expandMax(tmp, lhsValue, rhsValue, lhs.ty))
        case (BinOpKind.Pow, _, _)       => ("", expandPow(tmp, lhsValue, rhsValue, lhs.ty))
        case _                           => s"/* unknown binop ${kind} : result-ty: ${ty}, lhs-ty: ${lhs.ty} */"
      }
      operator match {
        case "" => Unit
        case _ =>
          out.print(s"${tmp} = ${operator} ${lhsValue}, ${rhsValue} : ${typeTag.toMLIR}\n")
      }
      tmp
    }

    def dumpUnaryOp(identifiers: SymbolMap, kind: UnaryOpKind, expr: Expr, ty: Type): String = {
      val exprValue = expr.toMLIR(identifiers)
      val tmp = newTmp

      val (operator: String) = (kind, ty) match {
        case (UnaryOpKind.Exp, F32) => "exp"
        case (UnaryOpKind.Exp, F64) => "exp"
        case (UnaryOpKind.Log, F32) => "log"
        case (UnaryOpKind.Log, F64) => "log"

        case (UnaryOpKind.Cos, F32) => "cos"
        case (UnaryOpKind.Cos, F64) => "cos"
        case (UnaryOpKind.Sin, F32) =>
          out.print(s"""${tmp} = "arc.sin"(${exprValue}) : (${ty.toMLIR}) -> ${ty.toMLIR}\n"""); ""
        case (UnaryOpKind.Sin, F64) =>
          out.print(s"""${tmp} = "arc.sin"(${exprValue}) : (${ty.toMLIR}) -> ${ty.toMLIR}\n"""); ""
        case (UnaryOpKind.Tan, F32) =>
          out.print(s"""${tmp} = "arc.tan"(${exprValue}) : (${ty.toMLIR}) -> ${ty.toMLIR}\n"""); ""
        case (UnaryOpKind.Tan, F64) =>
          out.print(s"""${tmp} = "arc.tan"(${exprValue}) : (${ty.toMLIR}) -> ${ty.toMLIR}\n"""); ""
      }
      operator match {
        case "" => Unit
        case _ =>
          out.print(s"${tmp} = ${operator} ${exprValue} : ${ty.toMLIR}\n")
      }
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
        case UnaryOp(kind, expr)     => dumpUnaryOp(identifiers, kind, expr, self.ty)
        case Merge(builder, value)   => s""
        case Result(expr)            => s""
        case NewBuilder(ty, args)    => s""
        case BinOp(kind, lhs, rhs)   => dumpBinOp(identifiers, kind, lhs, rhs, self.ty)
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
      case U8                                                               => s"i8"
      case U16                                                              => s"i16"
      case U32                                                              => s"i32"
      case U64                                                              => s"i64"
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
