package se.kth.cda.arc

import java.io.PrintStream

object PrettyPrint {
  import AST._

  val INDENT_INC = 2

  def print(tree: ASTNode): String = {
    val sb = new Utils.StringBuilderStream()
    this.print(tree, sb.asPrintStream())
    sb.result()
  }

  def println(tree: ASTNode, out: PrintStream): Unit = {
    PrettyPrint.print(tree, out)
    out.print('\n')
  }

  def print(tree: ASTNode, out: PrintStream): Unit = {
    tree match {
      case Program(macros, expr, _) =>
        macros.foreach { m =>
          print(m, out)
          out.print('\n')
        }
        print(expr, out)
      case Macro(name, params, body, _) =>
        out.print("macro ")
        printSymbol(name, out)
        out.print('(')
        for ((p, i) <- params.view.zipWithIndex) {
          printSymbol(p, out)
          if (i != (params.length - 1)) {
            out.print(',')
          }
        }
        out.print(')')
        out.print('=')
        print(body, out)
      case e: Expr   => printExpr(e, out, typed = true, 0, shouldIndent = true)
      case s: Symbol => printSymbol(s, out)
    }
  }

  def printSymbol(s: Symbol, out: PrintStream): Unit = out.print(s.text)

  def printType(t: Type, out: PrintStream): Unit = out.print(t.render)

  def printIter(iter: Iter, out: PrintStream, indent: Int): Unit = {
    import IterKind._

    val iterStr = iter.kind match {
      case ScalarIter  => "iter"
      case SimdIter    => "simditer"
      case FringeIter  => "fringeiter"
      case NdIter      => "nditer"
      case RangeIter   => "rangeiter"
      case NextIter    => "nextiter"
      case KeyByIter   => "keyby"
      case UnknownIter => "?iter"
    }

    if (iter.kind == NdIter) {
      out.print(iterStr)
      out.print('(')
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.start.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.shape.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.strides.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(')')
    } else if (iter.kind == KeyByIter) {
      out.print(iterStr)
      out.print('(')
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.keyFunc.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(')')
    } else if (iter.start.isDefined) {
      out.print(iterStr)
      out.print('(')
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.start.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.end.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(',')
      printExpr(iter.stride.get, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(')')
    } else if (iter.kind != ScalarIter) {
      out.print(iterStr)
      out.print('(')
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false)
      out.print(')')
    } else {
      printExpr(iter.data, out, typed = true, indent, shouldIndent = false)
    }
  }

  def printExpr(expr: Expr, out: PrintStream, typed: Boolean, indent: Int, shouldIndent: Boolean): Unit = {
    import ExprKind._
    lazy val indentStr = if (shouldIndent) {
      (0 until indent).foldLeft("")((acc, _) => acc + " ")
    } else {
      ""
    }
    // lazy val lessIndentStr = (0 until (indent - 2)).foldLeft("")((acc, _) => acc + " ")
    expr.kind match {
      case Let(name, bindingTy, value, body) =>
        if (typed) {
          out.print('(')
          out.print(' ')
        }
        out.print("let ")
        printSymbol(name, out)
        out.print(':')
        printType(bindingTy, out)
        out.print('=')
        printExpr(value, out, typed = true, indent + INDENT_INC, shouldIndent = true)
        out.print(';')
        out.print('\n')
        out.print(indentStr)
        if (typed) {
          out.print("  ")
        }
        printExpr(body, out, typed = false, if (typed) indent + INDENT_INC else indent, shouldIndent = true)
        if (typed) {
          out.print('\n')
          out.print(indentStr)
          out.print(')')
          out.print(':')
          printType(expr.ty, out)
        }
      case Lambda(params, body) =>
        if (params.isEmpty) {
          out.print("||")
          printExpr(body, out, typed = true, indent + INDENT_INC, shouldIndent = true)
        } else {
          out.print('|')
          for ((p, i) <- params.view.zipWithIndex) {
            printSymbol(p.name, out)
            out.print(':')
            printType(p.ty, out)
            if (i != (params.length - 1)) {
              out.print(',')
            }
          }
          out.print('|')
          out.print('\n')
          out.print(indentStr)
          out.print("  ")
          printExpr(body, out, typed = true, indent + INDENT_INC, shouldIndent = true)
        }
      case Negate(expr) =>
        out.print('-')
        printExpr(expr, out, typed = true, indent + 1, shouldIndent = false)
      case Not(expr) =>
        out.print('!')
        printExpr(expr, out, typed = true, indent + 1, shouldIndent = false)
      case UnaryOp(kind, expr) =>
        out.print(UnaryOpKind.print(kind))
        out.print('(')
        printExpr(expr, out, typed = false, indent + 1, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Ident(s) =>
        printSymbol(s, out)
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case l: Literal[_] =>
        out.print(l.raw)
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Cast(ty, e) =>
        printType(ty, out)
        out.print('(')
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
      case ToVec(e) =>
        out.print("tovec(")
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
      case Broadcast(e) =>
        out.print("broadcast(")
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
      case CUDF(ref, args, retT) =>
        out.print("cudf[")
        ref match {
          case Left(name) => printSymbol(name, out);
          case Right(pointer) =>
            out.print('*')
            printExpr(pointer, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        }
        out.print(',')
        printType(retT, out)
        out.print(']')
        out.print('(')
        for ((e, i) <- args.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
          if (i != (args.length - 1)) {
            out.print(',')
          }
        }
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Zip(params) =>
        out.print("zip(")
        for ((e, i) <- params.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + 4, shouldIndent = false)
          if (i != (params.length - 1)) {
            out.print(',')
          }
        }
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Hash(params) =>
        out.print("hash(")
        for ((e, i) <- params.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + 4, shouldIndent = false)
          if (i != (params.length - 1)) {
            out.print(',')
          }
        }
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case For(iterator, builder, body) =>
        out.print("for(")
        printIter(iterator, out, indent + 4)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("    ")
        printExpr(builder, out, typed = false, indent + 4, shouldIndent = true)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("    ")
        printExpr(body, out, typed = false, indent + 4, shouldIndent = true)
        out.print('\n')
        out.print(indentStr)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Len(e) =>
        out.print("len(")
        printExpr(e, out, typed = true, indent + 4, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Lookup(data, key) =>
        out.print("lookup(")
        printExpr(data, out, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        printExpr(key, out, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Slice(data, index, size) =>
        out.print("slice(")
        printExpr(data, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        printExpr(index, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        printExpr(size, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Sort(data, keyFunc) =>
        out.print("sort(")
        printExpr(data, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        printExpr(keyFunc, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Serialize(e) =>
        out.print("serialize(")
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Deserialize(ty, e) =>
        out.print("deserialize[")
        printType(ty, out)
        out.print(']')
        out.print('(')
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case If(cond, onTrue, onFalse) =>
        out.print("if (")
        printExpr(cond, out, typed = false, indent + 4, shouldIndent = false)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("    ")
        printExpr(onTrue, out, typed = true, indent + 4, shouldIndent = true)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("    ")
        printExpr(onFalse, out, typed = true, indent + 4, shouldIndent = true)
        out.print('\n')
        out.print(indentStr)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Select(cond, onTrue, onFalse) =>
        out.print("select(")
        printExpr(cond, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("  ")
        printExpr(onTrue, out, typed = true, indent + INDENT_INC, shouldIndent = true)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("  ")
        printExpr(onFalse, out, typed = true, indent + INDENT_INC, shouldIndent = true)
        out.print('\n')
        out.print(indentStr)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Iterate(init, updateFunc) =>
        out.print("iterate (")
        printExpr(init, out, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        out.print('\n')
        out.print(indentStr)
        out.print("  ")
        printExpr(updateFunc, out, typed = true, indent + INDENT_INC, shouldIndent = true)
        out.print('\n')
        out.print(indentStr)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case MakeStruct(elems) =>
        out.print('{')
        for ((e, i) <- elems.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + 1, shouldIndent = false)
          if (i != (elems.length - 1)) {
            out.print(',')
          }
        }
        out.print('}')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case MakeVec(elems) =>
        out.print('[')
        for ((e, i) <- elems.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + 1, shouldIndent = false)
          if (i != (elems.length - 1)) {
            out.print(',')
          }
        }
        out.print(']')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Merge(builder, value) =>
        out.print("merge(")
        printExpr(builder, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(',')
        printExpr(value, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Result(e) =>
        out.print("result(")
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case NewBuilder(ty, Some(arg)) =>
        printType(ty, out)
        out.print('(')
        printExpr(arg, out, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.print(')')
      // don't print type even if requested since it's redundant
      case NewBuilder(ty, None) =>
        printType(ty, out)
      // don't print type even if requested since it's redundant
      case BinOp(kind, left, right) =>
        if (kind.isInfix) {
          if (typed) {
            out.print('(')
          }
          printExpr(left, out, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
          out.print(kind.symbol)
          printExpr(right, out, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
          if (typed) {
            out.print(')')
            out.print(':')
            printType(expr.ty, out)
          }
        } else {
          out.print(kind.symbol)
          out.print('(')
          printExpr(left, out, typed = true, indent + 4, shouldIndent = false)
          out.print(',')
          printExpr(right, out, typed = true, indent + 4, shouldIndent = false)
          out.print(')')
          if (typed) {
            out.print(':')
            printType(expr.ty, out)
          }
        }
      case Application(fun, args) =>
        printExpr(fun, out, typed = false, indent, shouldIndent = false)
        out.print('(')
        for ((e, i) <- args.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false)
          if (i != (args.length - 1)) {
            out.print(',')
          }
        }
        out.print(')')
        if (typed) {
          out.print(':')
          printType(expr.ty, out)
        }
      case Ascription(e, ty) =>
        out.print('(')
        printExpr(e, out, typed = false, indent + 1, shouldIndent = false)
        out.print(')')
        out.print(':')
        printType(ty, out)
      case Projection(struct, index) =>
        if (typed) {
          out.print('(')
        }
        printExpr(struct, out, typed = false, indent, shouldIndent = false)
        out.print('.')
        out.print('$')
        out.print(index.toString)
        if (typed) {
          out.print(')')
          out.print(':')
          printType(expr.ty, out)
        }
    }
  }
}
