package se.kth.cda.arc.syntaxtree

import java.io.PrintStream

import se.kth.cda.arc.Utils

object PrettyPrint {
  import se.kth.cda.arc.syntaxtree.AST._

  val INDENT_INC = 2

  def pretty(tree: ASTNode): String = {
    val sb = new Utils.StringBuilderStream()
    sb.asPrintStream().prettyPrint(tree)
    sb.result()
  }

  def print(tree: ASTNode): Unit = {
    val sb = new Utils.StringBuilderStream()
    sb.asPrintStream().prettyPrint(tree)
  }

  implicit class PrettyPrintStream(val out: PrintStream) extends AnyVal {

    def prettyPrint(str: String): Unit = out.print(str)

    def prettyPrint(ch: Char): Unit = out.print(ch)

    def prettyPrintln(tree: ASTNode): Unit = {
      out.prettyPrint(tree)
      out.prettyPrint('\n')
    }

    def prettyPrint(tree: ASTNode): Unit = {
      tree match {
        case Program(macros, expr, _) =>
          macros.foreach { m =>
            out.prettyPrint(m)
            out.prettyPrint('\n')
          }
          out.prettyPrint(expr)
        case Macro(name, params, body, _) =>
          out.prettyPrint("macro ")
          out.prettyPrint(name)
          out.prettyPrint('(')
          for ((p, i) <- params.view.zipWithIndex) {
            out.prettyPrint(p)
            if (i != (params.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint(')')
          out.prettyPrint('=')
          out.prettyPrint(body)
        case e: Expr   => out.prettyPrint(e, typed = true, 0, shouldIndent = true)
        case s: Symbol => out.prettyPrint(s)
      }
    }

    def prettyPrint(s: Symbol): Unit = out.prettyPrint(s.name)

    def prettyPrint(t: Type): Unit = out.prettyPrint(t.render)

    def prettyPrint(iter: Iter, indent: Int): Unit = {
      import IterKind._

      val iterStr = iter.kind match {
        case ScalarIter  => "iter"
        case SimdIter    => "simditer"
        case FringeIter  => "fringeiter"
        case NdIter      => "nditer"
        case RangeIter   => "rangeiter"
        case NextIter    => "nextiter"
        case KeyByIter   => "keyby"
        case UnknownIter => "iter" // Make sure this doesn't happen
      }

      if (iter.kind == NdIter) {
        out.prettyPrint(iterStr)
        out.prettyPrint('(')
        out.prettyPrint(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.start.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.shape.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.strides.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(')')
      } else if (iter.kind == KeyByIter) {
        out.prettyPrint(iterStr)
        out.prettyPrint('(')
        out.prettyPrint(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.keyFunc.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(')')
      } else if (iter.start.isDefined) {
        out.prettyPrint(iterStr)
        out.prettyPrint('(')
        out.prettyPrint(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.start.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.end.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(',')
        out.prettyPrint(iter.stride.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(')')
      } else if (iter.kind != ScalarIter) {
        out.prettyPrint(iterStr)
        out.prettyPrint('(')
        out.prettyPrint(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.prettyPrint(')')
      } else {
        out.prettyPrint(iter.data, typed = true, indent, shouldIndent = false)
      }
    }

    def prettyPrint(expr: Expr, typed: Boolean, indent: Int, shouldIndent: Boolean): Unit = {
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
            out.prettyPrint('(')
            out.prettyPrint(' ')
          }
          out.prettyPrint("let ")
          out.prettyPrint(name)
          out.prettyPrint(':')
          out.prettyPrint(bindingTy)
          out.prettyPrint('=')
          out.prettyPrint(value, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.prettyPrint(';')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          if (typed) {
            out.prettyPrint("  ")
          }
          out.prettyPrint(body, typed = false, if (typed) indent + INDENT_INC else indent, shouldIndent = true)
          if (typed) {
            out.prettyPrint('\n')
            out.prettyPrint(indentStr)
            out.prettyPrint(')')
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Lambda(params, body) =>
          if (params.isEmpty) {
            out.prettyPrint("||")
            out.prettyPrint(body, typed = true, indent + INDENT_INC, shouldIndent = true)
          } else {
            out.prettyPrint('|')
            for ((p, i) <- params.view.zipWithIndex) {
              out.prettyPrint(p.symbol)
              out.prettyPrint(':')
              out.prettyPrint(p.ty)
              if (i != (params.length - 1)) {
                out.prettyPrint(',')
              }
            }
            out.prettyPrint('|')
            out.prettyPrint('\n')
            out.prettyPrint(indentStr)
            out.prettyPrint("  ")
            out.prettyPrint(body, typed = true, indent + INDENT_INC, shouldIndent = true)
          }
        case Negate(e) =>
          out.prettyPrint('-')
          out.prettyPrint(e, typed = true, indent + 1, shouldIndent = false)
        case Not(e) =>
          out.prettyPrint('!')
          out.prettyPrint(e, typed = true, indent + 1, shouldIndent = false)
        case UnaryOp(kind, e) =>
          out.prettyPrint(kind.toString)
          out.prettyPrint('(')
          out.prettyPrint(e, typed = false, indent + 1, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(e.ty)
          }
        case Ident(s) =>
          out.prettyPrint(s)
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case l: Literal[_] =>
          out.prettyPrint(l.raw)
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Cast(ty, e) =>
          out.prettyPrint(ty)
          out.prettyPrint('(')
          out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
        case ToVec(e) =>
          out.prettyPrint("tovec(")
          out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
        case Broadcast(e) =>
          out.prettyPrint("broadcast(")
          out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
        case CUDF(ref, args, retT) =>
          out.prettyPrint("cudf[")
          ref match {
            case Left(name) => out.prettyPrint(name);
            case Right(pointer) =>
              out.prettyPrint('*')
              out.prettyPrint(pointer, typed = false, indent + INDENT_INC, shouldIndent = false)
          }
          out.prettyPrint(',')
          out.prettyPrint(retT)
          out.prettyPrint(']')
          out.prettyPrint('(')
          for ((e, i) <- args.view.zipWithIndex) {
            out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
            if (i != (args.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Zip(params) =>
          out.prettyPrint("zip(")
          for ((e, i) <- params.view.zipWithIndex) {
            out.prettyPrint(e, typed = false, indent + 4, shouldIndent = false)
            if (i != (params.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Hash(params) =>
          out.prettyPrint("hash(")
          for ((e, i) <- params.view.zipWithIndex) {
            out.prettyPrint(e, typed = false, indent + 4, shouldIndent = false)
            if (i != (params.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case For(iterator, builder, body) =>
          out.prettyPrint("for(")
          out.prettyPrint(iterator, indent + 4)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("    ")
          out.prettyPrint(builder, typed = false, indent + 4, shouldIndent = true)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("    ")
          out.prettyPrint(body, typed = false, indent + 4, shouldIndent = true)
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Len(e) =>
          out.prettyPrint("len(")
          out.prettyPrint(e, typed = true, indent + 4, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Lookup(data, key) =>
          out.prettyPrint("lookup(")
          out.prettyPrint(data, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint(key, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Slice(data, index, size) =>
          out.prettyPrint("slice(")
          out.prettyPrint(data, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint(index, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint(size, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Sort(data, keyFunc) =>
          out.prettyPrint("sort(")
          out.prettyPrint(data, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint(keyFunc, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Drain(source, sink) =>
          out.prettyPrint("drain(")
          out.prettyPrint(source, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint(sink, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Serialize(e) =>
          out.prettyPrint("serialize(")
          out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Deserialize(ty, e) =>
          out.prettyPrint("deserialize[")
          out.prettyPrint(ty)
          out.prettyPrint(']')
          out.prettyPrint('(')
          out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case If(cond, onTrue, onFalse) =>
          out.prettyPrint("if (")
          out.prettyPrint(cond, typed = false, indent + 4, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("    ")
          out.prettyPrint(onTrue, typed = true, indent + 4, shouldIndent = true)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("    ")
          out.prettyPrint(onFalse, typed = true, indent + 4, shouldIndent = true)
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Select(cond, onTrue, onFalse) =>
          out.prettyPrint("select(")
          out.prettyPrint(cond, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("  ")
          out.prettyPrint(onTrue, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("  ")
          out.prettyPrint(onFalse, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Iterate(init, updateFunc) =>
          out.prettyPrint("iterate (")
          out.prettyPrint(init, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint("  ")
          out.prettyPrint(updateFunc, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.prettyPrint('\n')
          out.prettyPrint(indentStr)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case MakeStruct(elems) =>
          out.prettyPrint('{')
          for ((e, i) <- elems.view.zipWithIndex) {
            out.prettyPrint(e, typed = false, indent + 1, shouldIndent = false)
            if (i != (elems.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint('}')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case MakeVec(elems) =>
          out.prettyPrint('[')
          for ((e, i) <- elems.view.zipWithIndex) {
            out.prettyPrint(e, typed = false, indent + 1, shouldIndent = false)
            if (i != (elems.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint(']')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Merge(builder, value) =>
          out.prettyPrint("merge(")
          out.prettyPrint(builder, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(',')
          out.prettyPrint(value, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Result(e) =>
          out.prettyPrint("result(")
          out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case NewBuilder(ty, args) =>
          out.prettyPrint(ty)
          if (args.nonEmpty) {
            out.prettyPrint('(')
            for ((p, i) <- args.view.zipWithIndex) {
              out.prettyPrint(p, typed = true, indent + INDENT_INC, shouldIndent = false)
              if (i != (args.length - 1)) {
                out.prettyPrint(',')
              }
            }
            out.prettyPrint(')')
          }
        // don't print type even if requested since it's redundant
        case BinOp(kind, left, right) =>
          if (kind.isInfix) {
            if (typed) {
              out.prettyPrint('(')
            }
            out.prettyPrint(left, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
            out.prettyPrint(kind.symbol)
            out.prettyPrint(right, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
            if (typed) {
              out.prettyPrint(')')
              out.prettyPrint(':')
              out.prettyPrint(expr.ty)
            }
          } else {
            out.prettyPrint(kind.symbol)
            out.prettyPrint('(')
            out.prettyPrint(left, typed = true, indent + 4, shouldIndent = false)
            out.prettyPrint(',')
            out.prettyPrint(right, typed = true, indent + 4, shouldIndent = false)
            out.prettyPrint(')')
            if (typed) {
              out.prettyPrint(':')
              out.prettyPrint(expr.ty)
            }
          }
        case Application(fun, args) =>
          out.prettyPrint('(')
          out.prettyPrint(fun, typed = false, indent, shouldIndent = false)
          out.prettyPrint(')')
          out.prettyPrint('(')
          for ((e, i) <- args.view.zipWithIndex) {
            out.prettyPrint(e, typed = false, indent + INDENT_INC, shouldIndent = false)
            if (i != (args.length - 1)) {
              out.prettyPrint(',')
            }
          }
          out.prettyPrint(')')
          if (typed) {
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
        case Ascription(e, ty) =>
          out.prettyPrint('(')
          out.prettyPrint(e, typed = false, indent + 1, shouldIndent = false)
          out.prettyPrint(')')
          out.prettyPrint(':')
          out.prettyPrint(ty)
        case Projection(struct, index) =>
          if (typed) {
            out.prettyPrint('(')
          }
          out.prettyPrint(struct, typed = false, indent, shouldIndent = false)
          out.prettyPrint('.')
          out.prettyPrint('$')
          out.prettyPrint(index.toString)
          if (typed) {
            out.prettyPrint(')')
            out.prettyPrint(':')
            out.prettyPrint(expr.ty)
          }
      }
    }
  }
}
