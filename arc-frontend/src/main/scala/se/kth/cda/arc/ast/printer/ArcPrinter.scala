package se.kth.cda.arc.ast.printer

import java.io.PrintStream

import se.kth.cda.arc.ast.AST.ExprKind.{Application, Ascription, BinOp, Broadcast, CUDF, Cast, Deserialize, Drain, For, Hash, Ident, If, Iterate, Lambda, Len, Let, Literal, Lookup, MakeStruct, MakeVec, Merge, Negate, NewBuilder, Not, Projection, Result, Select, Serialize, Slice, Sort, ToVec, UnaryOp, Zip}
import se.kth.cda.arc.ast.AST.IterKind.{FringeIter, KeyByIter, NdIter, NextIter, RangeIter, ScalarIter, SimdIter, UnknownIter}
import se.kth.cda.arc.ast.AST.{Expr, ExprKind, Iter, IterKind, Macro, Program, Symbol}
import se.kth.cda.arc.ast.{ASTNode, Type}

object ArcPrinter {

  val INDENT_INC = 2

  implicit class ArcPrintStream(val out: PrintStream) extends AnyVal {

    def printArc(str: String): Unit = out.print(str)

    def printArc(ch: Char): Unit = out.print(ch)

    def printArc(tree: ASTNode): Unit = {
      tree match {
        case Program(macros, expr, _) =>
          macros.foreach { m =>
            out.printArc(m)
            out.printArc('\n')
          }
          out.printArc(expr)
        case Macro(name, params, body, _) =>
          out.printArc("macro ")
          out.printArc(name)
          out.printArc('(')
          for ((p, i) <- params.view.zipWithIndex) {
            out.printArc(p)
            if (i != (params.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc(')')
          out.printArc('=')
          out.printArc(body)
        case e: Expr   => out.printArc(e, typed = true, 0, shouldIndent = true)
        case s: Symbol => out.printArc(s)
      }
    }

    def printArc(s: Symbol): Unit = out.printArc(s.name)

    def printArc(t: Type): Unit = out.printArc(t.render)

    def printArc(iter: Iter, indent: Int): Unit = {
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
        out.printArc(iterStr)
        out.printArc('(')
        out.printArc(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.start.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.shape.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.strides.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(')')
      } else if (iter.kind == KeyByIter) {
        out.printArc(iterStr)
        out.printArc('(')
        out.printArc(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.keyFunc.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(')')
      } else if (iter.start.isDefined) {
        out.printArc(iterStr)
        out.printArc('(')
        out.printArc(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.start.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.end.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(',')
        out.printArc(iter.stride.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(')')
      } else if (iter.kind != ScalarIter) {
        out.printArc(iterStr)
        out.printArc('(')
        out.printArc(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printArc(')')
      } else {
        out.printArc(iter.data, typed = true, indent, shouldIndent = false)
      }
    }

    def printArc(expr: Expr, typed: Boolean, indent: Int, shouldIndent: Boolean): Unit = {
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
            out.printArc('(')
            out.printArc(' ')
          }
          out.printArc("let ")
          out.printArc(name)
          out.printArc(':')
          out.printArc(bindingTy)
          out.printArc('=')
          out.printArc(value, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printArc(';')
          out.printArc('\n')
          out.printArc(indentStr)
          if (typed) {
            out.printArc("  ")
          }
          out.printArc(body, typed = false, if (typed) indent + INDENT_INC else indent, shouldIndent = true)
          if (typed) {
            out.printArc('\n')
            out.printArc(indentStr)
            out.printArc(')')
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Lambda(params, body) =>
          if (params.isEmpty) {
            out.printArc("||")
            out.printArc(body, typed = true, indent + INDENT_INC, shouldIndent = true)
          } else {
            out.printArc('|')
            for ((p, i) <- params.view.zipWithIndex) {
              out.printArc(p.symbol)
              out.printArc(':')
              out.printArc(p.ty)
              if (i != (params.length - 1)) {
                out.printArc(',')
              }
            }
            out.printArc('|')
            out.printArc('\n')
            out.printArc(indentStr)
            out.printArc("  ")
            out.printArc(body, typed = true, indent + INDENT_INC, shouldIndent = true)
          }
        case Negate(e) =>
          out.printArc('-')
          out.printArc(e, typed = true, indent + 1, shouldIndent = false)
        case Not(e) =>
          out.printArc('!')
          out.printArc(e, typed = true, indent + 1, shouldIndent = false)
        case UnaryOp(kind, e) =>
          out.printArc(kind.toString)
          out.printArc('(')
          out.printArc(e, typed = false, indent + 1, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(e.ty)
          }
        case Ident(s) =>
          out.printArc(s)
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case l: Literal[_] =>
          out.printArc(l.raw)
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Cast(ty, e) =>
          out.printArc(ty)
          out.printArc('(')
          out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
        case ToVec(e) =>
          out.printArc("tovec(")
          out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
        case Broadcast(e) =>
          out.printArc("broadcast(")
          out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
        case CUDF(ref, args, retT) =>
          out.printArc("cudf[")
          ref match {
            case Left(name) => out.printArc(name);
            case Right(pointer) =>
              out.printArc('*')
              out.printArc(pointer, typed = false, indent + INDENT_INC, shouldIndent = false)
          }
          out.printArc(',')
          out.printArc(retT)
          out.printArc(']')
          out.printArc('(')
          for ((e, i) <- args.view.zipWithIndex) {
            out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
            if (i != (args.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Zip(params) =>
          out.printArc("zip(")
          for ((e, i) <- params.view.zipWithIndex) {
            out.printArc(e, typed = false, indent + 4, shouldIndent = false)
            if (i != (params.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Hash(params) =>
          out.printArc("hash(")
          for ((e, i) <- params.view.zipWithIndex) {
            out.printArc(e, typed = false, indent + 4, shouldIndent = false)
            if (i != (params.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case For(iterator, builder, body) =>
          out.printArc("for(")
          out.printArc(iterator, indent + 4)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("    ")
          out.printArc(builder, typed = false, indent + 4, shouldIndent = true)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("    ")
          out.printArc(body, typed = false, indent + 4, shouldIndent = true)
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Len(e) =>
          out.printArc("len(")
          out.printArc(e, typed = true, indent + 4, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Lookup(data, key) =>
          out.printArc("lookup(")
          out.printArc(data, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc(key, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Slice(data, index, size) =>
          out.printArc("slice(")
          out.printArc(data, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc(index, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc(size, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Sort(data, keyFunc) =>
          out.printArc("sort(")
          out.printArc(data, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc(keyFunc, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Drain(source, sink) =>
          out.printArc("drain(")
          out.printArc(source, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc(sink, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Serialize(e) =>
          out.printArc("serialize(")
          out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Deserialize(ty, e) =>
          out.printArc("deserialize[")
          out.printArc(ty)
          out.printArc(']')
          out.printArc('(')
          out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case If(cond, onTrue, onFalse) =>
          out.printArc("if (")
          out.printArc(cond, typed = false, indent + 4, shouldIndent = false)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("    ")
          out.printArc(onTrue, typed = true, indent + 4, shouldIndent = true)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("    ")
          out.printArc(onFalse, typed = true, indent + 4, shouldIndent = true)
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Select(cond, onTrue, onFalse) =>
          out.printArc("select(")
          out.printArc(cond, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("  ")
          out.printArc(onTrue, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("  ")
          out.printArc(onFalse, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Iterate(init, updateFunc) =>
          out.printArc("iterate (")
          out.printArc(init, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc("  ")
          out.printArc(updateFunc, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printArc('\n')
          out.printArc(indentStr)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case MakeStruct(elems) =>
          out.printArc('{')
          for ((e, i) <- elems.view.zipWithIndex) {
            out.printArc(e, typed = false, indent + 1, shouldIndent = false)
            if (i != (elems.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc('}')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case MakeVec(elems) =>
          out.printArc('[')
          for ((e, i) <- elems.view.zipWithIndex) {
            out.printArc(e, typed = false, indent + 1, shouldIndent = false)
            if (i != (elems.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc(']')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Merge(builder, value) =>
          out.printArc("merge(")
          out.printArc(builder, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(',')
          out.printArc(value, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Result(e) =>
          out.printArc("result(")
          out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case NewBuilder(ty, args) =>
          out.printArc(ty)
          if (args.nonEmpty) {
            out.printArc('(')
            for ((p, i) <- args.view.zipWithIndex) {
              out.printArc(p, typed = true, indent + INDENT_INC, shouldIndent = false)
              if (i != (args.length - 1)) {
                out.printArc(',')
              }
            }
            out.printArc(')')
          }
        // don't print type even if requested since it's redundant
        case BinOp(kind, left, right) =>
          if (kind.isInfix) {
            if (typed) {
              out.printArc('(')
            }
            out.printArc(left, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
            out.printArc(kind.symbol)
            out.printArc(right, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
            if (typed) {
              out.printArc(')')
              out.printArc(':')
              out.printArc(expr.ty)
            }
          } else {
            out.printArc(kind.symbol)
            out.printArc('(')
            out.printArc(left, typed = true, indent + 4, shouldIndent = false)
            out.printArc(',')
            out.printArc(right, typed = true, indent + 4, shouldIndent = false)
            out.printArc(')')
            if (typed) {
              out.printArc(':')
              out.printArc(expr.ty)
            }
          }
        case Application(fun, args) =>
          out.printArc('(')
          out.printArc(fun, typed = false, indent, shouldIndent = false)
          out.printArc(')')
          out.printArc('(')
          for ((e, i) <- args.view.zipWithIndex) {
            out.printArc(e, typed = false, indent + INDENT_INC, shouldIndent = false)
            if (i != (args.length - 1)) {
              out.printArc(',')
            }
          }
          out.printArc(')')
          if (typed) {
            out.printArc(':')
            out.printArc(expr.ty)
          }
        case Ascription(e, ty) =>
          out.printArc('(')
          out.printArc(e, typed = false, indent + 1, shouldIndent = false)
          out.printArc(')')
          out.printArc(':')
          out.printArc(ty)
        case Projection(struct, index) =>
          if (typed) {
            out.printArc('(')
          }
          out.printArc(struct, typed = false, indent, shouldIndent = false)
          out.printArc('.')
          out.printArc('$')
          out.printArc(index.toString)
          if (typed) {
            out.printArc(')')
            out.printArc(':')
            out.printArc(expr.ty)
          }
      }
    }
  }
}
