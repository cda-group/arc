package se.kth.cda.arc.ast.printer

import java.io.PrintStream

import se.kth.cda.arc.ast.AST.ExprKind.{Application, Ascription, BinOp, Broadcast, CUDF, Cast, Deserialize, Drain, For, Hash, Ident, If, Iterate, Lambda, Len, Let, Literal, Lookup, MakeStruct, MakeVec, Merge, Negate, NewBuilder, Not, Projection, Result, Select, Serialize, Slice, Sort, ToVec, UnaryOp, Zip}
import se.kth.cda.arc.ast.AST.IterKind.{FringeIter, KeyByIter, NdIter, NextIter, RangeIter, ScalarIter, SimdIter, UnknownIter}
import se.kth.cda.arc.ast.AST.{Expr, ExprKind, Iter, IterKind, Macro, Program, Symbol}
import se.kth.cda.arc.ast.{ASTNode, Type}

object MLIRPrinter {

  val INDENT_INC = 2

  implicit class MLIRPrintStream(val out: PrintStream) extends AnyVal {

    def printMLIR(str: String): Unit = out.print(str)

    def printMLIR(ch: Char): Unit = out.print(ch)

    def printMLIR(tree: ASTNode): Unit = {
      tree match {
        case Program(macros, expr, _) =>
          macros.foreach { m =>
            out.printMLIR(m)
            out.printMLIR('\n')
          }
          out.printMLIR(expr)
        case Macro(name, params, body, _) =>
          out.printMLIR("macro ")
          out.printMLIR(name)
          out.printMLIR('(')
          for ((p, i) <- params.view.zipWithIndex) {
            out.printMLIR(p)
            if (i != (params.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR(')')
          out.printMLIR('=')
          out.printMLIR(body)
        case e: Expr   => out.printMLIR(e, typed = true, 0, shouldIndent = true)
        case s: Symbol => out.printMLIR(s)
      }
    }

    def printMLIR(s: Symbol): Unit = out.printMLIR(s.name)

    def printMLIR(t: Type): Unit = out.printMLIR(t.render)

    def printMLIR(iter: Iter, indent: Int): Unit = {
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
        out.printMLIR(iterStr)
        out.printMLIR('(')
        out.printMLIR(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.start.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.shape.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.strides.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(')')
      } else if (iter.kind == KeyByIter) {
        out.printMLIR(iterStr)
        out.printMLIR('(')
        out.printMLIR(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.keyFunc.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(')')
      } else if (iter.start.isDefined) {
        out.printMLIR(iterStr)
        out.printMLIR('(')
        out.printMLIR(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.start.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.end.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(',')
        out.printMLIR(iter.stride.get, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(')')
      } else if (iter.kind != ScalarIter) {
        out.printMLIR(iterStr)
        out.printMLIR('(')
        out.printMLIR(iter.data, typed = true, indent + INDENT_INC, shouldIndent = false)
        out.printMLIR(')')
      } else {
        out.printMLIR(iter.data, typed = true, indent, shouldIndent = false)
      }
    }

    def printMLIR(expr: Expr, typed: Boolean, indent: Int, shouldIndent: Boolean): Unit = {
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
            out.printMLIR('(')
            out.printMLIR(' ')
          }
          out.printMLIR("let ")
          out.printMLIR(name)
          out.printMLIR(':')
          out.printMLIR(bindingTy)
          out.printMLIR('=')
          out.printMLIR(value, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printMLIR(';')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          if (typed) {
            out.printMLIR("  ")
          }
          out.printMLIR(body, typed = false, if (typed) indent + INDENT_INC else indent, shouldIndent = true)
          if (typed) {
            out.printMLIR('\n')
            out.printMLIR(indentStr)
            out.printMLIR(')')
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Lambda(params, body) =>
          if (params.isEmpty) {
            out.printMLIR("||")
            out.printMLIR(body, typed = true, indent + INDENT_INC, shouldIndent = true)
          } else {
            out.printMLIR('|')
            for ((p, i) <- params.view.zipWithIndex) {
              out.printMLIR(p.symbol)
              out.printMLIR(':')
              out.printMLIR(p.ty)
              if (i != (params.length - 1)) {
                out.printMLIR(',')
              }
            }
            out.printMLIR('|')
            out.printMLIR('\n')
            out.printMLIR(indentStr)
            out.printMLIR("  ")
            out.printMLIR(body, typed = true, indent + INDENT_INC, shouldIndent = true)
          }
        case Negate(e) =>
          out.printMLIR('-')
          out.printMLIR(e, typed = true, indent + 1, shouldIndent = false)
        case Not(e) =>
          out.printMLIR('!')
          out.printMLIR(e, typed = true, indent + 1, shouldIndent = false)
        case UnaryOp(kind, e) =>
          out.printMLIR(kind.toString)
          out.printMLIR('(')
          out.printMLIR(e, typed = false, indent + 1, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(e.ty)
          }
        case Ident(s) =>
          out.printMLIR(s)
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case l: Literal[_] =>
          out.printMLIR(l.raw)
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Cast(ty, e) =>
          out.printMLIR(ty)
          out.printMLIR('(')
          out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
        case ToVec(e) =>
          out.printMLIR("tovec(")
          out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
        case Broadcast(e) =>
          out.printMLIR("broadcast(")
          out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
        case CUDF(ref, args, retT) =>
          out.printMLIR("cudf[")
          ref match {
            case Left(name) => out.printMLIR(name);
            case Right(pointer) =>
              out.printMLIR('*')
              out.printMLIR(pointer, typed = false, indent + INDENT_INC, shouldIndent = false)
          }
          out.printMLIR(',')
          out.printMLIR(retT)
          out.printMLIR(']')
          out.printMLIR('(')
          for ((e, i) <- args.view.zipWithIndex) {
            out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
            if (i != (args.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Zip(params) =>
          out.printMLIR("zip(")
          for ((e, i) <- params.view.zipWithIndex) {
            out.printMLIR(e, typed = false, indent + 4, shouldIndent = false)
            if (i != (params.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Hash(params) =>
          out.printMLIR("hash(")
          for ((e, i) <- params.view.zipWithIndex) {
            out.printMLIR(e, typed = false, indent + 4, shouldIndent = false)
            if (i != (params.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case For(iterator, builder, body) =>
          out.printMLIR("for(")
          out.printMLIR(iterator, indent + 4)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("    ")
          out.printMLIR(builder, typed = false, indent + 4, shouldIndent = true)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("    ")
          out.printMLIR(body, typed = false, indent + 4, shouldIndent = true)
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Len(e) =>
          out.printMLIR("len(")
          out.printMLIR(e, typed = true, indent + 4, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Lookup(data, key) =>
          out.printMLIR("lookup(")
          out.printMLIR(data, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR(key, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Slice(data, index, size) =>
          out.printMLIR("slice(")
          out.printMLIR(data, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR(index, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR(size, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Sort(data, keyFunc) =>
          out.printMLIR("sort(")
          out.printMLIR(data, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR(keyFunc, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Drain(source, sink) =>
          out.printMLIR("drain(")
          out.printMLIR(source, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR(sink, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Serialize(e) =>
          out.printMLIR("serialize(")
          out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Deserialize(ty, e) =>
          out.printMLIR("deserialize[")
          out.printMLIR(ty)
          out.printMLIR(']')
          out.printMLIR('(')
          out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case If(cond, onTrue, onFalse) =>
          out.printMLIR("if (")
          out.printMLIR(cond, typed = false, indent + 4, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("    ")
          out.printMLIR(onTrue, typed = true, indent + 4, shouldIndent = true)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("    ")
          out.printMLIR(onFalse, typed = true, indent + 4, shouldIndent = true)
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Select(cond, onTrue, onFalse) =>
          out.printMLIR("select(")
          out.printMLIR(cond, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("  ")
          out.printMLIR(onTrue, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("  ")
          out.printMLIR(onFalse, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Iterate(init, updateFunc) =>
          out.printMLIR("iterate (")
          out.printMLIR(init, typed = true, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR("  ")
          out.printMLIR(updateFunc, typed = true, indent + INDENT_INC, shouldIndent = true)
          out.printMLIR('\n')
          out.printMLIR(indentStr)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case MakeStruct(elems) =>
          out.printMLIR('{')
          for ((e, i) <- elems.view.zipWithIndex) {
            out.printMLIR(e, typed = false, indent + 1, shouldIndent = false)
            if (i != (elems.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR('}')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case MakeVec(elems) =>
          out.printMLIR('[')
          for ((e, i) <- elems.view.zipWithIndex) {
            out.printMLIR(e, typed = false, indent + 1, shouldIndent = false)
            if (i != (elems.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR(']')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Merge(builder, value) =>
          out.printMLIR("merge(")
          out.printMLIR(builder, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(',')
          out.printMLIR(value, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Result(e) =>
          out.printMLIR("result(")
          out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case NewBuilder(ty, args) =>
          out.printMLIR(ty)
          if (args.nonEmpty) {
            out.printMLIR('(')
            for ((p, i) <- args.view.zipWithIndex) {
              out.printMLIR(p, typed = true, indent + INDENT_INC, shouldIndent = false)
              if (i != (args.length - 1)) {
                out.printMLIR(',')
              }
            }
            out.printMLIR(')')
          }
        // don't print type even if requested since it's redundant
        case BinOp(kind, left, right) =>
          if (kind.isInfix) {
            if (typed) {
              out.printMLIR('(')
            }
            out.printMLIR(left, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
            out.printMLIR(kind.symbol)
            out.printMLIR(right, typed = true, if (typed) indent + 1 else indent, shouldIndent = false)
            if (typed) {
              out.printMLIR(')')
              out.printMLIR(':')
              out.printMLIR(expr.ty)
            }
          } else {
            out.printMLIR(kind.symbol)
            out.printMLIR('(')
            out.printMLIR(left, typed = true, indent + 4, shouldIndent = false)
            out.printMLIR(',')
            out.printMLIR(right, typed = true, indent + 4, shouldIndent = false)
            out.printMLIR(')')
            if (typed) {
              out.printMLIR(':')
              out.printMLIR(expr.ty)
            }
          }
        case Application(fun, args) =>
          out.printMLIR('(')
          out.printMLIR(fun, typed = false, indent, shouldIndent = false)
          out.printMLIR(')')
          out.printMLIR('(')
          for ((e, i) <- args.view.zipWithIndex) {
            out.printMLIR(e, typed = false, indent + INDENT_INC, shouldIndent = false)
            if (i != (args.length - 1)) {
              out.printMLIR(',')
            }
          }
          out.printMLIR(')')
          if (typed) {
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
        case Ascription(e, ty) =>
          out.printMLIR('(')
          out.printMLIR(e, typed = false, indent + 1, shouldIndent = false)
          out.printMLIR(')')
          out.printMLIR(':')
          out.printMLIR(ty)
        case Projection(struct, index) =>
          if (typed) {
            out.printMLIR('(')
          }
          out.printMLIR(struct, typed = false, indent, shouldIndent = false)
          out.printMLIR('.')
          out.printMLIR('$')
          out.printMLIR(index.toString)
          if (typed) {
            out.printMLIR(')')
            out.printMLIR(':')
            out.printMLIR(expr.ty)
          }
      }
    }
  }
}
