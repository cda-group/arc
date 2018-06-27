package se.kth.cda.arc

import java.io.PrintStream

object PrettyPrint {
  import AST._;

  val INDENT_INC = 2;

  def print(tree: ASTNode): String = {
    val sb = new Utils.StringBuilderStream();
    this.print(tree, sb.asPrintStream());
    sb.result()
  }
  def print(tree: ASTNode, out: PrintStream): Unit = {
    tree match {
      case Program(macros, expr, _) => {
        macros.foreach { m =>
          print(m, out);
          out.append('\n');
        }
        print(expr, out);
      }
      case Macro(name, params, body, _) => {
        out.append("macro ");
        printSymbol(name, out);
        out.append('(');
        for ((p, i) <- params.view.zipWithIndex) {
          printSymbol(p, out);
          if (i != (params.length - 1)) {
            out.append(',');
          }
        }
        out.append(')');
        out.append('=');
        print(body);
      }
      case e: Expr   => printExpr(e, out, true, 0, true)
      case s: Symbol => printSymbol(s, out)
    }
  }

  def printSymbol(s: Symbol, out: PrintStream): Unit = out.append(s.text);
  def printType(t: Type, out: PrintStream): Unit = out.append(t.render);
  def printIter(iter: Iter, out: PrintStream, indent: Int): Unit = {
    import IterKind._;

    val iterStr = (iter.kind match {
      case ScalarIter  => ""
      case SimdIter    => "simd"
      case FringeIter  => "fringe"
      case NdIter      => "nd"
      case RangeIter   => "range"
      case NextIter    => "next"
      case UnknownIter => "?"
    }) + "iter";

    if (iter.kind == NdIter) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, true, indent + INDENT_INC, false);
      out.append(',');
      printExpr(iter.start.get, out, true, indent + INDENT_INC, false);
      out.append(',');
      printExpr(iter.shape.get, out, true, indent + INDENT_INC, false);
      out.append(',');
      printExpr(iter.strides.get, out, true, indent + INDENT_INC, false);
      out.append(')');
    } else if (iter.start.isDefined) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, true, indent + INDENT_INC, false);
      out.append(',');
      printExpr(iter.start.get, out, true, indent + INDENT_INC, false);
      out.append(',');
      printExpr(iter.end.get, out, true, indent + INDENT_INC, false);
      out.append(',');
      printExpr(iter.stride.get, out, true, indent + INDENT_INC, false);
      out.append(')');
    } else if (iter.kind != ScalarIter) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, true, indent + INDENT_INC, false);
      out.append(')');
    } else {
      printExpr(iter.data, out, true, indent, false);
    }
  }
  def printExpr(expr: Expr, out: PrintStream, typed: Boolean, indent: Int, shouldIndent: Boolean): Unit = {
    import ExprKind._;
    lazy val indentStr = (0 until indent).foldLeft("")((acc, _) => acc + " ");
    lazy val lessIndentStr = (0 until (indent - 2)).foldLeft("")((acc, _) => acc + " ");
    expr.kind match {
      case Let(name, value, body) => {
        if (typed) {
          out.append('(');
          out.append(' ');
        }
        out.append("let ");
        printSymbol(name, out);
        out.append(':');
        printType(value.ty, out);
        out.append('=');
        printExpr(value, out, true, indent + INDENT_INC, true);
        out.append(';');
        out.append('\n');
        out.append(indentStr);
        if (typed) {
          out.append("  ");
        }
        printExpr(body, out, false, if (typed) indent + INDENT_INC else indent, true);
        if (typed) {
          out.append('\n');
          out.append(lessIndentStr);
          out.append(')');
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Lambda(params, body) => {
        if (params.isEmpty) {
          out.append("||");
          printExpr(body, out, true, indent + INDENT_INC, true);
        } else {
          out.append('|');
          for ((p, i) <- params.view.zipWithIndex) {
            printSymbol(p.name, out);
            out.append(':');
            printType(p.ty, out);
            if (i != (params.length - 1)) {
              out.append(',');
            }
          }
          out.append('|');
          out.append('\n');
          out.append(indentStr);
          out.append("  ");
          printExpr(body, out, true, indent + INDENT_INC, true);
        }
      }
      case Negate(expr: Expr) => {
        out.append('-');
        printExpr(expr, out, true, indent + 1, false);
      }
      case Not(expr: Expr) => {
        out.append('!');
        printExpr(expr, out, true, indent + 1, false);
      }
      case UnaryOp(kind: UnaryOpKind.UnaryOpKind, expr: Expr) => {
        out.append(UnaryOpKind.print(kind));
        out.append('(');
        printExpr(expr, out, false, indent + 1, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Ident(s) => {
        printSymbol(s, out);
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case l: Literal[_] => {
        out.append(l.raw)
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Cast(ty, e) => {
        printType(ty, out);
        out.append('(');
        printExpr(e, out, false, indent + INDENT_INC, false);
        out.append(')');
      }
      case ToVec(e) => {
        out.append("tovec(");
        printExpr(e, out, false, indent + INDENT_INC, false);
        out.append(')');
      }
      case Broadcast(e) => {
        out.append("broadcast(");
        printExpr(e, out, false, indent + INDENT_INC, false);
        out.append(')');
      }
      case CUDF(ref, args, retT) => {
        out.append("cudf[");
        ref match {
          case Left(name) => printSymbol(name, out);
          case Right(pointer) => {
            out.append('*');
            printExpr(pointer, out, false, indent + INDENT_INC, false);
          }
        }
        out.append(',');
        printType(retT, out);
        out.append(']');
        out.append('(');
        for ((e, i) <- args.view.zipWithIndex) {
          printExpr(e, out, false, indent + INDENT_INC, false);
          if (i != (args.length - 1)) {
            out.append(',');
          }
        }
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Zip(params) => {
        out.append("zip(");
        for ((e, i) <- params.view.zipWithIndex) {
          printExpr(e, out, false, indent + 4, false);
          if (i != (params.length - 1)) {
            out.append(',');
          }
        }
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case For(iterator, builder, body) => {
        out.append("for(");
        printIter(iterator, out, indent + 4);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(builder, out, false, indent + 4, true);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(body, out, false, indent + 4, true);
        out.append('\n');
        out.append(indentStr);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Len(e) => {
        out.append("len(");
        printExpr(e, out, true, indent + 4, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Lookup(data, key) => {
        out.append("lookup(");
        printExpr(data, out, true, indent + INDENT_INC, false);
        out.append(',');
        printExpr(key, out, true, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Slice(data, index, size) => {
        out.append("slice(");
        printExpr(data, out, false, indent + INDENT_INC, false);
        out.append(',');
        printExpr(index, out, false, indent + INDENT_INC, false);
        out.append(',');
        printExpr(size, out, false, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Sort(data, keyFunc) => {
        out.append("sort(");
        printExpr(data, out, false, indent + INDENT_INC, false);
        out.append(',');
        printExpr(keyFunc, out, false, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Serialize(e) => {
        out.append("serialize(");
        printExpr(e, out, false, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Deserialize(ty, e) => {
        out.append("deserialize[");
        printType(ty, out);
        out.append(']');
        out.append('(');
        printExpr(e, out, false, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case If(cond, onTrue, onFalse) => {
        out.append("if (");
        printExpr(cond, out, false, indent + 4, false);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(onTrue, out, true, indent + 4, true);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(onFalse, out, true, indent + 4, true);
        out.append('\n');
        out.append(indentStr);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Select(cond, onTrue, onFalse) => {
        out.append("select(");
        printExpr(cond, out, false, indent + INDENT_INC, false);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("  ");
        printExpr(onTrue, out, true, indent + INDENT_INC, true);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("  ");
        printExpr(onFalse, out, true, indent + INDENT_INC, true);
        out.append('\n');
        out.append(indentStr);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Iterate(init, updateFunc) => {
        out.append("iterate (");
        printExpr(init, out, true, indent + INDENT_INC, false);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("  ");
        printExpr(updateFunc, out, true, indent + INDENT_INC, true);
        out.append('\n');
        out.append(indentStr);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case MakeStruct(elems) => {
        out.append('{');
        for ((e, i) <- elems.view.zipWithIndex) {
          printExpr(e, out, false, indent + 1, false);
          if (i != (elems.length - 1)) {
            out.append(',');
          }
        }
        out.append('}');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case MakeVec(elems) => {
        out.append('[');
        for ((e, i) <- elems.view.zipWithIndex) {
          printExpr(e, out, false, indent + 1, false);
          if (i != (elems.length - 1)) {
            out.append(',');
          }
        }
        out.append(']');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Merge(builder, value) => {
        out.append("merge(");
        printExpr(builder, out, false, indent + INDENT_INC, false);
        out.append(',');
        printExpr(value, out, false, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Result(e) => {
        out.append("result(");
        printExpr(e, out, false, indent + INDENT_INC, false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case NewBuilder(ty, Some(arg)) => {
        printType(ty, out);
        out.append('(');
        printExpr(arg, out, true, indent + INDENT_INC, false);
        out.append(')');
        // don't print type even if requested since it's redundant
      }
      case NewBuilder(ty, None) => {
        printType(ty, out);
        // don't print type even if requested since it's redundant
      }
      case BinOp(kind, left, right) => {
        if (kind.isInfix) {
          if (typed) {
            out.append('(');
          }
          printExpr(left, out, true, if (typed) indent + 1 else indent, false);
          out.append(kind.symbol);
          printExpr(right, out, true, if (typed) indent + 1 else indent, false);
          if (typed) {
            out.append(')');
            out.append(':');
            printType(expr.ty, out);
          }
        } else {
          out.append(kind.symbol);
          out.append('(');
          printExpr(left, out, true, indent + 4, false);
          out.append(',');
          printExpr(right, out, true, indent + 4, false);
          out.append(')');
          if (typed) {
            out.append(':');
            printType(expr.ty, out);
          }
        }
      }
      case Application(fun, args) => {
        printExpr(fun, out, false, indent, false);
        out.append('(');
        for ((e, i) <- args.view.zipWithIndex) {
          printExpr(e, out, false, indent + INDENT_INC, false);
          if (i != (args.length - 1)) {
            out.append(',');
          }
        }
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Ascription(e, ty) => {
        out.append('(');
        printExpr(e, out, false, indent + 1, false);
        out.append(')');
        out.append(':');
        printType(ty, out);
      }
      case Projection(struct, index) => {
        if (typed) {
          out.append('(');
        }
        printExpr(struct, out, false, indent, false);
        out.append('.');
        out.append('$');
        out.append(index.toString);
        if (typed) {
          out.append(')');
          out.append(':');
          printType(expr.ty, out);
        }
      }
    }
  }
}
