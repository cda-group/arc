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
      case e: Expr   => printExpr(e, out, typed = true, 0, shouldIndent = true)
      case s: Symbol => printSymbol(s, out)
    }
  }

  def printSymbol(s: Symbol, out: PrintStream): Unit = out.append(s.text);
  def printType(t: Type, out: PrintStream): Unit = out.append(t.render);

  def printIter(iter: Iter, out: PrintStream, indent: Int): Unit = {
    import IterKind._;

    val iterStr = iter.kind match {
      case ScalarIter  => "iter"
      case SimdIter    => "simditer"
      case FringeIter  => "fringeiter"
      case NdIter      => "nditer"
      case RangeIter   => "rangeiter"
      case NextIter    => "nextiter"
      case KeyByIter   => "keyby"
      case UnknownIter => "?iter"
    };

    if (iter.kind == NdIter) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.start.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.shape.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.strides.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(')');
    } else if (iter.kind == KeyByIter) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.keyFunc.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(')');
    } else if (iter.start.isDefined) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.start.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.end.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(',');
      printExpr(iter.stride.get, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(')');
    } else if (iter.kind != ScalarIter) {
      out.append(iterStr);
      out.append('(');
      printExpr(iter.data, out, typed = true, indent + INDENT_INC, shouldIndent = false);
      out.append(')');
    } else {
      printExpr(iter.data, out, typed = true, indent, shouldIndent = false);
    }
  }

  def printExpr(expr: Expr, out: PrintStream, typed: Boolean, indent: Int, shouldIndent: Boolean): Unit = {
    import ExprKind._;
    lazy val indentStr = (0 until indent).foldLeft("")((acc, _) => acc + " ");
    lazy val lessIndentStr = (0 until (indent - 2)).foldLeft("")((acc, _) => acc + " ");
    expr.kind match {
      case Let(name, bindingTy, value, body) => {
        if (typed) {
          out.append('(');
          out.append(' ');
        }
        out.append("let ");
        printSymbol(name, out);
        out.append(':');
        printType(bindingTy, out);
        out.append('=');
        printExpr(value, out, typed = true, indent + INDENT_INC, shouldIndent = true);
        out.append(';');
        out.append('\n');
        out.append(indentStr);
        if (typed) {
          out.append("  ");
        }
        printExpr(body, out, typed = false, if (typed) indent + INDENT_INC else indent, shouldIndent = true);
        if (typed) {
          out.append('\n');
          out.append(indentStr);
          out.append(')');
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Lambda(params, body) => {
        if (params.isEmpty) {
          out.append("||");
          printExpr(body, out, typed = true, indent + INDENT_INC, shouldIndent = true);
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
          printExpr(body, out, typed = true, indent + INDENT_INC, shouldIndent = true);
        }
      }
      case Negate(expr) => {
        out.append('-');
        printExpr(expr, out, typed = true, indent + 1, shouldIndent = false);
      }
      case Not(expr) => {
        out.append('!');
        printExpr(expr, out, typed = true, indent + 1, shouldIndent = false);
      }
      case UnaryOp(kind, expr) => {
        out.append(UnaryOpKind.print(kind));
        out.append('(');
        printExpr(expr, out, typed = false, indent + 1, shouldIndent = false);
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
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
      }
      case ToVec(e) => {
        out.append("tovec(");
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
      }
      case Broadcast(e) => {
        out.append("broadcast(");
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
      }
      case CUDF(ref, args, retT) => {
        out.append("cudf[");
        ref match {
          case Left(name) => printSymbol(name, out);
          case Right(pointer) => {
            out.append('*');
            printExpr(pointer, out, typed = false, indent + INDENT_INC, shouldIndent = false);
          }
        }
        out.append(',');
        printType(retT, out);
        out.append(']');
        out.append('(');
        for ((e, i) <- args.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
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
          printExpr(e, out, typed = false, indent + 4, shouldIndent = false);
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
      case Hash(params) => {
        out.append("hash(");
        for ((e, i) <- params.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + 4, shouldIndent = false);
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
        printExpr(builder, out, typed = false, indent + 4, shouldIndent = true);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(body, out, typed = false, indent + 4, shouldIndent = true);
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
        printExpr(e, out, typed = true, indent + 4, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Lookup(data, key) => {
        out.append("lookup(");
        printExpr(data, out, typed = true, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        printExpr(key, out, typed = true, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Slice(data, index, size) => {
        out.append("slice(");
        printExpr(data, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        printExpr(index, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        printExpr(size, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Sort(data, keyFunc) => {
        out.append("sort(");
        printExpr(data, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        printExpr(keyFunc, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Serialize(e) => {
        out.append("serialize(");
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
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
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case If(cond, onTrue, onFalse) => {
        out.append("if (");
        printExpr(cond, out, typed = false, indent + 4, shouldIndent = false);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(onTrue, out, typed = true, indent + 4, shouldIndent = true);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("    ");
        printExpr(onFalse, out, typed = true, indent + 4, shouldIndent = true);
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
        printExpr(cond, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("  ");
        printExpr(onTrue, out, typed = true, indent + INDENT_INC, shouldIndent = true);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("  ");
        printExpr(onFalse, out, typed = true, indent + INDENT_INC, shouldIndent = true);
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
        printExpr(init, out, typed = true, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        out.append('\n');
        out.append(indentStr);
        out.append("  ");
        printExpr(updateFunc, out, typed = true, indent + INDENT_INC, shouldIndent = true);
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
          printExpr(e, out, typed = false, indent + 1, shouldIndent = false);
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
          printExpr(e, out, typed = false, indent + 1, shouldIndent = false);
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
        printExpr(builder, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(',');
        printExpr(value, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case Result(e) => {
        out.append("result(");
        printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
        out.append(')');
        if (typed) {
          out.append(':');
          printType(expr.ty, out);
        }
      }
      case NewBuilder(ty, Some(arg)) => {
        printType(ty, out);
        out.append('(');
        printExpr(arg, out, typed = true, indent + INDENT_INC, shouldIndent = false);
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
          printExpr(left, out, typed = true, if (typed) indent + 1 else indent, shouldIndent = false);
          out.append(kind.symbol);
          printExpr(right, out, typed = true, if (typed) indent + 1 else indent, shouldIndent = false);
          if (typed) {
            out.append(')');
            out.append(':');
            printType(expr.ty, out);
          }
        } else {
          out.append(kind.symbol);
          out.append('(');
          printExpr(left, out, typed = true, indent + 4, shouldIndent = false);
          out.append(',');
          printExpr(right, out, typed = true, indent + 4, shouldIndent = false);
          out.append(')');
          if (typed) {
            out.append(':');
            printType(expr.ty, out);
          }
        }
      }
      case Application(fun, args) => {
        printExpr(fun, out, typed = false, indent, shouldIndent = false);
        out.append('(');
        for ((e, i) <- args.view.zipWithIndex) {
          printExpr(e, out, typed = false, indent + INDENT_INC, shouldIndent = false);
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
        printExpr(e, out, typed = false, indent + 1, shouldIndent = false);
        out.append(')');
        out.append(':');
        printType(ty, out);
      }
      case Projection(struct, index) => {
        if (typed) {
          out.append('(');
        }
        printExpr(struct, out, typed = false, indent, shouldIndent = false);
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
