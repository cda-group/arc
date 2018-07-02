package se.kth.cda.arc.transform

import se.kth.cda.arc.Arc
import se.kth.cda.arc.AST._
import scala.util.{ Try, Success, Failure }
import scala.collection.mutable

object MacroExpansion {
  lazy val standardMacros: List[Macro] = {
    val stream = getClass.getResourceAsStream("/standard_macros.weld");
    if (stream == null) {
      throw new java.io.FileNotFoundException("standard_macros.weld");
    }
    Arc.macros(stream) match {
      case Success(macros) => macros
      case Failure(e)      => throw e
    }
  };

  def expand(prog: Program): Try[Expr] = {
    this.expand(standardMacros ++ prog.macros, prog.expr)
  }

  def expand(expr: Expr): Try[Expr] = {
    this.expand(standardMacros, expr)
  }

  private def expand(macros: List[Macro], expr: Expr): Try[Expr] = {
    val macroEnv = mutable.Map.empty[String, Macro];
    macros.foreach { m =>
      val name = m.name.text;
      if (macroEnv.contains(name)) {
        return Failure(new MacroException(s" Macro names must be unique! There are two entries for macro $name."))
      } else {
        macroEnv += (name -> m)
      }
    }
    expand(macroEnv.toMap, expr)
  }

  private def expand(macroEnv: Map[String, Macro], expr: Expr): Try[Expr] = {
    val t = Transformer.exprTransformer[Map[String, Macro]](macroTransform);
    var lastExpr = expr;
    var res = Option(expr);
    while (res.isDefined) {
      lastExpr = res.get;
      t.transform(res.get, macroEnv) match {
        case Success(r) => {
          res = r;
        }
        case Failure(f) => return Failure(f)
      }
    }
    Success(lastExpr)
  }

  def macroTransform(expr: Expr, env: Map[String, Macro]): Try[(Option[Expr], Map[String, Macro])] = {
    import ExprKind._;
    val bodyT = Transformer.exprTransformer(bodyTransform);
    val newExprT: Try[Option[Expr]] = expr.kind match {
      case Application(funcExpr, args) => {
        funcExpr.kind match {
          case Ident(sym) => {
            env.get(sym.text) match {
              case Some(m) =>
                if (m.parameters.size != args.size) {
                  Failure(new MacroException(s"Macro ${m.name.text} takes ${m.parameters.size} parameters, was only given ${args.size} arguments at l.${sym.line}."))
                } else {
                  val bindings = m.parameters.zip(args).map {
                    case (param, arg) => (param.text -> arg)
                  }.toMap;
                  bodyT.transform(m.body, bindings)
                }
              case None => Success(None)
            }
          }
          case _ => Success(None)
        }
      }
      case _ => Success(None)
    };
    newExprT.map(newExpr => (newExpr, env))
  }

  def bodyTransform(expr: Expr, env: Map[String, Expr]): Try[(Option[Expr], Map[String, Expr])] = {
    import ExprKind._;
    val newExprT = expr.kind match {
      case Ident(sym) => {
        env.get(sym.text) match {
          case Some(e) => Success(Some(e))
          case None    => Success(None)
        }
      }
      case _ => Success(None)
    };
    newExprT.map(newExpr => (newExpr, env))
  }

  class MacroException(message: String) extends Exception(message) {

    def this(message: String, cause: Throwable) {
      this(message)
      initCause(cause)
    }

    def this(cause: Throwable) {
      this(Option(cause).map(_.toString).orNull, cause)
    }

    def this() {
      this(null: String)
    }
  }
}
