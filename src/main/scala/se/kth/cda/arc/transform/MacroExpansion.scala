package se.kth.cda.arc.transform

import se.kth.cda.arc.AST._
import se.kth.cda.arc.Arc

import scala.util.{Failure, Success, Try}

object MacroExpansion {

  case class Env(macros: Map[String, Macro], symbols: Map[String, Int], params: Map[String, Expr]) {

    def +(m: Macro): Try[Env] = {
      val name = m.name.text
      if (macros.contains(name)) {
        Failure(new MacroException(s" Macro names must be unique! There are two entries for macro $name."))
      } else {
        val newMacros = macros + (name -> m)
        Success(this.copy(macros = newMacros))
      }
    }

    def ++(ms: Iterable[Macro]): Try[Env] = {
      var newMacros = macros
      ms.foreach { m =>
        val name = m.name.text
        if (macros.contains(name)) {
          return Failure(new MacroException(s" Macro names must be unique! There are two entries for macro $name."))
        } else {
          newMacros += (name -> m)
        }
      }
      Success(this.copy(macros = newMacros))
    }

    def paramsAdd(ps: Iterable[(String, Expr)]): Env = {
      this.copy(params = params ++ ps)
    }

    //def paramsDrop(): Env = this.copy(params = Map.empty);
    def addSymbol(sym: Symbol): Env = {
      val name = sym.name
      symbols.get(name) match {
        case Some(_) => {
          throw new MacroException(s"Symbol is already in store: $name!")
        }
        case None => {
          this.copy(symbols = symbols + (name -> 0))
        }
      }
    }

    def addAndRename(sym: Symbol): (Env, Option[Symbol]) = {
      val name = sym.name
      symbols.get(name) match {
        case Some(scope) => {
          val newScope = scope + 1
          val newName = rename(name, newScope)
          val newSymb = Symbol(newName, None, newScope)
          (this.copy(symbols = symbols + (name -> newScope)), Some(newSymb))
        }
        case None => {
          (this.copy(symbols = symbols + (name -> 0)), None)
        }
      }
    }

    def renameSymbol(sym: Symbol): Option[Symbol] = {
      val name = sym.name
      symbols.get(name) match {
        case Some(scope) if scope == 0 => None
        case Some(scope) if scope != 0 => Some(Symbol(rename(name, scope), None, scope))
        case None                      => None
      }
    }

    def dropSymbol(sym: Symbol): Env = {
      val name = sym.name
      symbols.get(name) match {
        case Some(scope) if scope == 0 => {
          this.copy(symbols = symbols - name)
        }
        case Some(scope) if scope != 0 => {
          val newScope = scope - 1
          this.copy(symbols = symbols + (name -> newScope))
        }
        case None => {
          throw new MacroException(s"Inconsistent symbol store: Failed to remove $name!")
        }
      }
    }
    private val sep = "$__"

    private def rename(name: String, scope: Int): String = s"$name$sep$scope"
  }
  val emptyEnv = Env(Map.empty, Map.empty, Map.empty)

  lazy val standardMacros: List[Macro] = {
    val stream = getClass.getResourceAsStream("/standard_macros.weld")
    if (stream == null) {
      throw new java.io.FileNotFoundException("standard_macros.weld")
    }
    Arc.macros(stream) match {
      case Success(macros) => macros
      case Failure(e)      => throw e
    }
  }

  def expand(prog: Program): Try[Expr] = {
    this.expand(standardMacros ++ prog.macros, prog.expr)
  }

  def expand(expr: Expr): Try[Expr] = {
    this.expand(standardMacros, expr)
  }

  private def expand(macros: List[Macro], expr: Expr): Try[Expr] = {
    (this.emptyEnv ++ macros).flatMap { env =>
      expand(env, expr)
    }
  }

  private def expand(env: Env, expr: Expr): Try[Expr] = {
    val t = Transformer.exprTransformer[Env](macroTransform)
    var lastExpr = expr
    var res = Option(expr)
    while (res.isDefined) {
      lastExpr = res.get
      t.transform(res.get, env) match {
        case Success(r) => {
          res = r
        }
        case Failure(f) => return Failure(f)
      }
    }
    Success(lastExpr)
  }

  def macroTransform(expr: Expr, env: Env): Try[(Option[Expr], Env)] = {
    import ExprKind._
    val bodyT = Transformer.exprTransformer(bodyTransform)
    val alphaC = Transformer.exprTransformer(alphaConvert)
    val newExprT: Try[Option[Expr]] = expr.kind match {
      case Let(name, _, _, _) => {
        return Success((None, env.addSymbol(name)))
      }
      case Lambda(params, body) => {
        val newEnv = params.foldLeft(env) { (accEnv, p) =>
          accEnv.addSymbol(p.name)
        }
        return Success((None, newEnv))
      }
      case Application(funcExpr, args) => {
        funcExpr.kind match {
          case Ident(sym) => {
            env.macros.get(sym.text) match {
              case Some(m) =>
                if (m.parameters.size != args.size) {
                  Failure(new MacroException(
                    s"Macro ${m.name.text} takes ${m.parameters.size} parameters, was only given ${args.size} arguments at l.${sym.line}."))
                } else {
                  val bindings = m.parameters
                    .zip(args)
                    .map {
                      case (param, arg) => param.text -> arg
                    }
                    .toMap
                  for {
                    bodyRenamed <- alphaC.transform(m.body, env)
                    bodyTransformed <- bodyT.transform(bodyRenamed.getOrElse(m.body), env.paramsAdd(bindings))
                  } yield bodyTransformed
                }
              case None => Success(None)
            }
          }
          case _ => Success(None)
        }
      }
      case _ => Success(None)
    }
    newExprT.map(newExpr => (newExpr, env))
  }

  def bodyTransform(expr: Expr, env: Env): Try[(Option[Expr], Env)] = {
    import ExprKind._
    val newExprT = expr.kind match {
      case Ident(sym) => {
        env.params.get(sym.text) match {
          case Some(e) => Success(Some(e))
          case None    => Success(None)
        }
      }
      case _ => Success(None)
    }
    newExprT.map(newExpr => (newExpr, env))
  }

  def alphaConvert(expr: Expr, env: Env): Try[(Option[Expr], Env)] = {
    import ExprKind._
    expr.kind match {
      case Let(name, bty, v, b) =>
        env.addAndRename(name) match {
          case (newEnv, Some(newSymb)) =>
            Success((Some(Expr(Let(name, bty, v, b), expr.ty, expr.ctx)), newEnv))
          case (newEnv, None) =>
            Success((None, newEnv))
        }
      case Lambda(params, body) =>
        val (newEnv, newParamsO) = params.foldLeft((env, Vector.empty[Option[Parameter]])) { (acc, p) =>
          acc match {
            case (accEnv, accParams) =>
              accEnv.addAndRename(p.name) match {
                case (newEnv, Some(newSymb)) => (newEnv, accParams :+ Some(Parameter(newSymb, p.ty)))
                case (newEnv, None)          => (newEnv, accParams :+ None)
              }
          }
        };
        if (newParamsO.forall(_.isEmpty)) {
          Success((None, newEnv))
        } else {
          val newParams = params.zip(newParamsO).map {
            case (_, Some(p)) => p
            case (oldP, None) => oldP
          }
          val lam = Lambda(newParams, body)
          Success((Some(Expr(lam, expr.ty, expr.ctx)), newEnv))
        }
      case Ident(sym) =>
        env.renameSymbol(sym) match {
          case Some(newSym) => Success((Some(Expr(Ident(newSym), expr.ty, expr.ctx)), env))
          case None         => Success((None, env))
        }
      case _ => Success((None, env))
    }
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
