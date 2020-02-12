package se.kth.cda.arc.ast.transformer

import se.kth.cda.arc.ast.AST._
import se.kth.cda.arc.Compiler

import scala.util.{Failure, Success, Try}

object MacroExpansion {

  final case class Env(macros: Map[String, Macro], symbols: Map[String, Int], parameters: Map[String, Expr]) {

    def addMacro(m: Macro): Try[Env] =
      if (macros.contains(m.symbol.name)) {
        Failure(new MacroException(s"Macro names must be unique! There are two entries for macro ${m.symbol.name}."))
      } else {
        Success(this.copy(macros = macros + (m.symbol.name -> m)))
      }

    def addMacros(ms: Iterable[Macro]): Try[Env] = {
      var newMacros = macros
      ms.foreach { m =>
        val name = m.symbol.name
        if (macros.contains(name)) {
          return Failure(new MacroException(s"Macro names must be unique! There are two entries for macro $name."))
        } else {
          newMacros += (name -> m)
        }
      }
      Success(this.copy(macros = newMacros))
    }

    def addParams(ps: Iterable[(String, Expr)]): Env = this.copy(parameters = parameters ++ ps)

    //def paramsDrop(): Env = this.copy(params = Map.empty);

    def addSymbol(symbol: Symbol): Env =
      symbols.get(symbol.name) match {
        case Some(_) =>
          throw new MacroException(s"Symbol is already in store: ${symbol.name}!")
        case None =>
          this.copy(symbols = symbols + (symbol.name -> 0))
      }

    private val sep = "$__"

    private def rename(name: String, scope: Int): String = s"$name$sep$scope"

    def addAndRename(symbol: Symbol): (Env, Option[Symbol]) =
      symbols.get(symbol.name) match {
        case Some(scope) =>
          val newScope = scope + 1
          val newName = rename(symbol.name, newScope)
          val newSymbol = Symbol(newName, None, newScope)
          (this.copy(symbols = symbols + (symbol.name -> newScope)), Some(newSymbol))
        case None =>
          (this.copy(symbols = symbols + (symbol.name -> 0)), None)
      }

    def renameSymbol(symbol: Symbol): Option[Symbol] =
      symbols.get(symbol.name) match {
        case Some(scope) if scope == 0 => None
        case Some(scope) if scope != 0 => Some(Symbol(rename(symbol.name, scope), None, scope))
        case None                      => None
      }

    def dropSymbol(symbol: Symbol): Env =
      symbols.get(symbol.name) match {
        case Some(scope) if scope == 0 =>
          this.copy(symbols = symbols - symbol.name)
        case Some(scope) if scope != 0 =>
          val newScope = scope - 1
          this.copy(symbols = symbols + (symbol.name -> newScope))
        case None =>
          throw new MacroException(s"Inconsistent symbol store: Failed to remove ${symbol.name}!")
      }
  }

  val emptyEnv = Env(Map.empty, Map.empty, Map.empty)

  lazy val standardMacros: List[Macro] =
    Option(classOf[Macro].getResourceAsStream("/standard_macros.weld"))
      .map(Compiler.macros)
      .get
      .getOrElse(throw new java.io.FileNotFoundException("standard_macros.weld"))

  def expand(prog: Program): Try[Expr] = expand(standardMacros ++ prog.macros, prog.expr)

  def expand(expr: Expr): Try[Expr] = expand(standardMacros, expr)

  private def expand(macros: List[Macro], expr: Expr): Try[Expr] = emptyEnv.addMacros(macros).flatMap(expand(_, expr))

  private def expand(originalEnv: Env, originalExpr: Expr): Try[Expr] = {
    val transformer = Transformer.exprTransformer[Env](macroTransform)
    var finalExpr = originalExpr
    var result = Option(originalExpr)
    while (result.isDefined) {
      finalExpr = result.get
      transformer.transform(finalExpr, originalEnv) match {
        case Success(r) => result = r
        case Failure(f) => return Failure(f)
      }
    }
    Success(finalExpr)
  }

  import ExprKind._

  def macroTransform(originalExpr: Expr, originalEnv: Env): Try[(Option[Expr], Env)] = {
    val bodyTransformer = Transformer.exprTransformer(bodyTransform)
    val alphaConverter = Transformer.exprTransformer(alphaConvert)
    originalExpr.kind match {
      case Let(symbol, _, _, _) =>
        Success((None, originalEnv.addSymbol(symbol)))
      case Lambda(params, _) =>
        Success((None, params.foldLeft(originalEnv) { (accEnv, p) =>
          accEnv.addSymbol(p.symbol)
        }))
      case Application(funcExpr, args) =>
        funcExpr.kind match {
          case Ident(symbol) =>
            originalEnv.macros.get(symbol.name) match {
              case Some(mac) =>
                if (mac.params.size != args.size) {
                  Failure(new MacroException(
                    s"Macro ${mac.symbol.name} takes ${mac.params.size} parameters, was only given ${args.size} arguments at l.${symbol.line}."))
                } else {
                  val bindings = mac.params
                    .zip(args)
                    .map {
                      case (param, arg) => param.name -> arg
                    }
                    .toMap
                  val newEnv = originalEnv.addParams(bindings)
                  for {
                    bodyRenamed <- alphaConverter.transform(mac.body, newEnv)
                    bodyTransformed <- bodyTransformer.transform(bodyRenamed.getOrElse(mac.body), newEnv)
                  } yield (bodyTransformed, newEnv)
                }
              case None => Success((None, originalEnv))
            }
          case _ => Success((None, originalEnv))
        }
      case _ => Success((None, originalEnv))
    }
  }

  /** Replaces an identifier with its value.
    */
  def bodyTransform(originalExpr: Expr, originalEnv: Env): Try[(Option[Expr], Env)] =
    originalExpr.kind match {
      case Ident(symbol) => Success((originalEnv.parameters.get(symbol.name), originalEnv))
      case _             => Success((None, originalEnv))
    }

  def alphaConvert(originalExpr: Expr, originalEnv: Env): Try[(Option[Expr], Env)] =
    originalExpr.kind match {
      case ExprKind.Let(symbol, bindingTy, value, body) =>
        originalEnv.addAndRename(symbol) match {
          case (newEnv, Some(newSymbol)) =>
            val newExpr = Expr(
              kind = ExprKind.Let(newSymbol, bindingTy, value, body),
              ty = originalExpr.ty,
              ctx = originalExpr.ctx
            )
            Success((Some(newExpr), newEnv))
          case (newEnv, None) =>
            Success((None, newEnv))
        }
      case Lambda(params, body) =>
        params.foldLeft((originalEnv, Vector.empty[Option[Parameter]])) { (acc, param) =>
          acc match {
            case (accEnv, accParams) =>
              accEnv.addAndRename(param.symbol) match {
                case (newEnv, Some(newSymbol)) => (newEnv, accParams :+ Some(Parameter(newSymbol, param.ty)))
                case (newEnv, None)            => (newEnv, accParams :+ None)
              }
          }
        } match {
          case (newEnv, newParams) if !newParams.forall(_.isEmpty) =>
            val newExpr = Expr(
              kind = Lambda(
                params = params.zip(newParams).map {
                  case (_, Some(newParam)) => newParam
                  case (oldParam, None)    => oldParam
                },
                body
              ),
              ty = originalExpr.ty,
              ctx = originalExpr.ctx
            )
            Success((Some(newExpr), newEnv))
          case (newEnv, _) =>
            Success((None, newEnv))
        }
      case Ident(symbol) =>
        originalEnv.renameSymbol(symbol) match {
          case Some(newSymbol) =>
            val newExpr = Expr(
              kind = Ident(newSymbol),
              ty = originalExpr.ty,
              ctx = originalExpr.ctx
            )
            Success((Some(newExpr), originalEnv))
          case None =>
            Success((None, originalEnv))
        }
      case _ => Success((None, originalEnv))
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
