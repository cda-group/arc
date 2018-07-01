package se.kth.cda.arc.typeinference

import se.kth.cda.arc._
import AST._
import scala.util.{ Try, Success, Failure }
import Utils.TryVector

object TypeInference {
  def solve(expr: Expr): Try[Expr] = {
    for {
      (constraints, updatedExpr) <- new ConstraintGenerator(expr).generate();
      result <- ConstraintSolver.solve(constraints);
      finalExpr <- Typer.applyTypes(updatedExpr, result)
    } yield finalExpr
  }
}

class TypingStore(stack: List[(Symbol, Type)]) {
  def +(t: (Symbol, Type)): TypingStore = new TypingStore(t :: stack);
  def ++(l: List[(Symbol, Type)]): TypingStore = new TypingStore(stack ::: l);
  def lookup(needle: Symbol): Option[Type] = {
    stack.foreach {
      case (s, t) if s.name == needle.name => return Some(t)
      case _                               => // ignore
    }
    None
  }
}
object TypingStore {
  def empty(): TypingStore = new TypingStore(List.empty);
}

class TypingException(message: String) extends Exception(message) {

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