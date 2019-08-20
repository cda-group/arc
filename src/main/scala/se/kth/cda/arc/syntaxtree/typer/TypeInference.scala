package se.kth.cda.arc.syntaxtree.typer

import se.kth.cda.arc.syntaxtree.AST._
import se.kth.cda.arc.syntaxtree.Type
import se.kth.cda.arc.syntaxtree.typer.PostProcess._

import scala.util.Try

object TypeInference {

  def solve(expr: Expr): Try[Expr] = {
    for {
      (constraints, updatedExpr) <- new ConstraintGenerator(expr).generate()
      result <- ConstraintSolver.solve(constraints)
      finalExpr <- Typer.applyTypes(updatedExpr, result)
    } yield finalExpr.postProcess
  }
}

class TypingStore(stack: List[(Symbol, Type)]) {
  def +(t: (Symbol, Type)): TypingStore = new TypingStore(t :: stack)

  def ++(l: List[(Symbol, Type)]): TypingStore = new TypingStore(stack ::: l)

  def lookup(needle: Symbol): Option[Type] = {
    stack.foreach {
      case (s, t) if s.name == needle.name => return Some(t)
      case _                               => // ignore
    }
    None
  }
}

object TypingStore {
  def empty(): TypingStore = new TypingStore(List.empty)
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
