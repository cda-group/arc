package se.kth.cda.arc

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.tree._
import scala.util.{Failure, Success, Try}

class CollectingErrorListener extends BaseErrorListener {
  private var errors: List[String] = List.empty

  override def syntaxError(
      recognizer: Recognizer[_, _],
      offendingSymbol: Object,
      line: Int,
      charPositionInLine: Int,
      msg: String,
      e: RecognitionException): Unit = {
    val errorMsg = s"line $line:$charPositionInLine $msg"
    errors ::= errorMsg
  }

  def hasErrors: Boolean = errors.nonEmpty

  def getErrors: List[String] = errors.reverse

  def map[T](res: Try[T]): Try[T] = res match {
    case Success(t) =>
      if (this.hasErrors) {
        Failure(new ParsingException(getErrors))
      } else {
        Success(t)
      }
    case Failure(f) =>
      val newF = new ParsingException(getErrors)
      newF.addSuppressed(f)
      Failure(newF)
  }
}

class ParsingException(message: String) extends Exception(message) {

  def this(errors: List[String]) {
    this(errors.mkString("\n"))
  }

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
