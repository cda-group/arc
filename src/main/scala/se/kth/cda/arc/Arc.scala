package se.kth.cda.arc

import java.io.InputStream

import org.antlr.v4.runtime._
import se.kth.cda.arc.AST._

import scala.util.Try

object Arc {

  def macros(in: InputStream): Try[List[Macro]] = {
    val (translator, ec) = translatorForStream(in)
    ec.map(Try(translator.macros()))
  }

  def program(in: InputStream): Try[Program] = {
    val (translator, ec) = translatorForStream(in)
    ec.map(Try(translator.program()))
  }

  def expr(in: InputStream): Try[Expr] = {
    val (translator, ec) = translatorForStream(in)
    ec.map(Try(translator.expr()))
  }

  def translatorForStream(in: InputStream): (ASTTranslator, CollectingErrorListener) = {
    val input = CharStreams.fromStream(in)
    val lexer = new ArcLexer(input)
    val tokens = new CommonTokenStream(lexer)
    val parser = new ArcParser(tokens)
    parser.removeErrorListeners()
    val errorCollector = new CollectingErrorListener()
    parser.addErrorListener(errorCollector)
    (ASTTranslator(parser), errorCollector)
  }
}
