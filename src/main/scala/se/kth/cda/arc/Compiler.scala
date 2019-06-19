package se.kth.cda.arc

import java.io.InputStream

import org.antlr.v4.runtime._
import se.kth.cda.arc.syntaxtree.AST._
import se.kth.cda.arc.syntaxtree.parser.{ErrorListener, Translator}

import scala.util.Try

object Compiler {

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

  def translatorForStream(in: InputStream): (Translator, ErrorListener) = {
    val input = CharStreams.fromStream(in)
    val lexer = new ArcLexer(input)
    val tokens = new CommonTokenStream(lexer)
    val parser = new ArcParser(tokens)
    parser.removeErrorListeners()
    val errorCollector = new ErrorListener()
    parser.addErrorListener(errorCollector)
    (Translator(parser), errorCollector)
  }
}
