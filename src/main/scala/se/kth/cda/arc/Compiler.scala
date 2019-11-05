package se.kth.cda.arc

import java.io.InputStream

import org.antlr.v4.runtime._
import se.kth.cda.arc.ast.AST
import se.kth.cda.arc.ast.AST._
import se.kth.cda.arc.ast.parser.{ErrorListener, Translator}
import se.kth.cda.arc.ast.transformer.MacroExpansion
import se.kth.cda.arc.ast.typer.TypeInference

import scala.util.{Success, Try}

final case class CompilerException(msg: String) extends Exception

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

  def run(input: String): AST.Expr = {
    val inputStream = CharStreams.fromString(input)
    val lexer = new ArcLexer(inputStream)
    val tokenStream = new CommonTokenStream(lexer)
    val parser = new ArcParser(tokenStream)
    val translator = Translator(parser)
    val ast = translator.expr()
    val expanded = MacroExpansion.expand(ast).get
    val typed = TypeInference.solve(expanded).get
    typed
  }

}
