package se.kth.cda.arc

import org.antlr.v4.runtime._
import se.kth.cda.arc.syntaxtree.PrettyPrint._
import se.kth.cda.arc.syntaxtree.parser.{ErrorListener, Translator}
import se.kth.cda.arc.syntaxtree.transformer.MacroExpansion
import se.kth.cda.arc.syntaxtree.typer.TypeInference

import scala.runtime.NonLocalReturnControl
import scala.util.control.Breaks._

// Read-Parse-Print Loop
object RPPL {

  def main(args: Array[String]): Unit = {
    while (true) {
      try {
        breakable {

          // read standard input
          Console.out.print("=> ")
          val line = Console.in.readLine()
          Console.out.println()
          if (line == null || line.equals("")) {
            return // exit on EOF
          }

          // create a CharStream that reads from standard input
          val input = CharStreams.fromString(line)

          Console.out.println("Starting lexical analysis")
          // create a lexer that feeds off of input CharStream
          val lexer = new ArcLexer(input)

          // create a buffer of tokens pulled from the lexer
          val tokens = new CommonTokenStream(lexer)
          tokens.fill()
          Console.out.print("<=")
          Console.out.println(tokens.getTokens)

          // create a parser that feeds off the token buffer
          Console.out.println("Starting syntactic analysis")
          val parser = new ArcParser(tokens)
          val errorCollector = new ErrorListener()

          parser.removeErrorListeners()
          parser.addErrorListener(errorCollector)

          val tree = parser.expr() // begin parsing at expr rule
          if (errorCollector.hasErrors) {
            Console.err.println(s"There were parsing errors:\n ${errorCollector.getErrors.mkString("\n")}")
            break
          }

          Console.out.print("<=")
          Console.out.println(tree.toStringTree(parser)) // print LISP-style tree

          val ast = Translator(parser).translate(tree)
          Console.out.print("<= ")
          Console.out.prettyPrintln(ast)

          Console.out.println("Starting macro expansion")
          val expanded = MacroExpansion
            .expand(ast)
            .recover {
              case f =>
                Console.err.println(s"An error occurred during macro expansion!: ${f.getMessage}")
                f.printStackTrace(Console.err)
                break
            }
            .get

          Console.out.print("<= ")
          Console.out.prettyPrintln(expanded)

          Console.out.println("Starting type inference")
          val typed = TypeInference
            .solve(expanded)
            .recover {
              case f =>
                Console.err.println(s"An error occurred during type inference! ${f.getMessage}")
                f.printStackTrace(Console.err)
                break
            }
            .get

          Console.out.print("<= ")
          Console.out.println(typed.ty.render)
          Console.out.prettyPrintln(typed)

        }
      } catch {
        case _: NonLocalReturnControl[_] =>
          return
        case e: Throwable =>
          Console.err.println("An error occurred during parsing!")
          e.printStackTrace(Console.err)
      }
    }
  }
}
