package se.kth.cda.arc

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.tree._
import se.kth.cda.arc.typeinference.TypeInference
import se.kth.cda.arc.transform.MacroExpansion

// Read-Parse-Print Loop
object Rppl {

  def main(args: Array[String]): Unit = {
    // create a CharStream that reads from standard input
    var active = true
    while (active) {
      try {
        Console.out.print("=> ")
        val line = Console.in.readLine()
        if (line == null) {
          return; // exit on EOF
        }
        val input = CharStreams.fromString(line)

        // create a lexer that feeds off of input CharStream
        val lexer = new ArcLexer(input)

        // create a buffer of tokens pulled from the lexer
        val tokens = new CommonTokenStream(lexer)

        Console.out.print("<= ")
        tokens.fill()
        Console.out.println(tokens.getTokens)

        // create a parser that feeds off the token buffer
        val parser = new ArcParser(tokens)
        parser.removeErrorListeners()
        val errorCollector = new CollectingErrorListener()
        parser.addErrorListener(errorCollector)
        // TODO do two step parsing

        val tree = parser.expr(); // begin parsing at r rule
        if (errorCollector.hasErrors) {
          Console.err.println(s"There were parsing errors:")
          errorCollector.getErrors.foreach { e =>
            Console.err.println(e);
          }
        } else {
          Console.out.print("<= ")
          Console.out.println(tree.toStringTree(parser)); // print LISP-style tree

          val translator = ASTTranslator(parser)
          val ast = translator.translate(tree)
          //        Console.out.print("<= ");
          //        Console.out.println(ast);
          Console.out.print("<= ")
          Console.out.print("\n")
          PrettyPrint.print(ast, Console.out)
          Console.out.print("\n")
          Console.out.println("Starting macro expansion")
          MacroExpansion.expand(ast) match {
            case util.Success(expanded) => {
              Console.out.print("<= ")
              Console.out.print("\n")
              PrettyPrint.print(expanded, Console.out)
              Console.out.print("\n")
              Console.out.println("Starting type inference")
              TypeInference.solve(expanded) match {
                case util.Success(typed) => {
                  Console.out.println("Typed Expression:")
                  Console.out.print("<= ")
                  Console.out.print(typed.ty.render)
                  Console.out.print("\n")
                  PrettyPrint.print(typed, Console.out)
                  Console.out.print("\n")
                }
                case util.Failure(f) => {
                  Console.err.println(s"An error occurred during type inference!")
                  f.printStackTrace(Console.err)
                }
              }
            }
            case util.Failure(f) => {
              Console.err.println(s"An error occurred during macro expansion!")
              f.printStackTrace(Console.err)
            }
          }
        }
      } catch {
        case e: Throwable => {
          Console.err.println(s"An error occurred during parsing!")
          e.printStackTrace(Console.err)
        }
      }
    }
  }
}
