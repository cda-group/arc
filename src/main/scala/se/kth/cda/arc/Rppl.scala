package se.kth.cda.arc

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.tree._
import se.kth.cda.arc.typeinference.TypeInference

// Read-Parse-Print Loop
object Rppl {
  def main(args: Array[String]): Unit = {
    // create a CharStream that reads from standard input
    var active = true;
    while (active) {
      try {
        Console.out.print("=> ");
        val line = Console.in.readLine();
        if (line == null) {
          return ; // exit on EOF
        }
        val input = CharStreams.fromString(line);

        // create a lexer that feeds off of input CharStream
        val lexer = new ArcLexer(input);

        // create a buffer of tokens pulled from the lexer
        val tokens = new CommonTokenStream(lexer);

        Console.out.print("<= ");
        tokens.fill();
        Console.out.println(tokens.getTokens);

        // create a parser that feeds off the tokens buffer
        val parser = new ArcParser(tokens);
        // TODO do two step parsing

        val translator = ASTTranslator(parser);

        val tree = parser.expr(); // begin parsing at r rule
        Console.out.print("<= ");
        Console.out.println(tree.toStringTree(parser)); // print LISP-style tree
        val ast = translator.translate(tree);
        Console.out.print("<= ");
        Console.out.println(ast);
        Console.out.print("<= ");
        Console.out.print("\n");
        PrettyPrint.print(ast, Console.out);
        Console.out.print("\n");
        Console.out.println("Starting type inference");
        TypeInference.solve(ast) match {
          case util.Success(typed) => {
            Console.out.println("Typed Expression:");
            Console.out.print("<= ");
            Console.out.print(typed.ty.render);
            Console.out.print("\n");
            PrettyPrint.print(typed, Console.out);
            Console.out.print("\n");
          }
          case util.Failure(f) => {
            Console.err.println(s"An error occurred during type inference!");
            f.printStackTrace(Console.err)
          }
        }
      } catch {
        case e: Throwable => {
          Console.err.println(s"An error occurred during parsing!");
          e.printStackTrace(Console.err);
        }
      }
    }
  }
}
