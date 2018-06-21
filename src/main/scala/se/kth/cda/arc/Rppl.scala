package se.kth.cda.arc

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.tree._

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

        val tree = parser.expr(); // begin parsing at r rule
        Console.out.print("<= ");
        Console.out.println(tree.toStringTree(parser)); // print LISP-style tree
      } catch {
        case e: Throwable => Console.err.println(s"An error occurred during parsing:\n$e")
      }
    }
  }
}
