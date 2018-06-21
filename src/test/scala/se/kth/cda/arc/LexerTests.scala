package se.kth.cda.arc

import org.scalatest._
import org.antlr.v4.runtime._
import org.antlr.v4.runtime.tree._
import org.antlr.v4.runtime.atn.ATNConfigSet;
import org.antlr.v4.runtime.atn.DecisionInfo;
import org.antlr.v4.runtime.atn.ParserATNSimulator;
import org.antlr.v4.runtime.atn.PredictionMode;
import org.antlr.v4.runtime.dfa.DFA;
import java.util.BitSet;

class LexerTests extends FunSuite with Matchers {
  implicit class StringTokenAux(val input: String) {
    def ===(tokenIds: Array[Int]): Unit = {
      val inputStream = CharStreams.fromString(input);
      val lexer = new ArcLexer(inputStream);
      val errorC = new ErrorCounter();
      lexer.addErrorListener(errorC);
      val tokenStream = new CommonTokenStream(lexer);
      tokenStream.fill();
      assert(!errorC.hasErrors, "Lexer encountered a syntax error!");
      val tokenList = tokenStream.getTokens;
      val tokens = tokenList.toArray(new Array[Token](tokenList.size()));
      val vocab = lexer.getVocabulary;
      var tokenPos = 0;
      var idPos = 0;
      while (tokenPos < tokens.length) {
        val token = tokens(tokenPos);
        if (token.getChannel == Lexer.DEFAULT_TOKEN_CHANNEL) {
          assert(idPos < tokenIds.length, s"Expected no input, but got $token.");
          val expected = tokenIds(idPos);
          assert(token.getType == expected, s"Expected ${vocab.getDisplayName(expected)}, but got $token at position $idPos.");
          idPos += 1;
        }
        tokenPos += 1;
      }
    }

    def isInvalid(): Unit = {
      val inputStream = CharStreams.fromString(input);
      val lexer = new ArcLexer(inputStream);
      val errorC = new ErrorCounter();
      lexer.addErrorListener(errorC);
      val tokenStream = new CommonTokenStream(lexer);
      tokenStream.fill();
      assert(errorC.hasErrors, s"Expected an error but tokenized fine to:\n${tokenStream.getTokens}");
    }
  }

  test("basic token types") {
    import ArcLexer._;
    import Token.EOF;

    """"test string"""" === Array(TStringLit, EOF);
    """"test" string""" === Array(TStringLit, TIdentifier, EOF);

    "a for 23 + z0" === Array(TIdentifier, TFor, TI32Lit, TPlus, TIdentifier, EOF);

    "= == | || & &&" === Array(TEqual, TEqualEqual, TBar, TBarBar, TAnd, TAndAnd, EOF);
    "|t| !t" === Array(TBar, TIdentifier, TBar, TBang, TIdentifier, EOF);
    "|a:i8| a" === Array(TBar, TIdentifier, TColon, TI8, TBar, TIdentifier, EOF);

    "(a)".isInvalid; // temporarily until the parser introduces parens

    "42si" === Array(TI16Lit, EOF);
    "0b10" === Array(TI32Lit, EOF);
    "0x10" === Array(TI32Lit, EOF);

    "1e-5f" == Array(TF32Lit, EOF);
    "1e-5" == Array(TF64Lit, EOF);
  }
}

class ErrorCounter extends ANTLRErrorListener {
  private var errors: Int = 0;

  def hasErrors: Boolean = errors > 0;

  override def syntaxError(
    recognizer:         Recognizer[_, _],
    offendingSymbol:    Object,
    line:               Int,
    charPositionInLine: Int,
    msg:                String,
    e:                  RecognitionException): Unit = {
    errors += 1;
  }

  override def reportAmbiguity(
    recognizer: Parser,
    dfa:        DFA,
    startIndex: Int,
    stopIndex:  Int,
    exact:      Boolean,
    ambigAlts:  BitSet,
    configs:    ATNConfigSet): Unit = ();

  override def reportAttemptingFullContext(
    recognizer:      Parser,
    dfa:             DFA,
    startIndex:      Int,
    stopIndex:       Int,
    conflictingAlts: BitSet,
    configs:         ATNConfigSet): Unit = ();

  override def reportContextSensitivity(
    recognizer: Parser,
    dfa:        DFA,
    startIndex: Int,
    stopIndex:  Int,
    prediction: Int,
    configs:    ATNConfigSet): Unit = ();
}
