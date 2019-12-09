package se.kth.cda.arc

import java.util.BitSet

import org.antlr.v4.runtime._
import org.antlr.v4.runtime.atn.ATNConfigSet
import org.antlr.v4.runtime.dfa.DFA
import org.scalatest._

import scala.language.implicitConversions

object LexerTests {

  type TokenType = Int;

  implicit class StringTokenAux(val input: String) extends AnyVal {
    def =?=(tokenIds: Array[TokenType]): Unit = {
      val inputStream = CharStreams.fromString(input)
      val lexer = new ArcLexer(inputStream)
      val errorC = new ErrorCounter()
      lexer.addErrorListener(errorC)
      val tokenStream = new CommonTokenStream(lexer)
      tokenStream.fill()
      assert(!errorC.hasErrors, "Lexer encountered a syntax error!")
      val tokenList = tokenStream.getTokens
      val tokens = tokenList.toArray(new Array[Token](tokenList.size()))
      val vocab = lexer.getVocabulary
      var tokenPos = 0
      var idPos = 0
      while (tokenPos < tokens.length) {
        val token = tokens(tokenPos)
        if (token.getChannel == Lexer.DEFAULT_TOKEN_CHANNEL) {
          assert(idPos < tokenIds.length, s"Expected no input, but got $token.")
          val expected = tokenIds(idPos)
          val found = token.getType
          assert(expected == found, s"""
            |Expected ${vocab.getDisplayName(expected)},
            |found ${vocab.getDisplayName(found)} $found at position $idPos.
            |""".stripMargin)

          idPos += 1
        }
        tokenPos += 1
      }
    }
  }
}

class LexerTests extends FunSuite with Matchers {

  test("basic token types") {
    import LexerTests.StringTokenAux
    import ArcLexer._
    import Token.EOF

    """"test string"""" =?= Array(TStringLit, EOF)
    """"test" string""" =?= Array(TStringLit, TString, EOF)

    "a for 23 + z0" =?= Array(TIdentifier, TFor, TI32Lit, TPlus, TIdentifier, EOF)

    "= == | || & &&" =?= Array(TEqual, TEqualEqual, TBar, TBarBar, TAnd, TAndAnd, EOF)
    "|t| !t" =?= Array(TBar, TIdentifier, TBar, TBang, TIdentifier, EOF)

    "0b10" =?= Array(TI32Lit, EOF)
    "0x10" =?= Array(TI32Lit, EOF)

    "1e-5f" =?= Array(TF32Lit, EOF)
    "1e-5" =?= Array(TF64Lit, EOF)

    "123c" =?= Array(TI8Lit, EOF)
    "123C" =?= Array(TI8Lit, EOF)
    "123si" =?= Array(TI16Lit, EOF)
    "123" =?= Array(TI32Lit, EOF)
    "123l" =?= Array(TI64Lit, EOF)
    "123L" =?= Array(TI64Lit, EOF)
    "123.0f" =?= Array(TF32Lit, EOF)
    "123.0F" =?= Array(TF32Lit, EOF)
    "123.0" =?= Array(TF64Lit, EOF)
    "123.0" =?= Array(TF64Lit, EOF)

    "1e-5f" =?= Array(TF32Lit, EOF)
    "1e-5" =?= Array(TF64Lit, EOF)
  }
}

class ErrorCounter extends ANTLRErrorListener {
  private var errors: Int = 0

  def hasErrors: Boolean = errors > 0

  override def syntaxError(
    recognizer:         Recognizer[_, _],
    offendingSymbol:    Object,
    line:               Int,
    charPositionInLine: Int,
    msg:                String,
    e:                  RecognitionException): Unit = {
    errors += 1
  }

  override def reportAmbiguity(
    recognizer: Parser,
    dfa:        DFA,
    startIndex: Int,
    stopIndex:  Int,
    exact:      Boolean,
    ambigAlts:  BitSet,
    configs:    ATNConfigSet): Unit = ()

  override def reportAttemptingFullContext(
    recognizer:      Parser,
    dfa:             DFA,
    startIndex:      Int,
    stopIndex:       Int,
    conflictingAlts: BitSet,
    configs:         ATNConfigSet): Unit = ()

  override def reportContextSensitivity(
    recognizer: Parser,
    dfa:        DFA,
    startIndex: Int,
    stopIndex:  Int,
    prediction: Int,
    configs:    ATNConfigSet): Unit = ()
}
