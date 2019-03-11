package se.kth.cda.arc

import org.antlr.v4.runtime._
import org.scalatest._

import scala.language.implicitConversions

class ParserTests extends FunSuite with Matchers {
  implicit class StringTokenAux(val input: String) {

    private val inputStream = CharStreams.fromString(input)
    private val lexer = new ArcLexer(inputStream)
    private val errorC = new ErrorCounter()
    lexer.addErrorListener(errorC)
    private val tokenStream = new CommonTokenStream(lexer)
    private val parser = new ArcParser(tokenStream)

    def expr(): String = {
      val tree = parser.expr()
      assert(!errorC.hasErrors, "An error occurred during parsing!")
      tree.toStringTree(parser)
    }

    def `type`(): String = {
      val tree = parser.`type`()
      assert(!errorC.hasErrors, "An error occurred during parsing!")
      tree.toStringTree(parser)
    }

    def program(): String = {
      val tree = parser.program()
      assert(!errorC.hasErrors, "An error occurred during parsing!")
      tree.toStringTree(parser)
    }

    def macros: String = {
      val tree = parser.macros()
      assert(!errorC.hasErrors, "An error occurred during parsing!")
      tree.toStringTree(parser)
    }
  }

  test("raw parsing") {
    "let x: i32 = 5; x".expr() shouldBe
      "(expr (valueExpr (letExpr let x (typeAnnot : (type i32)) = (operatorExpr (literalExpr 5)) ; (valueExpr (operatorExpr x)))))"

    "|x:i32, y:f32| x".expr() shouldBe
      "(expr (lambdaExpr | (lambdaParams (param x (typeAnnot : (type i32))) , (param y (typeAnnot : (type f32)))) | (valueExpr (operatorExpr x))))"

    "stream[vec[i32]]".`type`() shouldBe
      "(type stream [ (type vec [ (type i32) ]) ])"

    "{i32, vec[u32]}".`type`() shouldBe
      "(type { (type i32) , (type vec [ (type u32) ]) })"
  }
}
