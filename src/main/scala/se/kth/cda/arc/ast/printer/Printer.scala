package se.kth.cda.arc.ast.printer

import se.kth.cda.arc.Utils
import se.kth.cda.arc.ast.ASTNode

object Printer {
  import se.kth.cda.arc.ast.printer.ArcPrinter._
  import se.kth.cda.arc.ast.printer.MLIRPrinter._

  implicit class Printer(val ast: ASTNode) extends AnyVal {
    def toStringFormat(format: String): String = {
      val sb = new Utils.StringBuilderStream()
      val ps = sb.asPrintStream()
      format match {
        case "ARC" => ps.printArc(ast)
        case "MLIR" => ps.printMLIR(ast)
        case _ => ps.print("Unrecognized format " + format)
      }
      sb.result()
    }

  }

}
