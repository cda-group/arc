package se.kth.cda.arc.ast.printer

import se.kth.cda.arc.Utils
import se.kth.cda.arc.ast.ASTNode

object Printer {
  import se.kth.cda.arc.ast.printer.ArcPrinter._
  import se.kth.cda.arc.ast.printer.MLIRPrinter._

  implicit class Printer(val ast: ASTNode) extends AnyVal {

    def toStringFormat(format: String): String = {
      format match {
        case "ARC" => {
          val sb = new Utils.StringBuilderStream()
          val ps = sb.asPrintStream()
          ps.printArc(ast)
          sb.result()
        }
        case "MLIR" => ast.toMLIR
        case _      => s"Unrecognized format ${format}"
      }
    }

  }

}
