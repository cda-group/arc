package se.kth.cda.arc

import java.io.{BufferedWriter, File, FileWriter}

import se.kth.cda.arc.ast.printer.Printer._

import scala.util.{Failure, Success}

object Driver {
  def main(args: Array[String]): Unit = {
    args match {
        // REPL
      case Array("-r") =>
        RPPL.run()
        // Read input file, write to output file
      case Array("-f", format, "-i", inputPath, "-o", outputPath) =>
        val inputFile = scala.io.Source.fromFile(inputPath)
        val code = inputFile.getLines.mkString("\n")
        inputFile.close()
        val ast = Compiler.run(code)
        val outputFile = new File(outputPath)
        val bw = new BufferedWriter(new FileWriter(outputFile))
        bw.write(ast.toStringFormat(format))
        bw.close()
        // Read string from command line, write to stdout
      case Array("-f", format, code) =>
        val ast = Compiler.run(code)
        Console.out.println(ast.toStringFormat(format))
        // Read from stdin, write to stdout
      case Array("-f", format) =>
        val code = Console.in.readLine()
        val ast = Compiler.run(code)
        Console.out.println(ast.toStringFormat(format))
      // Read from stdin, write to stdout, default to ARC format
      case Array() =>
        val code = Console.in.readLine()
        val ast = Compiler.run(code)
        Console.out.println(ast.toStringFormat("ARC"))
    }
  }
}
