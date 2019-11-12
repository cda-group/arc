package se.kth.cda.arc

import java.io.{BufferedWriter, FileWriter}

import se.kth.cda.arc.ast.printer.Printer._

object Driver {

  /**
    *  Open and read the input from the given file.
    *
    *  @param filename - The file to read or '-' for stdin.
    */
  def getInput(filename: String): String = {
    filename match {
      case "-" => {
        Iterator.continually(scala.io.StdIn.readLine()).takeWhile(x => x != null).mkString("\n")
      }
      case _ => {
        val source = scala.io.Source.fromFile(filename)
        val data = source.getLines.mkString("\n")
        source.close()
        data
      }
    }
  }

  /**
    *  Open and write data to the file.
    *
    *  @param filename - The file to read or '-' for stdout.
    *  @param data - The data to write.
    */
  def writeOutput(filename: String, data: String): Unit = {
    filename match {
      case "-" => print(data)
      case _ =>
        val bw = new BufferedWriter(new FileWriter(filename))
        bw.write(data)
        bw.close()
    }
  }

  def displayUsage(): Unit = {
    println("[-r] [-f format] [-i filename] [-o filename] [expression]")
  }

  type OptionsMap = Map[Symbol, String]

  val defaultOptions = Map('format -> "ARC", 'input -> "-", 'output -> "-", 'repl -> "", 'expression -> "")

  /**
    * Parse and return the command line options.
    *
    * @param options - The current configuration.
    * @param args - The remaining command line options.
    */
  def parseOpts(options: OptionsMap, args: List[String]): OptionsMap = {
    args match {
      case Nil       => options // We're done
      case "-r" :: _ =>
        // We're done, the user wants to be interactive
        defaultOptions ++ Map('repl -> "true")
      case "-f" :: format :: rest =>
        parseOpts(options ++ Map('format -> format), rest)
      case "-i" :: input :: rest =>
        parseOpts(options ++ Map('input -> input), rest)
      case "-o" :: output :: rest =>
        parseOpts(options ++ Map('output -> output), rest)
      case unknown :: _ =>
        println(s"Unknown option: ${unknown}")
        displayUsage()
        System.exit(2)
        options
    }

  }

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      displayUsage()
      System.exit(1)
    }

    val options = parseOpts(defaultOptions, args.toList)

    if (options('repl) != "") {
      RPPL.run()
      System.exit(0)
    }

    val input = if (options('expression) != "") options('expression) else getInput(options('input))

    val ast = Compiler.run(input)
    writeOutput(options('output), ast.toStringFormat(options('format)))
  }
}
