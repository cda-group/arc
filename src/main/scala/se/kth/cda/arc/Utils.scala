package se.kth.cda.arc

import java.io.{ByteArrayOutputStream, PrintStream}
import java.nio.charset.StandardCharsets

import scala.util.{Failure, Success, Try}

object Utils {

  class StringBuilderStream {

    private val baos = new ByteArrayOutputStream()
    lazy val ps = new PrintStream(baos, true, "UTF-8")

    def asPrintStream(): PrintStream = ps

    def result(): String = {
      ps.flush()
      val data = new String(baos.toByteArray, StandardCharsets.UTF_8)
      ps.close()
      baos.close()
      data
    }
  }

  implicit class TryVector[T](tl: Vector[Try[T]]) {

    def sequence: Try[Vector[T]] = {
      val (successes, failures) = tl.partition(_.isSuccess)
      if (failures.isEmpty) {
        Success(successes.flatMap(_.toOption))
      } else {
        failures(0) match {
          case Failure(f) => Failure(f)
          case _          => ??? // unreachable
        }
      }
    }
  }

  implicit class OptionTry[T](o: Option[Try[T]]) {

    def invert: Try[Option[T]] = o match {
      case Some(t) => t.map(v => Some(v))
      case None    => Success(None)
    }
  }

  object PrettyPrint {

    val ps: PrintStream = new PrintStream("~/Desktop/lol")
    println(pretty"asd ${"123"}"(ps))

    trait PrettyPrint {
      def pretty(ps: PrintStream): Unit
    }

    implicit class PrettySeq(val list: Seq[PrettyPrint]) extends AnyVal {

      def sep(sep: String): PrettyPrint = (ps: PrintStream) => {
        val it = list.iterator
        if (it.hasNext) {
          it.next().pretty(ps)
          it.foreach { x =>
            ps.print(sep)
            x.pretty(ps)
          }
        }
      }
    }

    implicit class PrettyPrintString(val str: String) extends PrettyPrint {
      def pretty(pw: PrintStream): Unit = pw.print(str)
    }

    implicit class PrintWriterInterpolator(val sc: StringContext) extends AnyVal {

      def pretty(splices: PrettyPrint*)(pw: PrintStream): Unit = {
        val partsIter = sc.parts.iterator
        val splicesIter = splices.iterator
        pw.print(partsIter.next())
        while (partsIter.hasNext) {
          splicesIter.next().pretty(pw)
          pw.print(partsIter.next)
        }
      }
    }
  }
}
