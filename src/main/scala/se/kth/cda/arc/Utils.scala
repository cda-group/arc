package se.kth.cda.arc

import java.io.{ ByteArrayOutputStream, OutputStream, PrintStream, IOException }
import java.nio.charset.StandardCharsets;
import scala.util.{ Try, Success, Failure }

object Utils {
  class StringBuilderStream {

    private val baos = new ByteArrayOutputStream();
    lazy val ps = new PrintStream(baos, true, "UTF-8");

    def asPrintStream(): PrintStream = ps;
    def result(): String = {
      ps.flush();
      val data = new String(baos.toByteArray(), StandardCharsets.UTF_8);
      ps.close();
      baos.close();
      data
    }
  }

  implicit class TryVector[T](tl: Vector[Try[T]]) {
    def sequence: Try[Vector[T]] = {
      val (successes, failures) = tl.partition(_.isSuccess);
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
}
