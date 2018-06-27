package se.kth.cda.arc

import java.io.{ OutputStream, PrintStream, IOException }

object Utils {
  class StringBuilderStream extends OutputStream {

    private val buffer = new StringBuilder();

    @throws[IOException]
    override def write(ch: Int): Unit = {
      buffer.append(ch);
    }

    def asPrintStream(): PrintStream = new PrintStream(this);
    def result(): String = buffer.result();
  }
}
