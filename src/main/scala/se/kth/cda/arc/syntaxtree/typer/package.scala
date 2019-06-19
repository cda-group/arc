package se.kth.cda.arc.syntaxtree

package object typer {
  type Substituter = Type => scala.util.Try[Type]
}
