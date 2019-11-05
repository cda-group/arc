package se.kth.cda.arc.ast

package object typer {
  type Substituter = Type => scala.util.Try[Type]
}
