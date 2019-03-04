package se.kth.cda.arc

package object typeinference {
  type Substituter = Type => scala.util.Try[Type]
}
