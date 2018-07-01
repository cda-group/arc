package se.kth.cda.arc.typeinference

import se.kth.cda.arc._
import AST.Expr
import Types._
import scala.util.{ Try, Success, Failure }

object ConstraintSolver {
  sealed trait Result {
    def isSolved: Boolean;
    def typeSubstitutions: Substituter;
    def describe: String;
  }
  case class Solution(assignments: Map[Int, Type]) extends Result {
    override def isSolved: Boolean = true;
    override def typeSubstitutions: Substituter = substitute(_);
    private def substitute(ty: Type): Try[Type] = ty match {
      case tv: TypeVariable => assignments.get(tv.id) match {
        case Some(t) => Success(t)
        case None    => Failure(new TypingException(s"Could not find mapping for ${tv.render} in complete solution!"))
      }
      case t => TypeConstraints.substituteType(t, assignments) match {
        case Some(newTy) => Success(newTy)
        case None        => Success(t)
      }
    }
    override def describe: String = {
      assignments.toList.sortBy(_._1).map(t => s"?${t._1} <- ${t._2.render}").mkString("[", ",", "]");
    }
  }
  case class PartialSolution(assignments: Map[Int, Type], constraints: List[TypeConstraint]) extends Result {
    override def isSolved: Boolean = false;
    override def typeSubstitutions: Substituter = substitute(_);
    private def substitute(ty: Type): Try[Type] = ty match {
      case tv: TypeVariable => Success(assignments.get(tv.id).getOrElse(tv)) // don't fail
      case t => TypeConstraints.substituteType(t, assignments) match {
        case Some(newTy) => Success(newTy)
        case None        => Success(t)
      }
    }
    override def describe: String = {
      assignments.toList.sortBy(_._1).map(t => s"?${t._1} <- ${t._2.render}").mkString("[", ",", "]");
    }
    def describeUnresolvedConstraints(e: Expr): Failure[Expr] = {
      val descriptions = constraints.map(_.describe);
      val description = descriptions.mkString("\n∧ ");
      val exprS = PrettyPrint.print(e);
      val msg = s"""Expression could not be typed! Closest partially typed expr:
$exprS
Unresolved Constraints:
$description
To solve this issue try annotating types where type variables remain.""";
      Failure(new TypingException(msg))
    }
  }
  def solve(initialConstraints: List[TypeConstraint]): Try[Result] = {
    import TypeConstraints._;

    var typeAssignments = Map.empty[Int, Type];
    var topLevelConjunction = initialConstraints;
    var changed = true;
    while (changed) {
      changed = false;
      coalesce(topLevelConjunction) match {
        case Some(cs) => {
          changed = true;
          topLevelConjunction = cs;
        }
        case None => // no change
      }
      //println(s"=== Constraints after coalescing:\n${topLevelConjunction.map(_.describe).mkString("\n∧")}\n===");
      //System.exit(1); // abort to avoid spam
      // rewrite first to avoid doing an empty substitution pass in the beginning
      rewrite(topLevelConjunction, typeAssignments) match {
        case Some((cs, ass)) => {
          changed = true;
          topLevelConjunction = cs;
          typeAssignments = ass;
        }
        case None => // no change
      }
      //println(s"=== Constraints after rewriting:\n${topLevelConjunction.map(_.describe).mkString("\n∧")}\n===");
      topLevelConjunction = substituteAndNormalise(topLevelConjunction, typeAssignments) match {
        case Some(cs) => {
          changed = true;
          cs
        }
        case None => topLevelConjunction
      };
      //println(s"=== Constraints after substitution:\n${topLevelConjunction.map(_.describe).mkString("\n∧")}\n===");
    }
    if (topLevelConjunction.isEmpty) {
      Success(Solution(typeAssignments))
    } else {
      Success(PartialSolution(typeAssignments, topLevelConjunction))
    }
  }

  private def coalesce(cs: List[TypeConstraint]): Option[List[TypeConstraint]] = {
    //println(s"Coalescing:\n ${cs.map(_.describe).mkString("\n∧")}");
    var changed = false;
    var typeConstraints = Map.empty[Int, Either[Int, List[TypeConstraint]]]; // either a forwarding position or a bunch of type constraints
    var newCS = List.empty[TypeConstraint];
    cs.foreach { c =>
      //println(s"Processing ${c.describe}...");
      val vars = c.variables();
      if (vars.isEmpty) {
        //println(s"Skiping ${c.describe} as it has no type vars");
        newCS ::= c;
      } else {
        val lowestVar = vars.sortBy(_.id).head;
        val (rId, lowestConstraints) = resolveForward(lowestVar, typeConstraints);
        val lId = if (rId < 0) lowestVar.id else rId;
        var newLCS = c :: lowestConstraints;
        vars.foreach { v =>
          if (v.id != lId && v.id != lowestVar.id) {
            resolveForward(v, typeConstraints) match {
              case (-1, _) => typeConstraints += (v.id -> Left(lId))
              case (rId, cs) => {
                newLCS ++= cs;
                typeConstraints += (rId -> Left(lId));
              }
            }
          } else if (v.id != lId && v.id == lowestVar.id) {
            typeConstraints += (v.id -> Left(lId))
          }
        }
        typeConstraints += (lId -> Right(newLCS));
      }
    }
    //    val tcS = typeConstraints.toList.sortBy(_._1).map(tc => tc match {
    //      case (id, Left(rId)) => s"?${id} -> ?${rId}"
    //      case (id, Right(cs)) => s"?${id} -> ${cs.map(_.describe).mkString("∧")}"
    //    }).mkString("{\n	", "\n	", "\n}");
    //    println(s"var constraints (raw): ${tcS}");
    typeConstraints.foreach {
      case (_, Left(_)) => // leave as they are
      case (id, Right(cs)) => minimiseRelated(cs) match {
        case Some(newCS) => {
          changed = true;
          typeConstraints += (id -> Right(newCS));
        }
        case None => // leave as they are
      }
    }
    //    println(s"no-var constraints: ${newCS.map(_.describe).mkString("∧")}");
    //    val tcS2 = typeConstraints.toList.sortBy(_._1).map(tc => tc match {
    //      case (id, Left(rId)) => s"?${id} -> ?${rId}"
    //      case (id, Right(cs)) => s"?${id} -> ${cs.map(_.describe).mkString("∧")}"
    //    }).mkString("{\n	", "\n	", "\n}");
    //println(s"var constraints (minimised): ${tcS2}");
    newCS = typeConstraints.foldLeft(newCS)((acc, e) => e match {
      case (_, Right(cs)) => acc ++ cs
      case _              => acc
    });
    //println(s"New constraints after fold: ${newCS.map(_.describe).mkString("∧")}")
    if (changed) Some(newCS) else None
  }

  private def minimiseRelated(cs: List[TypeConstraint]): Option[List[TypeConstraint]] = {
    //println(s"Minimising ${cs.map(_.describe).mkString("∧")}");
    var changed = false;
    var changedThisIter = true;
    var newCS = cs;
    while (changedThisIter) {
      changedThisIter = false;
      var heads = List.empty[TypeConstraint];
      var tails = newCS;
      while (!tails.isEmpty) {
        var merging = tails.head;
        val res = tails.tail.foldLeft(List.empty[TypeConstraint]) { (acc, c) =>
          c.merge(merging) match {
            case Some(newC) => {
              //println(s"Merged ${merging.describe} and ${c.describe} into ${newC.describe}");
              changed = true;
              changedThisIter = true;
              merging = newC;
              acc
            }
            case None => {
              //println(s"Failed to merged ${merging.describe} and ${c.describe}");
              (c :: acc)
            }
          }
        };
        heads ::= merging;
        tails = res;
      }
      newCS = heads;
    }
    //println(s"Minimised to ${cs.map(_.describe).mkString("∧")} (changed? ${changed})");
    if (changed) Some(newCS) else None
  }

  private def resolveForward(tv: TypeVariable, data: Map[Int, Either[Int, List[TypeConstraint]]]): (Int, List[TypeConstraint]) = {
    var lookup = Option(tv.id);
    while (lookup.isDefined) {
      data.get(lookup.get) match {
        case Some(Left(id))  => lookup = Some(id)
        case Some(Right(cs)) => return (lookup.get, cs)
        case None            => lookup = None
      }
    }
    (-1, List.empty)
  }

  private def rewrite(cs: List[TypeConstraint], assignments: Map[Int, Type]): Option[(List[TypeConstraint], Map[Int, Type])] = {
    import TypeConstraints._;

    var newCS = List.empty[TypeConstraint];
    var newAssign = assignments;
    var changed = false;
    cs.foreach {
      case meq: MultiEquality => {
        if (meq.isResolved) {
          changed = true;
          val ty = meq.resolvedType.get;
          meq.variables().foreach { tv =>
            newAssign += (tv.id -> ty)
          }
        } else {
          newCS ::= meq
        }
      }
      case MultiConj(members) => {
        changed = true;
        newCS ++= members;
      }
      case pred: Predicate     => newCS ::= pred
      case bk: BuilderKind     => newCS ::= bk
      case ik: IterableKind    => newCS ::= ik
      case pk: ProjectableKind => newCS ::= pk
      case Tautology           => changed = true // drop since (x and true) = x
    }
    if (changed) {
      Some((newCS, newAssign))
    } else {
      None
    }
  }

  private def substituteAndNormalise(cs: List[TypeConstraint], assignments: Map[Int, Type]): Option[List[TypeConstraint]] = {
    var changed = false;
    val newCS = cs.map(c => (c, c.substitute(assignments))).map {
      case (c, Some(sub)) => (c, Some(sub), sub.normalise())
      case (c, None)      => (c, None, c.normalise())
    } map {
      case (_, _, Some(norm)) => {
        changed = true;
        norm
      }
      case (_, Some(sub), None) => {
        changed = true;
        sub
      }
      case (orig, None, None) => orig
    };
    if (changed) {
      Some(newCS)
    } else {
      None
    }
  }
}
