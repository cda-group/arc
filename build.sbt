enablePlugins(Antlr4Plugin)

name := "Arc"

organization := "se.kth.cda"


version := "0.1.0-SNAPSHOT"

scalaVersion := "2.12.7"

scalacOptions ++= Seq("-deprecation","-feature")


resolvers += Resolver.mavenLocal
resolvers += "Kompics Releases" at "http://kompics.sics.se/maven/repository/"
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.+"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.+" % "test"


antlr4PackageName in Antlr4 := Some("se.kth.cda.arc")
antlr4GenListener in Antlr4 := true
antlr4GenVisitor in Antlr4 := true

wartremoverErrors ++= Warts.allBut(
  // Wart.ArrayEquals,
  Wart.Any,
  // Wart.AnyVal,
  Wart.AsInstanceOf,
  Wart.DefaultArguments,
  // Wart.EitherProjectionPartial,
  Wart.Enumeration,
  Wart.Equals,
  // Wart.ExplicitImplicitTypes,
  // Wart.FinalCaseClass,
  // Wart.FinalVal,
  // Wart.ImplicitConversion,
  // Wart.ImplicitParameter,
  // Wart.IsInstanceOf,
  // Wart.JavaConversions,
  // Wart.JavaSerializable,
  // Wart.LeakingSealed,
  // Wart.MutableDataStructures,
  Wart.NonUnitStatements,
  Wart.Nothing,
  Wart.Null,
  Wart.Option2Iterable,
  Wart.OptionPartial,
  Wart.Overloading,
  Wart.Product,
  Wart.PublicInference,
  Wart.Recursion,
  Wart.Return,
  Wart.Serializable,
  // Wart.StringPlusAny,
  Wart.Throw,
  Wart.ToString,
  Wart.TryPartial,
  Wart.TraversableOps,
  Wart.Var,
  Wart.While,
)