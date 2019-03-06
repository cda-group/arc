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

wartremoverErrors ++= Seq(
  Wart.ArrayEquals,
  // Wart.Any,
  Wart.AnyVal,
  // Wart.AsInstanceOf,
  // Wart.DefaultArguments,
  Wart.EitherProjectionPartial,
  // Wart.Enumeration,
  // Wart.Equals,
  Wart.ExplicitImplicitTypes,
  Wart.FinalCaseClass,
  Wart.FinalVal,
  Wart.ImplicitConversion,
  Wart.ImplicitParameter,
  Wart.IsInstanceOf,
  Wart.JavaConversions,
  Wart.JavaSerializable,
  Wart.LeakingSealed,
  Wart.MutableDataStructures,
  // Wart.NonUnitStatements,
  // Wart.Nothing,
  // Wart.Null,
  // Wart.Option2Iterable,
  // Wart.OptionPartial,
  // Wart.Overloading,
  // Wart.Product,
  // Wart.PublicInference,
  // Wart.Recursion,
  // Wart.Return,
  // Wart.Serializable,
  Wart.StringPlusAny,
  // Wart.Throw,
  // Wart.ToString,
  // Wart.TryPartial,
  // Wart.TraversableOps,
  // Wart.Var,
  // Wart.While,
)

scalacOptions ++= Seq(
  "-deprecation",                  // Emit warning and location for usages of deprecated APIs.
  "-encoding", "utf-8",            // Specify character encoding used by source files.
  "-explaintypes",                 // Explain type errors in more detail.
  "-feature",                      // Emit warning and location for usages of features that should be imported explicitly.
  "-language:existentials",        // Existential types (besides wildcard types) can be written and inferred
  "-language:experimental.macros", // Allow macro definition (besides implementation and application)
  "-language:higherKinds",         // Allow higher-kinded types
  "-language:implicitConversions", // Allow definition of implicit functions called views
  "-unchecked",                    // Enable additional warnings where generated code depends on assumptions.
  "-Xcheckinit",                   // Wrap field accessors to throw an exception on uninitialized access.
  "-Xdev",                         // Indicates user is a developer - issue warnings about anything which seems amiss
  "-Xlint:_",                      // Enable all lint warnings
  "-Xfuture",                      // Turn on future language features.
  "-Yno-adapted-args",             // Do not adapt an argument list (either by inserting () or creating a tuple) to match the receiver.
  "-Ypartial-unification",         // Enable partial unification in type constructor inference
  "-Ywarn-dead-code",              // Warn when dead code is identified.
  "-Ywarn-extra-implicit",         // Warn when more than one implicit parameter section is defined.
  "-Ywarn-inaccessible",           // Warn about inaccessible types in method signatures.
  "-Ywarn-infer-any",              // Warn when a type argument is inferred to be `Any`.
  "-Ywarn-nullary-override",       // Warn when non-nullary `def f()' overrides nullary `def f'.
  "-Ywarn-nullary-unit",           // Warn when nullary methods return Unit.
  "-Ywarn-numeric-widen",          // Warn when numerics are widened.
  "-Ywarn-unused:_",               // Warn for all unused declarations
  "-Ywarn-value-discard"           // Warn when non-Unit expression results are unused.
)

scalacOptions in (Compile, console) --= Seq("-Ywarn-unused:imports", "-Xfatal-warnings")