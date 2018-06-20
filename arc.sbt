enablePlugins(Antlr4Plugin)

name := "Arc"

organization := "se.kth.cda"


version := "0.1.0-SNAPSHOT"

scalaVersion := "2.12.4"

scalacOptions ++= Seq("-deprecation","-feature")


resolvers += Resolver.mavenLocal
resolvers += "Kompics Releases" at "http://kompics.sics.se/maven/repository/"
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.+"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.+" % "test"


antlr4PackageName in Antlr4 := Some("se.kth.cda.arc")
antlr4GenListener in Antlr4 := true
antlr4GenVisitor in Antlr4 := true
