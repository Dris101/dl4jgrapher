ThisBuild / organization := "com.drissoft"
ThisBuild / maintainer := "me@drissoft.com"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.11"
ThisBuild / crossScalaVersions := Seq("2.12.11", "2.13.1")

ThisBuild / useCoursier := true
ThisBuild / scalacOptions := Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Ywarn-unused",
  "-Yrangepos",
  "-Ywarn-macros:after"
)

val dl4jVersion      = "1.0.0-beta7"
val circeVersion     = "0.13.0"
val logbackVersion   = "1.2.3"
val scalaTestVersion = "3.1.1"

lazy val dl4jgrapher = (project in file("."))
  .enablePlugins(JavaAppPackaging)
  .settings(
    libraryDependencies ++= commonDependencies ++
      dl4jDependencies ++
      circeDependencies
  )

val dl4jDependencies = Seq(
  "org.deeplearning4j" % "deeplearning4j-core",
  "org.nd4j"           % "nd4j-native"
).map(_ % dl4jVersion)

val circeDependencies = Seq(
  "io.circe" %% "circe-core",
  "io.circe" %% "circe-generic",
  "io.circe" %% "circe-parser",
  "io.circe" %% "circe-optics"
).map(_ % circeVersion)

val commonDependencies = Seq(
  "org.scalatest"     %% "scalatest"          % scalaTestVersion % Test,
  "org.deeplearning4j" % "deeplearning4j-zoo" % dl4jVersion      % Test
)

Global / fork := true
Global / cancelable := true
Global / onChangedBuildSource := ReloadOnSourceChanges
logBuffered in Test := false
