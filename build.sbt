ThisBuild / organization := "com.drissoft"
ThisBuild / organizationName := "drissoft"
ThisBuild / maintainer := "dris@drissoft.com"
ThisBuild / homepage := Some(url("https://github.com/Dris101/dl4jgrapher"))
ThisBuild / version := "0.1.0"

ThisBuild / developers := List(
  Developer(
    id = "Dris101",
    name = "Chris",
    email = "dris@drissoft.com",
    url = url("https://github.com/Dris101")
  )
)

ThisBuild / scmInfo := Some(
  ScmInfo(
    browseUrl = url("https://github.com/Dris101/dl4jgrapher"),
    connection = "scm:git@github.com:Dris101/dl4jgrapher.git"
  )
)

ThisBuild / description := "Generates Graphviz DOT files from DL4J MultiLayerNetworks and ComputationGraphs"
ThisBuild / licenses := List("Apache 2" -> new URL("http://www.apache.org/licenses/LICENSE-2.0.txt"))

// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }
ThisBuild / publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value) Some("snapshots" at nexus + "content/repositories/snapshots")
  else Some("releases" at nexus + "service/local/staging/deploy/maven2")
}
ThisBuild / publishMavenStyle := true
ThisBuild / publishArtifact in Test := false

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

credentials += Credentials(Path.userHome / ".sbt" / "sonatype_credentials")
usePgpKeyHex("11A931AE0AC089B715CCA7C31D98028EE2F5E09C")

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
