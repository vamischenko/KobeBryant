name := "KobeBryant"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq("org.apache.spark" % "spark-core_2.11" % "1.6.1",
  "org.apache.spark" % "spark-mllib_2.11" % "1.6.1",
  "org.apache.spark" % "spark-sql_2.11" % "1.6.1",
  "com.databricks" % "spark-csv_2.11" % "1.4.0"
)
    