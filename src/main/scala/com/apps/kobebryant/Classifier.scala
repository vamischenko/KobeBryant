package com.apps.kobebryant

import java.io.File

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{SQLContext, DataFrame}

object Classifier {
  val conf = new SparkConf().setAppName("KobeBryant").setMaster("local[*]")
  val sc: SparkContext = new SparkContext(conf)
  val sqlContext: SQLContext = SQLContext.getOrCreate(sc)

  def main(args: Array[String]) = {
    println("Start")
    val absPath = new File(".").getAbsolutePath
    /*val data = sc.textFile(absPath.substring(0, absPath.length-1) + "/data/data.csv").zipWithIndex().filter(_._2 > 0).map(_._1.split(","))
    val structData = data.map(x => ((x(14), x(24)), Array(x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), x(10), x(11), x(12), x(13), x(15), x(16), x(17), x(18), x(19), x(20), x(21), x(22), x(23))))
    val trainData = structData.filter(d => d._1._1 != "")
    println(trainData.count())
    val testData = structData.filter(d => d._1._1 == "")
    println(testData.count())

    val trainDF = sqlContext.createDataFrame(trainData).toDF("label", "features")
    /*trainDF.printSchema()
    trainDF.show(10)*/

    val datsa = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val indexer = new StringIndexer()
      .setInputCol("features")
      .setOutputCol("featIndex")

    val indexed = indexer.fit(trainDF).transform(trainDF)
    indexed.show(10)*/

    val data: DataFrame = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(absPath.substring(0, absPath.length-1) + "/data/data.csv")

    val trainData = data.filter(data("shot_made_flag").isNotNull)
    println(trainData.count())
    val stringColumns = Array("action_type","combined_shot_type", "season", "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "opponent", "game_date")

    trainData.filter(trainData("game_event_id").isNull).show(10)

    /*val indexer: Array[PipelineStage] = stringColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_id")
    )
    val indexPipeline = new Pipeline().setStages(indexer)

    /* val featureColumns = Array("action_type_id","combined_shot_type_id", "game_event_id", "game_id", "lat", "loc_x", "loc_y", "lon",
       "minutes_remaining", "period", "playoffs", "season_id", "seconds_remaining", "shot_distance", "shot_type_id", "shot_zone_area_id",
       "shot_zone_basic_id", "shot_zone_range_id", "team_id", "game_date_id", "opponent_id")*/
    val featureColumns = Array("action_type_id","combined_shot_type_id", "game_event_id")
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val trainIndexModel = indexPipeline.fit(trainData)
    val trainIndexedDF = trainIndexModel.transform(trainData)
    val train = assembler.transform(trainIndexedDF)

    val testData = data.filter(data("shot_made_flag").isNull)
    val testIndexModel = indexPipeline.fit(testData)
    val testIndexedDF = testIndexModel.transform(testData)
    val test = assembler.transform(testIndexedDF)*/

    println("Stop")
  }

}
