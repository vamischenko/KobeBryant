package com.apps.kobebryant

import java.io.{PrintWriter, File}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.{PipelineStage, Pipeline}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.{LinearRegressionModel, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}

object Starter {
  val conf = new SparkConf().setAppName("KobeBryant").setMaster("local[*]")
  val sc: SparkContext = new SparkContext(conf)
  val sqlContext: SQLContext = SQLContext.getOrCreate(sc)

  def main (args: Array[String]): Unit = {
    println("Start")
    val absPath = new File(".").getAbsolutePath
    val data: DataFrame = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(absPath.substring(0, absPath.length-1) + "/data/data.csv")

    //game_date
    val stringColumns = Array("action_type","combined_shot_type", "season", "shot_type", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "opponent", "game_date")

    val indexer: Array[PipelineStage] = stringColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_id")
    )
    val indexPipeline = new Pipeline().setStages(indexer)

    val featureColumns = Array("action_type_id","combined_shot_type_id", "game_event_id", "game_id", "lat", "loc_x", "loc_y", "lon",
      "minutes_remaining", "period", "playoffs", "season_id", "seconds_remaining", "shot_distance", "shot_type_id", "shot_zone_area_id",
      "shot_zone_basic_id", "shot_zone_range_id", "team_id", "game_date_id", "opponent_id")

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val dataIndexModel = indexPipeline.fit(data)
    val dataIndexedDF = dataIndexModel.transform(data)
    val dataDF = assembler.transform(dataIndexedDF).select("shot_made_flag", "features", "shot_id")

    val trainingData = dataDF.filter(data("shot_made_flag").isNotNull)
    val testData = dataDF.filter(data("shot_made_flag").isNull)
    val schema = testData.schema
    val tst = testData.map(row => {
      Row(0, row(1), row(2))
    })
    val newTestData = sqlContext.createDataFrame(tst, schema)

    val Array(train1, test1) = trainingData.randomSplit(Array(0.6, 0.3))
    val testCheck = test1.collect()
    //val predictions = randomForest(train1, test1)
   // val predictions = logisticRegression(train1, test1)
    //val predictions = linearRegression(train1, test1)
    //val predictions = decisionTree(train1, test1)
    //val predictions = decisionTreeRegression(train1, test1)
    //val predictions = gbt(train1, test1)

    //val predictions = decisionTree(trainingData, newTestData)
    // val predictions = decisionTreeRegression(trainingData, newTestData)
    //val predictions = randomForest(trainingData, newTestData)
    //val predictions = logisticRegression(trainingData, newTestData)
    val predictions = linearRegression(trainingData, newTestData)
    //val predictions = gbt(trainingData, newTestData)
   //val predictions = mlp(trainingData, newTestData)
    //println(predictions.count())
   // predictions.printSchema()

    val outFile = new File(absPath.substring(0, absPath.length-1) + "/data/sumbission_linreg.csv")
    val lines = predictions.select("shot_id", "prediction").collect()
    val pw = new PrintWriter(outFile)
    pw.write("shot_id,shot_made_flag\n")
    for (line: Row <- lines) {
     // val prob = line(2).asInstanceOf[DenseVector]
      val prob = line(1)
     // val avg = (prob.values.max + prob.values.min)/2
      val str = s"${line(0)},${prob}"
      pw.write(str + "\n")
    }
    pw.close()
    println("Stop")
  }

  //bad result
  def randomForest(trainingData: DataFrame, test: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(trainingData)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(21)
      .fit(trainingData)

    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(21)
      .setMaxBins(1600)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val model = pipeline.fit(trainingData)
    val predictions = model.transform(test)

   /* val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))*/

    predictions.show(10)
    predictions.printSchema()
    predictions

  }

  //bad result
  def gbt(trainingData: DataFrame, test: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(trainingData)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(21)
      .fit(trainingData)

    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxBins(1600)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(test)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    predictions.show(10)
    predictions.printSchema()
    predictions

  }

  //bad result
  def mlp(train: DataFrame, test: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(train)

    val layers = Array[Int](4, 5)

    val trainer = new MultilayerPerceptronClassifier()
      .setFeaturesCol("features")
      .setLabelCol("indexedLabel")
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, trainer))
    val model = pipeline.fit(train)
    val predictions = model.transform(test)

    predictions.show(100)
    predictions.printSchema()
    predictions

  }

  def logisticRegression(training: DataFrame, test: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(training)

    val lr = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setRegParam(0.003)
      .setElasticNetParam(0.5)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, lr))
    val model = pipeline.fit(training)
    val predictions = model.transform(test)

     /*val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))*/

    predictions.show(10)
    predictions.printSchema()
    predictions
  }

  def linearRegression(training: DataFrame, test: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(training)

    val lr = new LinearRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setRegParam(0.03)
      .setElasticNetParam(0.5)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, lr))
    val model = pipeline.fit(training)
    val predictions = model.transform(test)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    predictions.show(100)
    predictions.printSchema()
    predictions
  }

  //bad result
  def decisionTree(trainingData: DataFrame, testData: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(trainingData)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(21)
      .fit(trainingData)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures").setMaxBins(1600)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    predictions.show(100)
    predictions.printSchema()
    predictions
  }


  def decisionTreeRegression(trainingData: DataFrame, testData: DataFrame) = {
    val labelIndexer = new StringIndexer()
      .setInputCol("shot_made_flag")
      .setOutputCol("indexedLabel")
      .fit(trainingData)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(25)
      .fit(trainingData)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val dt = new DecisionTreeRegressor()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxBins(1570)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    predictions.show(100)
    predictions.printSchema()
    predictions
  }
}
