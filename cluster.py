# Databricks notebook source
from pyspark.ml.clustering import KMeans,BisectingKMeans,GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array


class cluster: # Pour la classification des données
    def __init__(self,plt,df):
        self.plt = plt
        self.df = df

    def getDF(self):
        return self.df

    def evalClusterPlot(self,Mcluster,n):
        cluster_num = range(2,n)
        scores=[]
        evaluator = ClusteringEvaluator(featuresCol="features")
        for i in range(2,n):
            cl = Mcluster(k=i,featuresCol="features")
            results = cl.fit(self.df).transform(self.df)
            r = evaluator.evaluate(results)
            scores.append(r) 
        self.plt.figure(figsize=(15,5))
        self.plt.xlabel('Numbres de clusters')
        self.plt.ylabel('Scores')
        self.plt.plot(cluster_num,scores)
    
    def clusterPlot(self,Mcluster,fn,n): 
        cl = Mcluster(k=n,featuresCol="features")
        self.clusterDF = cl.fit(self.df).transform(self.df)
        p = self.clusterDF.withColumn("x",vector_to_array("features")).select(["prediction"]+[fn.col("x")[i] for i in range(2)]).toPandas() 
        p.plot.scatter('x[0]','x[1]',c='prediction', cmap='coolwarm')


    def clusterPlotbyPCA(self,Mcluster,n): 
        cl = Mcluster(k=n,featuresCol="features")
        self.clusterDF = cl.fit(self.df).transform(self.df) 
        pcaDF = PCA().setInputCol("features").setOutputCol("PCAfeatures").setK(2).fit( self.clusterDF).transform(self.clusterDF)
        p = pcaDF.withColumn("x",vector_to_array("PCAfeatures")).select(["prediction"]+[fn.col("x")[i] for i in range(2)]).toPandas() 
        p.plot.scatter('x[0]','x[1]',c='prediction', cmap='coolwarm')



    def getCluster(self):
        return self.clusterDF

    








