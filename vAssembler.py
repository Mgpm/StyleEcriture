# Databricks notebook source
from pyspark.ml.feature import VectorAssembler,StringIndexer,StandardScaler

class vAssembler: # vectorise le dataframe
    def __init__(self,df):
        self.df = df

    def vectStringIndex(self):
        self.VectStgdf = StringIndexer().setInputCol("label").setOutputCol("labelCode").fit(self.df).transform(self.df) 

    def vectAssembler(self):
         self.vectdf = VectorAssembler().setInputCols(self.df.columns[:-2]).setOutputCol('featuresv').transform(self.VectStgdf)
        
    def scalerStandard(self):
        self.df = StandardScaler().setInputCol("featuresv").setOutputCol('scalerfeatures').setWithStd(True).setWithMean(True).fit( self.vectdf).transform(self.vectdf)
        
    def getDF(self):
        return self.df
