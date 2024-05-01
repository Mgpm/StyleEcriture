# Databricks notebook source
class loadData:
    
    def realData(self, spark, sch,fn):
        self.spark = spark
        self.sch = sch
        self.fn = fn
        self.df = spark.read.format("csv").schema(schema).load("data.csv")
        
       
    
    def loadDataCleeans(self,spark):
        self.spark = spark
        self.df = self.spark.read.table("dataCleans")
    
        
    def getDF(self):
        return self.df
    









































# COMMAND ----------


