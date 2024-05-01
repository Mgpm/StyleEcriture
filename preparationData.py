# Databricks notebook source
class preparationData: # pour la preparation des données
    def __init__(self,df,fn):
        self.df = df
        self.fn = fn

    def getDF(self):
        return self.df

    def dataShape(self):
        print((self.df.count(), len(self.df.columns)))

    def dropDuplicates(self):
        self.df = self.df.dropDuplicates()
    
    def dropMissing(self):
        self.df = self.df.dropna()   

    def addIdCol(self):
        self.df = self.df.select(
        [self.fn.monotonically_increasing_id().alias('Id')]\
        +[c for c in self.df.columns]
        )

