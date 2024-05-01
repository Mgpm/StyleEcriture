# Databricks notebook source
from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier
from pyspark.ml.feature import PCA,ChiSqSelector


class featuresSelect: 

    def __init__(self,df,pd,plt):
        self.df = df
        self.pd = pd
        self.plt = plt
    
    def getDF(self):
        return self.df
      
    def pcaSelect(self,k):
        pcaModel = PCA().setK(k).setInputCol("scalerfeatures").setOutputCol("features").fit(self.df)
        self.df = pcaModel.transform(self.df)
        

    def chisqSelect(self,numberFeat):
        chisqsel = ChiSqSelector(numTopFeatures=numberFeat, featuresCol="scalerfeatures",outputCol="features",labelCol='labelCode')
        result_df = chisqsel.fit(self.df).transform(self.df)
        self.df = result_df
        

    def ModelByTreeDecision(self):
        self.model = DecisionTreeClassifier(featuresCol="scalerfeatures",labelCol='labelCode')\
        .fit(self.df)
       
    def ModelByRandomForest(self):
        self.model = RandomForestClassifier(featuresCol="scalerfeatures",labelCol='labelCode')\
       .fit(self.df)
       
    
    def TreeSelect(self): # determine les variables importantes par le model choisi
        self.featureImp = [self.df.columns[i] for i in self.model.featureImportances.indices]
        self.featureImportances = self.pd.Series(index=self.featureImp, data=self.model.featureImportances.values)
        

    def plotFeatureImportances(self): # Affichage graphique des features Importances
        self.featureImportances.sort_values().plot(kind='barh',figsize=(5,50),title='Features Importances')
        self.plt.show()

    def dataFeatureImportances(self): # Nouvelle dataframe après selection des variables selon leurs importances
        self.df = self.df.select(self.featureImp+["label"])




