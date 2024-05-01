# Databricks notebook source
from pyspark.ml.feature import PCA
from pyspark.ml.functions import vector_to_array
from pyspark.mllib import stat as st
import numpy as np


class dataAnalysis: 
    def __init__(self,plt,df):
        self.df = df
        self.plt = plt
        self.cols = [c for c in self.df.columns[:-1]]
        self.colrdd = self.df.select(self.cols).rdd.map(lambda r: [e for e in r])

    def getDF(self):
        return self.df

    def statsDescriptive(self):
        return self.df.describe([c for c in self.cols]).toPandas().head()

    def ListCorrelationVariables(self):
        self.listCorreled = []
        corrs = st.Statistics.corr(self.colrdd)
        for i, el in enumerate(corrs >0.5):
            correlated = [(self.cols[j], corrs[i][j]) for j, e in enumerate(el) if e == 1.0 and j != i]
            if len(correlated) > 0:
                for e in correlated:
                    if [e[0],self.cols[i]] in self.listCorreled:
                        continue
                    else:
                        self.listCorreled.append([self.cols[i],e[0]])
                        #print('{0}-to-{1}: {2:.2f}'.format(self.cols[i], e[0], e[1]))

    
    def removeCorreled(self):
        cols = self.df.columns
        for i in self.listCorreled:
            if i[1] in cols:
                cols.remove(i[1])
        self.df = self.df.select(cols)


    
    def computeVariance(self):
        stats = st.Statistics.colStats(self.colrdd)
        self.variances = [v for v in stats.variance()]
        
    def minVariance(self):
        return min(self.variances)

    def maxVariance(self):
        return max(self.variances)

    def meanVariance(self):
         return np.mean(np.array(self.variances))

    def getColVariance(self,v):
        i = self.variances.index(v)
        return self.cols[i]


    def histoFunc(self,col):
        p = self.df.select(self.df[col]).collect()
        h = [i[0] for i in p]
        self.plt.hist(h,bins=10)


    def scatterFunc(self,col1,col2):
        p = self.df.select(self.df[col1],self.df[col2]).collect()
        c1 = [i[0] for i in p]
        c2 = [i[1] for i in p]
        self.plt.scatter(c1,c2,s=50,c='green')


    def vizData2D(self,fn):  # affiche le graphique 2D des données
        self.pcaDF = PCA()\
        .setInputCol("scalerfeatures").setOutputCol("features")\
        .setK(2).fit(self.df).transform(self.df)
        p = self.pcaDF\
        .withColumn("x",vector_to_array("features"))\
        .select(["labelCode"]+[fn.col("x")[i] for i in range(2)]).toPandas() 
        p.plot.scatter('x[0]','x[1]')
        
        
    def getPCA(self):
        return self.pcaDF
     

