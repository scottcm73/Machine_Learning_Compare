
### Take in data for model1 ###
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

class ML_model_compare:

    
    def __init__(self, model_dict, data_df):
        # self.model_dict=model_dict
        self.data_df=data_df
        self.model_dict=model_dict
        print(data_df)
 
    #make_predictions function assumes all numbers in data_df dataframe. This may mean converting datetimes to number columns
    
    def traintestsplit(self):
        this_df=self.data_df
 
       
        X =self.data_df[["Open", "Low", "Close", "Adj Close", "Volume"]]
        y = self.data_df[["High"]]
        model_dict = self.model_dict
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = X
        self.y = y
    def feature_selection(self):
        #apply SelectKBest class to extract top 10 best features
        X = self.X
        y = self.y  
        
        #Using Pearson Correlation
        plt.figure(figsize=(12,10))
        cor = X.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
        return
  
    
    def make_predictions(self):
        
        
       
        

        y_pred=0

        
        return y_pred


    def crossvalidator(self):



    

        return  
    
    def do_accuracy_score(self):

        score=[]
        acuracy_score(y_true, y_pred)
        return




def getdata():
    the_file="EOS-USD.csv"
    the_path=os.path.join("data", the_file)
    data_df=pd.read_csv(the_path)
    return data_df

def linearreg():
    print("1")
    return
def multiLR():
    print("2")
    return

def randFR():
    print("3")
    return
names=[]
train_scores=[]
test_scores=[]

data_df=getdata()
print(data_df.head())


# data_df['conv_date'] = pd.to_datetime(data_df.Date, format="%Y-%M-%D") 
# data_df = data_df.index.to_julian_date()

print(data_df.info())
data_df['Date'] = pd.to_datetime(data_df['Date'])
print(data_df.head())

print(data_df.info())




# Work with datetime later
# For now remove datetime column 

data_df.drop(["Date"], axis=1, inplace=True)
# The line above will eventually be deleted.



model_dict={"LR":linearreg(), "MLR": multiLR(), "RFReg": randFR(),}


# key_list=model_dict.keys()
# print(key_list)

# for name, model in model_dict.items():
#     name_model = model
    # name_fit = name_model.fit(X_train, y_train)
    # name_pred = name_model.predict(X_test)
    # name_train_score = name_model.score(X_train, y_train).round(4)
    # name_test_score = name_model.score(X_test, y_test).round(4)
    # names.append(name)
    # train_scores.append(name_train_score)
    # test_scores.append(name_test_score)

# score_df = pd.DataFrame(names, train_scores, test_scores)
print(data_df.head())
ml_compare=ML_model_compare(model_dict, data_df)

this_df=ml_compare.data_df
print(this_df)

ml_compare.traintestsplit()

X_train = ml_compare.X_train
y_train = ml_compare.y_train
X_test = ml_compare.X_test
y_test = ml_compare.y_test

print(y_test.head())
ml_compare.feature_selection()