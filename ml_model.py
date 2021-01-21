

class ML_model():
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
    import pandas as pd
    
    def __init__(self):
        pass 
    def make_predictions(self, X, model_dict, y_true):
        X=self.X
        model_dict=self.model_dict
        

        y_pred=0

        
        return y_pred
    def crossvalidator(self, y_true):
        self.model_dict=model_dict
        result_ave_list=[]


    

        return  
    
    def do_accuracy_score(self, y_true, y_pred):

        score=[]
        acuracy_score(y_true, y_pred)
        return


### Take in data for model1 ###

def getdata():
    print("0")
    return

def traintestsplit():
    print("split")
    return

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
model_dict={"TTS": traintestsplit(), "GD": getdata(), "LR":linearreg(), "MLR": multiLR(), "RFReg": randFR(),}


key_list=model_dict.keys()
print(key_list)

for name, model in model_dict.items():
    name_model = model
    # name_fit = name_model.fit(X_train, y_train)
    # name_pred = name_model.predict(X_test)
    # name_train_score = name_model.score(X_train, y_train).round(4)
    # name_test_score = name_model.score(X_test, y_test).round(4)
    # names.append(name)
    # train_scores.append(name_train_score)
    # test_scores.append(name_test_score)

# score_df = pd.DataFrame(names, train_scores, test_scores)
