

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

def linearreg():
    print("1")
    return
def multiLR():
    print("2")
    return

def randFR():
    print("3")
    return

model_dict={"LR":linearreg, "MLR": multiLR, "RFReg": randFR,}


key_list=model_dict.keys()
print(key_list)


