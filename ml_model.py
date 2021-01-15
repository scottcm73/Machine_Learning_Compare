

class ML_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    def __init__(self):
        pass 
    def make_predictions(self, X, model_list):
        X=self.X
        model_list=self.model_list
        
        return 
    def crossvalidator(self, y_true):
        self.model_list=model_list
        result_ave_list=[]
        for ml_num in range(0, model_list):
            y=model[]()

            result=crossvalidate(model[],X,y)
            result_ave=(sum(result))/(len(result))
            result_ave_list.append(result_ave)
            self.result_ave_list=result_ave_list

        return result_ave_list
    
    def do_accuracy_score(self, y_true, y_pred):

        score=[]
        acuracy_score(y_true, y_pred)
        return