
import matplotlib.pyplot as plt
import joblib
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix
from statistics import mean
import pandas as pd
import prettytable
import numpy as np
import itertools
class XGBoost_model():
    def __init__(self, savepath):
        self.data = pd.read_csv(savepath)
        

    def upload_model(self):
        df = self.data
        y_test = df["头盆不称"].values.tolist()
        X_test = df.drop(columns=["患者编号", '头盆不称'], axis=1)
        aucs = []
        accuracys = []
        f1s = []
        for i in range(5):
            model_name = "rf" + str(i) + "knn.joblib"
            best_model = joblib.load(model_name)
            predicted = best_model.predict(X_test)
            # print(predicted)
            y_score = best_model.predict_proba(X_test)  # 随机森林
            fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)


            accuracy = accuracy_score(y_test, predicted)

            f1 = metrics.f1_score(y_test, predicted, average='weighted')
            f1s.append(f1)
            aucs.append(roc_auc)
            accuracys.append(accuracy)
        # df["预测标签"] = predicted
        # print(df)
        print(mean(f1s), mean(aucs), mean(accuracys))
        return mean(f1s), mean(aucs), mean(accuracys)
    def predict_the_label(self):
        df = self.imputed_data()

        X_test = df.drop(columns=["患者编号", '头盆不称'], axis=1)


        model_name = r"rf1knn.joblib"
        best_model = joblib.load(model_name)
        predicted = best_model.predict(X_test)
        score = best_model.predict_proba(X_test)
        score = score[:,1]

        df["预测标签"] = predicted
        df["预测分数"] = score
        # print(df)
        df.to_csv("预测.csv", sep=",", index=None)







if __name__ == "__main__":
    model = XGBoost_model(r"验证数据.csv")

    model.upload_model()
    
    model.predict_the_label()

