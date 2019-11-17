from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from ExtractText import ExtractText as ex
import numpy as np
from collections import Counter
import pickle 
import os
from sklearn.metrics import classification_report

class Bayes(object):
    def __init__(self):
        pass

    def train(self,output_path='./data/model',train_dir = "./data/samples/training/data"):
        features_matrix, labels, file_names = ex().extract_features(most_dictionary=3000)
        X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(features_matrix,labels,file_names,test_size=0.3,random_state=1)

        model = MultinomialNB()
        model = model.fit(X_train,y_train)
		
        y_true = y_test
        y_pred = model.predict(X_test)
		
        print("""Test Data:
           \nFile : \n{file_name}
           \nExactly Result : \n{true}
           \nPredict Result : \n{test}""".format(file_name=np.array(name_test),true=np.array(y_true),test=np.array(y_pred)))
		
        # Độ chính xác
        y_train_pred = classification_report(y_train,model.predict(X_train))
        y_test_pred  = classification_report(y_test,model.predict(X_test))

        print("""【{model_name}】\n Train Accuracy: \n{train}
           \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__, train=y_train_pred, test=y_test_pred))

        pickle.dump(model, open(output_path+'/bayes.model', 'wb'))
    

    @staticmethod
    def load_model(model_path = './data/model/bayes.model'):
        return pickle.load(open(model_path, 'rb'))

    def predict(self,text,model_path = './data/model/bayes.model'):
        text = ex().text2matrix(text)
        clf = self.load_model(model_path)
        y_pred = clf.predict([text])
        predict_proba = clf.predict_proba([text])
        return y_pred ,predict_proba

def load_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        text_file = f.read()
    return text_file

if __name__=='__main__':
    dt = Bayes()
    dt.train()
    ### test 
    # text = load_file('./data/samples/raw/data/the-thao/20191116181352301.txt')
    text = 'Bệnh ung thư rất nguy hiểm' 
    label , predict_proba = dt.predict(text)
    print(label) #kết quả
    print(os.listdir('./data/samples/training/data')) #list topic
    print(predict_proba) #độ chính xác ứng theo từng topic
