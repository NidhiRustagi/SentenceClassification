#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
from sklearn.metrics import classification_report
import pickle


# #### Initializing all the file names: <br> model_fn: name of the file that stores the trained SVM classifier<br> TRAIN_DATA_FN: file name that contains the training data <br> TEST_FN: file name with test data set. This is currently a .txt file with one english sentence to be tested in a line <br> RESULT_FN: file name that stores the final result.

# In[2]:


MODEL_FN = 'model\svm_classifier.sav'
TRAIN_DATA_FN = 'data\classification_data.tsv'
TEST_FN = 'data\\test_samples.txt'
RESULT_FN = 'result\classification_result.tsv'


# In[3]:


class SVMClassifier:
    # this method acts like a constructor to initialize the basic data structures/files
    def __init__(self, trainfn, modelfn, testfn, resultfn):
        self.train_fn = trainfn
        self.modelfn = modelfn
        self.testfn = testfn
        self.resultfn = resultfn
        text, outcome = self.read()
        self.corpus = text.to_list()
        self.y = np.array(outcome)
        print("y.shape : ", self.y.shape)
        
        self.transformobj, self.featuresX = self.transformFeaturesX()
        print("Input Features shape (X): ",self.featuresX.shape)
        
    # to read the training dataset
    def read(self):
        raw_data = pd.read_csv(self.train_fn, sep="\t")
        return (raw_data['sent'], raw_data['class'])
    
     #  converts raw english sentences to their vectorized representations
    def transformFeaturesX(self):
        tfidfobj = TfidfVectorizer(ngram_range = (1,3))
        features_X = tfidfobj.fit_transform(self.corpus)
#         print(tfidfobj.get_feature_names()[0:10])

        return(tfidfobj, features_X)
        
    # split the training data set into train and validation set.
    def generateTrainTestData(self):
        self.train_X,self.test_X,self.train_y,self.test_y = train_test_split(self.featuresX, self.y, test_size=0.30, random_state=42)
        print("Train X : ", self.train_X.shape)
        print("Test X : ", self.test_X.shape)
        print("Train y: ", self.train_y.shape)
        print("Test y :", self.test_X.shape)
        
    
    # method uses GridSearch to find the optimal values of model hyper-parameters
    def optimalModelParam(self):
        params ={
            'C': [0.1, 1, 10, 100, 1000],
            'gamma':[1, 0.1, 0.01, 0.001, 0.0001],
            'kernel':['linear', 'rbf']
        }

        grid = GridSearchCV(svm.SVC(), params, refit = True, verbose = 0) 
        grid.fit(self.train_X, self.train_y)
        print(grid.best_params_)
        return grid
        
    # build and train the SVM model
    def svmModel(self):
        print("split train and test set")
        self.generateTrainTestData()
        print('checking for optimal param values....')
        self.optimalModel = self.optimalModelParam().best_estimator_
        self.optimalModel.fit(self.train_X, self.train_y)
        #return optimalModel
        
    def saveModel(self):
        pickle.dump(self.optimalModel, open(self.modelfn, "wb"))
        
    def loadModel(self):
        self.optimalModel = pickle.load(open(self.modelfn, "rb"))
        
    # Use the trained model above to check with the validation dataset
    def validateSVM(self):
        predictions = self.optimalModel.predict(self.test_X)
        print(metrics.accuracy_score(self.test_y, predictions))
        #print(classification_report(self.test_y, predictions))
        
    # use this method to test inidividual sentence
    def testSentence(self, sent):
        testcase = self.transformobj.transform(sent)
        print(testcase)
        
        tgt_class = self.optimalModel.predict(testcase)
        print("target class : ",tgt_class)
        if tgt_class:
            print("Well formed sentence")
        else:
            print("Needs cleaning.....")
            
    # use this method to test sentences stored in a .txt file or .csv file with only one column
    def testFileofSentences(self):
        raw_test_data = pd.read_csv(self.testfn)
        raw_test_data.columns = ['sent']
        test_data = raw_test_data['sent'].to_list()
        text = self.transformobj.transform(test_data)
        
        result = self.optimalModel.predict(text)
        
        result_decoded = ["good" if x==1 else "bad" for x in result]
        result_df = pd.DataFrame(
        {
            'sentence':raw_test_data['sent'],
            'class_code':result,
            'class_name':result_decoded
        })
        
        result_df.to_csv(self.resultfn, sep='\t', index=False)        
        


# In[4]:


clsfr = SVMClassifier(TRAIN_DATA_FN, MODEL_FN, TEST_FN, RESULT_FN)
clsfr.svmModel()
clsfr.validateSVM()
clsfr.saveModel()


# In[ ]:


model = clsfr.loadModel()

test_sample_1 = "this sentence looks perfect"
test_sample_2 = "perfect no sentence"
clsfr.testSentence([test_sample_1])


# In[ ]:


clsfr.testSentence([test_sample_2])


# In[ ]:


clsfr.testSentence(['bulum43h - Stardoll | English'])


# #### To check for sentences stored in a .txt file, call testFileofSentences() method.

# In[5]:


clsfr.testFileofSentences()


# In[ ]:




