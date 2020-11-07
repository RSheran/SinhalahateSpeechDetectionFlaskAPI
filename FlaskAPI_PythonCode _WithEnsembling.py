#==================== Basic Imports ================================================
import sys
import datetime
import time
import sqlite3
import base64
import pickle,dill
import sklearn.datasets,sys,re,timeit


import numpy as np
import pandas as pd
import string
import nltk
import unidecode
import sklearn.datasets,re
import argparse
import matplotlib
import hashlib
import json
import sqlite3


from pytz import timezone
from Crypto.Cipher import AES
from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from flask import Flask,session, jsonify,abort, request, Response,render_template
from lxml import html
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP
# =========== < End > ===================

#========= Classifiers ============================================
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
# =========== < End > ===================

#========== <Count Vectorization,feature extraction and summaries > ======================
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
# =========== < End > ===================


app=Flask(__name__)


class DataCleaner:
    
    inputString=None   
    unideCodeList=None    
    
    
    def ValidatePhraseIsNullOrEmpty(self,inputPhrase):
        try:
            if len(inputPhrase)>0:        
                return True
            return False
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveHTMLElments: "+str(e)+"\n"
            raise Exception(fullError)
            #return fullError
            
    def ConvertToUnideCode(self,chkSentence):
        try:
            speech_Unidecode=unidecode(chkSentence)
            return speech_Unidecode   
        except:
            e = sys.exc_info()[1]
            print("ConvertToUnideCode: "+str(e)+"\n")
            raise Exception(fullError)
            
    def ConvertToUnideCodeForList(self,phraseList):
        try:
            phraseEnglishSyllable=[]

            for posSpeechPhrase in phraseList:
                posSplitPhrase=posSpeechPhrase.split()
                for splitPhraseElem in posSplitPhrase:
                    phraseEnglishSyllable.append(splitPhraseElem+"#"+unidecode(splitPhraseElem))
                    
            speech_Unidecode=[(unidecode(speechPhrase)) for speechPhrase in phraseList]            
            self.unideCodeList=phraseEnglishSyllable            
            return speech_Unidecode
        except:
            e = sys.exc_info()[1]
            print("ConvertToUnideCodeForList: "+str(e)+"\n")
            raise Exception(fullError)
            
    def ConvertToUnideCodeForDf(self,phraseDf):
        try:

            phraseEnglishSyllableWord=[]
            phraseEnglishSyllableSentence=[]
            
            for i in range(len(phraseDf)):
                currentPhrase= phraseDf['Phrase'].values[i]
                #split into words
                splitPhrase=currentPhrase.split()   
                
                #Save each word with the unicode representation
                for splitPhraseElem in splitPhrase:
                    phraseEnglishSyllableWord.append([splitPhraseElem,unidecode(splitPhraseElem)])
                    
                #Then save the whole sentence with the unicode representation
                phraseEnglishSyllableSentence.append([currentPhrase,unidecode(currentPhrase)])
                
                #update the dataframe with the unidecode representation
                phraseDf['Phrase'].values[i] =unidecode(currentPhrase.lower())               
            self.unideCodeList=phraseEnglishSyllableWord + phraseEnglishSyllableSentence            
            return phraseDf
        except:
            e = sys.exc_info()[1]
            print("ConvertToUnideCodeForList: "+str(e)+"\n")
            raise Exception(fullError)


    def RemoveHTMLElements(self,inputPhrase):
        try:
            cleanedPhrase=BeautifulSoup(inputPhrase,"html.parser").get_text()
            return cleanedPhrase
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveHTMLElments: "+str(e)+"\n"
            raise Exception(fullError)
            
        
    def RemoveHTMLElementsForList(self,inputListDf):
        try:
            htmlCleanedData=[BeautifulSoup(inputListDf.loc[i,].item(),"html.parser").get_text()  for (i, row) in inputListDf.iterrows()]
            return htmlCleanedData
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveHTMLElementsForList: "+str(e)+"\n"
            raise Exception(fullError)
            
            
    def RemoveHTMLElementsForDf(self,inputListDf):
        try:
            for i in range(len(inputListDf)):
                currentPhase= inputListDf['Phrase'].values[i]
                inputListDf['Phrase'].values[i] =BeautifulSoup(currentPhase,"html.parser").get_text()            
            return inputListDf
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveHTMLElementsForDf: "+str(e)+"\n"
            raise Exception(fullError)

    def RemoveSpecialCharacters(self,inputPhrase,pronounsList):
        try:
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            cleanedList = []       
            tokenizedList = []
            punctuationSplittedList=[]
            punctuationRemovedString = ""
            for splitPhrase in inputPhrase.split():
                splitPhrase = re.sub('\@\w+', '', splitPhrase)
                splitPhrase = re.sub('\#\w+','', splitPhrase)
                splitPhrase = re.sub('\#','',splitPhrase)
                splitPhrase = re.sub('RT','',splitPhrase)
                splitPhrase = re.sub('&amp;','',splitPhrase)
                splitPhrase = re.sub('[0-9]+','',splitPhrase)
                splitPhrase = re.sub('//t.co/\w+','',splitPhrase)
                splitPhrase = re.sub('w//','',splitPhrase)
                splitPhrase = splitPhrase.lower()                
                tokenizedList.append(splitPhrase.split())

            for tokenizedElem in tokenizedList:
                    punctuation_Removed_Elem=regex.sub('', str(tokenizedElem))
                    punctuationSplittedList.append(punctuation_Removed_Elem)                    
                                       
            for elem in punctuationSplittedList:         
                if  elem not in pronounsList:
                    punctuationRemovedString+=(" "+elem)                            
                    
            return punctuationRemovedString
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveSpecialCharacters: "+str(e)+"\n"
            raise Exception(fullError)
           
        
    def RemoveSpecialCharactersForList(self,inputPhrase,pronounsList):
        try:
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            cleanedList = []
           
            for currentPhrase in inputPhrase:
                tokenizedList = []
                punctuationSplittedList=[]
                punctuationRemovedString = ""
                for splitPhrase in currentPhrase.split():
                    splitPhrase = re.sub('\@\w+', '', splitPhrase)
                    splitPhrase = re.sub('\#\w+','', splitPhrase)
                    splitPhrase = re.sub('\#','',splitPhrase)
                    splitPhrase = re.sub('RT','',splitPhrase)
                    splitPhrase = re.sub('&amp;','',splitPhrase)
                    splitPhrase = re.sub('[0-9]+','',splitPhrase)
                    splitPhrase = re.sub('//t.co/\w+','',splitPhrase)
                    splitPhrase = re.sub('w//','',splitPhrase)
                    splitPhrase = splitPhrase.lower()
                    tokenizedList.append(splitPhrase.split())   
                    
                for tokenizedElem in tokenizedList:
                    punctuation_Removed_Elem=regex.sub('', str(tokenizedElem))
                    punctuationSplittedList.append(punctuation_Removed_Elem)
                    
                                       
                for elem in punctuationSplittedList:         
                    if  elem not in pronounsList:
                        punctuationRemovedString+=(" "+elem)                        
                        
                cleanedList.append(punctuationRemovedString)                 
            return cleanedList
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveSpecialCharactersForList: "+str(e)+"\n"
            raise Exception(fullError)
           
            
    def RemoveSpecialCharactersForDf(self,inputListDf,pronounsList):
        try:
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            cleanedList = []
           
            for i in range(len(inputListDf)):
                currentPhrase= inputListDf['Phrase'].values[i]
                tokenizedList = []
                punctuationSplittedList=[]
                punctuationRemovedString = ""
                for splitPhrase in currentPhrase.split():
                    splitPhrase = re.sub('\@\w+', '', splitPhrase)
                    splitPhrase = re.sub('\#\w+','', splitPhrase)
                    splitPhrase = re.sub('\#','',splitPhrase)
                    splitPhrase = re.sub('RT','',splitPhrase)
                    splitPhrase = re.sub('&amp;','',splitPhrase)
                    splitPhrase = re.sub('[0-9]+','',splitPhrase)
                    splitPhrase = re.sub('//t.co/\w+','',splitPhrase)
                    splitPhrase = re.sub('w//','',splitPhrase)
                    splitPhrase = splitPhrase.lower()
                    tokenizedList.append(splitPhrase.split())   
                    
                for tokenizedElem in tokenizedList:
                    punctuation_Removed_Elem=regex.sub('', str(tokenizedElem))
                    punctuationSplittedList.append(punctuation_Removed_Elem)
                    
                                       
                for elem in punctuationSplittedList:         
                    if  elem not in pronounsList:
                        punctuationRemovedString+=(" "+elem.lower())
                        
                        
                inputListDf['Phrase'].values[i]=punctuationRemovedString                
            return inputListDf
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="RemoveSpecialCharactersForDf: "+str(e)+"\n"
            raise Exception(fullError)
            
            
    def RemoveNonWordEntriesForList(self,datasetParam):
        try:
            nonWordEntryList=[]
            #Remove non words
            for chkSentence in datasetParam:   
                #Then append back after checking for numerical values
                nonWordEntryList.append((' '.join([word for word in chkSentence.split() if word.isdigit()==False])))     
            return nonWordEntryList   
        except Exception as e:
            e = sys.exc_info()[1]
            fullError="RemoveNonWordEntriesForList: "+str(e)+"\n"
            raise Exception(fullError)
            
            
    def RemoveNonWordEntriesForDf(self,inputDf):
        try:

            nonWordEntryList=[]
            #Remove non words
            for i in range(len(inputDf)):
                currentPhase= inputDf['Phrase'].values[i]    
                #Then append back after checking for numerical values
                inputDf['Phrase'].values[i]=(' '.join([word for word in currentPhase.split() if word.isdigit()==False]))       
            return inputDf     
        except Exception as e:
            e = sys.exc_info()[1]
            fullError="RemoveNonWordEntriesForDf: "+str(e)+"\n"
            raise Exception(fullError)
        
        
    def RemoveNonWordEntries(self,chkSentence):
        try:        
            chkSentence_NonWord_Filtered=(' '.join([word for word in chkSentence.split() if word.isdigit()==False]))    
            return chkSentence_NonWord_Filtered    
        except Exception as e:
            e = sys.exc_info()[1]
            fullError="RemoveNonWordEntries: "+str(e)+"\n"
            raise Exception(fullError)
            
    

class FeatureExtractor:   
    
              
    def VectorizeText(self,xTrain):
        try:          
            token = RegexpTokenizer(r'[a-zA-Z0-9]+')
            count_vect = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                                                encoding='utf-8', input='content',
                                                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                                                ngram_range=(1,2), preprocessor=None, stop_words=None,
                                                strip_accents=None,tokenizer=token.tokenize, vocabulary=None)
            
            X_train_counts = count_vect.fit_transform(xTrain)
            tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

            return count_vect,X_train_tfidf

        except Exception as e:
            e = sys.exc_info()[1]
            fullError="VectorizeText: "+str(e)
            raise Exception(fullError)
        

class PredictionInitializer:
    trainingDataPath=None
    testDataPath=None
    pronounsListPath=None
    trainingData_df=None
    trainingData_df_y_positive=None
    trainingData_df_y_negative=None
    testData_df=None
    testData_target=None
    pronounsList=None

    def SetTrainingAndTestingData(self,trainingDataPathParam,testDataPathParam,pronounsListPath):
        try: 
            
            self.trainingData_df= pd.read_csv(trainingDataPathParam,usecols=[1,2],header='infer',encoding='utf-8') 
            pronouns_df=pd.read_csv(pronounsListPath,usecols=[0],header=None,encoding='utf-8')            
            self.pronounsList=pronouns_df[0].values.tolist()
        except Exception as e:        
            e = sys.exc_info()[1]
            fullError="PredictionInitializer-SetTrainingAndTestingData: "+str(e)+"\n"
            return fullError


class SentimentPredictor:
    
    list_Classifier_Names=[]   
    TrainingDataSet=None
    TestingDataSet=None
    predictionModel=None
    countVecObj=None
    xTrainTfIdf=None
    
                 
    
    def ValidatePhraseIsNullOrEmpty (self,phraseParam):
        if phraseParam and phraseParam.strip():
           #myString is not None AND myString is not empty or blank
           return False        
        #myString is None OR myString is empty or blank
        return True    

    def BuildPredictionModel(self,xTrain,yTrain):   
        try:

            modelPath='/var/www/FlaskApps/GeneralSentimentDetector/Hate_speech_detection_model_new.h5'
            countVecObjPath='/var/www/FlaskApps/GeneralSentimentDetector/countVectorizer.pkl'
            tfIdfObjPath='/var/www/FlaskApps/GeneralSentimentDetector/tfIdfObj.pkl'

            dt=DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                            max_depth=3, max_features=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=10,
                                            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                            splitter='best')
            
            rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                                        max_depth=3, max_features='sqrt', max_leaf_nodes=None,
                                                        min_impurity_decrease=0.0, min_impurity_split=None,
                                                        min_samples_leaf=1, min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
                                                        oob_score=False, random_state=0, verbose=0, warm_start=False)

            ad=AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                                  learning_rate=0.05, n_estimators=16, random_state=None)

            lg=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                  intercept_scaling=1, max_iter=100, multi_class='multinomial',
                                                  n_jobs=1, penalty='l1', random_state=0, solver='saga',
                                                  tol=0.0001, verbose=0, warm_start=False)
        
            eclf=VotingClassifier(estimators=[('dt', dt), ('rf', rf), ('ad', ad),('lg',lg)], voting='soft')


            fe=FeatureExtractor()
            count_vect,xTrainTfidf=fe.VectorizeText(xTrain)            
            built_classifier = eclf.fit(xTrainTfidf, yTrain)
            pickle.dump(built_classifier, open(modelPath, 'wb'))
            pickle.dump(count_vect, open(countVecObjPath, 'wb'))
            pickle.dump(xTrainTfidf, open(tfIdfObjPath, 'wb'))
            modelBuildingStatus="Success"
            return modelBuildingStatus,count_vect,xTrainTfidf          
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="BuildClassifier: "+str(e)
            raise Exception(fullError) 


    def GetSentimentForPhrase(self,phraseParam,classifierParam):
        try:
            formatted_input=pd.Series(data=phraseParam)
            predictedVal=classifierParam.predict(formatted_input)#classifierParam.predict(count_vect.transform([unidecode(phraseParam)]))
            #print(predictedVal)
            predictedSentiment=""
            #print(predictedVal)                         
            if(predictedVal=="YES"):
                predictedSentiment="Negative"   
            else:
                predictedSentiment="Positive"   
                
            return predictedSentiment        
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="GetSentimentFromPhrase: "+str(e)
            raise Exception(fullError)           

    def GetSentimantForPhrase(self,count_vect,phraseParam,classifierParam):
        try:        
            predictedVal=classifierParam.predict(count_vect.transform([unidecode(phraseParam)]))
            predictedSentiment=""
            print(predictedVal)
            if(predictedVal=="YES"):
                predictedSentiment="Negative"
            else:
                predictedSentiment="Positive"
            return predictedSentiment
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="GetSentimentFromPhrase: "+str(e)
            raise Exception(fullError)   
    

    def GetSentimentForText(self,count_vect,phraseParam,classifierParam):
        try:
            predictedVal=classifierParam.predict(count_vect.transform([unidecode(phraseParam)]))
            predictedSentiment=""
            #print(predictedVal)
            if(predictedVal=="YES"):
                predictedSentiment="Negative"
            else:
                predictedSentiment="Positive"
            return predictedSentiment
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="GetSentimentFromPhrase: "+str(e)
            raise Exception(fullError)
            
            
    def ValidateSpeechForHateContent(self,uploadedPhrase):
        try:
            predictionResultStr=None          
                        
            if(self.ValidatePhraseIsNullOrEmpty(uploadedPhrase)==True):                
                predictionResultStr="Error: The received string is empty"                
            else:
                predictionInitializer=PredictionInitializer()
                predictionInitializer.trainingDataPath="https://s3-ap-southeast-1.amazonaws.com/hatespeechcorpusmsc/MScHateSpeechRefined.csv"
                predictionInitializer.testDataPath="https://s3-ap-southeast-1.amazonaws.com/hatespeechcorpusmsc/TestSpeechPhrases.csv"
                predictionInitializer.pronounsListPath="https://s3-ap-southeast-1.amazonaws.com/hatespeechcorpusmsc/Pronouns.csv"
                predictionInitializer.SetTrainingAndTestingData(predictionInitializer.trainingDataPath,predictionInitializer.testDataPath,predictionInitializer.pronounsListPath)

                #Operation 1 : Cleaning the data 
                dataCleaner=DataCleaner()
                                
                htmlCleanedTraining_Df=dataCleaner.RemoveHTMLElementsForDf(predictionInitializer.trainingData_df)
                htmlCleanedText=dataCleaner.RemoveHTMLElements(uploadedPhrase)
                            
                unideCode_Training_Df = dataCleaner.ConvertToUnideCodeForDf(htmlCleanedTraining_Df)
                unidecode_Text=dataCleaner.ConvertToUnideCode(htmlCleanedText)
                 
        
                specialSymbolCleanedPhrases_Df=dataCleaner.RemoveSpecialCharactersForDf(unideCode_Training_Df,predictionInitializer.pronounsList)
                specialSymbolCleanedPhrase=dataCleaner.RemoveSpecialCharacters(unidecode_Text,predictionInitializer.pronounsList)              
                speechCleaned_NonWord_Df=dataCleaner.RemoveNonWordEntriesForDf(specialSymbolCleanedPhrases_Df)
                speechCleaned_NonWord_Phrase=dataCleaner.RemoveNonWordEntries(specialSymbolCleanedPhrase)
                speechCleaned_NonWord_Phrase=speechCleaned_NonWord_Phrase.lower()                
               if(len(speechCleaned_NonWord_Phrase)==0):
                    return "Positive"
                
                #----------------------------------------------------------------------------------------------------
                modelPath='/var/www/FlaskApps/GeneralSentimentDetector/Hate_speech_detection_model_new.h5'
                countVecObjPath='/var/www/FlaskApps/GeneralSentimentDetector/countVectorizer.pkl'
                tfIdfObjPath='/var/www/FlaskApps/GeneralSentimentDetector/tfIdfObj.pkl'
                clf_for_prediction=pickle.load(open(modelPath, 'rb'))
                count_vect=pickle.load(open(countVecObjPath, 'rb'))
                X_train_tfidf=pickle.load(open(tfIdfObjPath, 'rb'))              

                #Operation 2 :Build up the classifier
                sp=SentimentPredictor()                                  
                
                #Operation 3 : Predicting the sentiment
                predictionResultStr =sp.GetSentimentForText(count_vect,speechCleaned_NonWord_Phrase,clf_for_prediction)               
                #----------------------------------------------------------------------------------------------
                return predictionResultStr #Response(json.dumps(predictionResultStr),  mimetype='application/json') 
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="ValidateSpeechForHateContent: "+str(e)
            raise Exception(fullError)
            
            

class LexiconBuilder:
    
    adminSecretKey="ABGBALP11F"
    
    def ValidateLexiconIsNullOrEmpty(self,textParam):
        try:
            if textParam and textParam.strip():
               #myString is not None AND myString is not empty or blank
               return True

            #myString is None OR myString is empty or blank
            return False
        
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-ValidateLexiconIsNullOrEmpty: "+str(e)
            raise Exception(fullError)
    
    def ValidateLexiconSpecialCharacters(self,textParam):
        try: 
            isOnlySpecialCharacters=all(i in string.punctuation for i in textParam)
            
            if (isOnlySpecialCharacters==True):
                return False
            else:
                return True
            
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-ValidateLexiconSpecialCharacters: "+str(e)
            raise Exception(fullError)
            
    def SaveLexiconToRepository(self,sqlCon,lexiconContentOriginal,lexiconContentUnidecode,applicationId):
        try:   
            
            
            dbCursor = sqlCon.cursor()
            
            currentTimeStamp = datetime.datetime.now()#.astimezone(timezone('Asia/Colombo'))

            currentTimeStampStr=str(currentTimeStamp).split('.')[0]
                      
           
            
            query="SELECT * FROM LexiconStore WHERE LOWER(LexiconContentUnidecode)=:lexiconContentUnidecode AND ApplicationID=:applicationId "       
                
            dbCursor.execute(query,{"lexiconContentUnidecode": lexiconContentUnidecode.lower(),"applicationId":applicationId.lower()})   
            
            resultRows=dbCursor.fetchall()

            if(len(resultRows)==0):
                
                lexiconStoreRecord = (lexiconContentOriginal, lexiconContentUnidecode,currentTimeStampStr,applicationId)
                        
                dbCursor.execute('INSERT INTO LexiconStore(LexiconContentOriginal,LexiconContentUnidecode,EntryTimeStamp,ApplicationID) VALUES (?,?,?,?)', lexiconStoreRecord)
                #query="INSERT INTO LexiconStore(LexiconContentOriginal,LexiconContentUnidecode,EntryTimeStamp) "+
                        #"VALUES('"+lexiconContentOriginal+"','"+lexiconContentUnidecode+"','"+currentTimeStampStr+"')"
                #dbCursor.execute(query)
            


         
            sqlCon.commit()           
            
            
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-SaveLexiconToRepository: "+str(e)
            raise Exception(fullError)
            
    def SaveSentimentDetectionLog(self,sqlCon,phraseContentOriginal,phraseContentUnidecode,phraseSentiment,predictionTOC,applicationId):
        try:            
            dbCursor = sqlCon.cursor()
            currentTimeStamp = datetime.datetime.now()
            currentTimeStampStr=str(currentTimeStamp)
            sentimentLogRecord=(phraseContentOriginal,phraseContentUnidecode,phraseSentiment,currentTimeStampStr,predictionTOC,applicationId)                   
            dbCursor.execute('INSERT INTO SentimentDetectionLog(PhraseContentOriginal,PhraseContentUnidecode,Sentiment,LogTimeStamp,PredictionTOC,ApplicationID) VALUES (?,?,?,?,?,?)', sentimentLogRecord)
            sqlCon.commit()
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-SaveSentimentDetectionLog: "+str(e)
            raise Exception(fullError)

    def DeleteAllSentimentDetectionLogRecordsForApplication(self,sqlCon,adminSecretKey,applicationId):
        try:

	    if(adminSecretKey.lower()==self.adminSecretKey.lower()):

		 dbCursor = sqlCon.cursor()
		 query="DELETE FROM SentimentDetectionLog WHERE LOWER(ApplicationID)=:applicationId"
		 dbCursor.execute(query,{"applicationId": applicationId.lower()})
		 result= dbCursor.fetchall()
         sqlCon.commit()
         return "Successfully deleted all sentiment detection records for the application"
	    else:
	         return "Error:Invalid administrator secret key."
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-DeleteAllSentimentDetectionLogRecordsForApplication: "+str(e)
            raise Exception(fullError)

            
            
    def GetAllLexicons(self,sqlCon):
        try:   
            resultList=[]             
            dbCursor = sqlCon.cursor()                
            query="SELECT * FROM LexiconStore"       
            dbCursor.execute(query)       
            result= dbCursor.fetchall()
            constructorList=[]
            for elem in result:
                dictionaryObj={"LexiconStoreID":elem[0],
                               "LexiconContentOriginal":elem[1],
                               "LexiconContentUnidecode":elem[2],
                               "EntryTimeStamp":elem[3],
                               "ApplicationID":elem[4]
                              }                
                constructorList.append(dictionaryObj)                  
            sqlCon.commit()           
            return json.dumps(constructorList,ensure_ascii=False)
            
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-InsertLexiconTuple: "+str(e)
            raise Exception(fullError)
    

    def DeleteAllLexiconsOfApplication(self,sqlCon,adminSecretKey,applicationId):
        try:

            if(adminSecretKey.lower()==self.adminSecretKey.lower()):

                 dbCursor = sqlCon.cursor()
                 currentTimeStamp = datetime.datetime.now()
                 currentTimeStampStr=str(currentTimeStamp)
                 query="DELETE FROM LexiconStore"# WHERE ApplicationID=:applicationId"
                 dbCursor.execute(query,{"applicationId":applicationId.lower()})
                 sqlCon.commit()
                 return "Successfully deleted all lexicons for the application"
            else:
                 return "Invalid administrator secret key.This key is required to get all the log records."
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-DeleteAllLexiconsOfApplication: "+str(e)
            raise Exception(fullError)

        
    def GetAllSentimentDetectionLogRecords(self,sqlCon,adminSecretKey):
        try: 
            
            resultList=[]                            
            dbCursor = sqlCon.cursor()         
            
            if(adminSecretKey.lower()==self.adminSecretKey.lower()):
                                  
                 query="SELECT * FROM SentimentDetectionLog" 

                 dbCursor.execute(query)

                 resultSentimentLog= dbCursor.fetchall()       

                 constructorList=[]

                                  
                 for elem in resultSentimentLog:
                     dictionaryObj={"SentimentDetectionLogID":elem[0],
                                   "PhraseContentOriginal":elem[1],
                                   "PhraseContentUnidecode":elem[2],
                                   "Sentiment":elem[3],
                                   "LogTimeStamp":elem[4],
                                   "PredictionTOC":elem[5],
                                   "ApplicationID":elem[6],
                                   }                
                     constructorList.append(dictionaryObj)
                   
                 
                 sqlCon.commit()           


                 return json.dumps(constructorList,ensure_ascii=False)
            
            else:
                 return json.dumps("Invalid administrator secret key.This key is required to get all the log records.",ensure_ascii=False)
            
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-GetSentimentDetextionLogRecords: "+str(e)
            raise Exception(fullError)
    
    
    def GetSentimentDetectionLogRecordsForApplication(self,sqlCon,applicationId):
        try: 
            
            resultList=[]              
            
            dbCursor = sqlCon.cursor()                                
                                  
            query="SELECT * FROM SentimentDetectionLog WHERE LOWER(ApplicationID)=:applicationId"       
                
            dbCursor.execute(query,{"applicationId": applicationId.lower()})    
                
            constructorList=[]
                
            for elem in result:
                dictionaryObj={"SentimentDetectionLogID":elem[0],
                                   "PhraseContentOriginal":elem[1],
                                   "PhraseContentUnidecode":elem[2],
                                   "Sentiment":elem[3],
                                   "LogTimeStamp":elem[4],
                                   "PredictionTOC":elem[5],
                                   "ApplicationID":elem[6],
                                   }                
                constructorList.append(dictionaryObj)
                   
         
            sqlCon.commit()           


            return json.dumps(constructorList,ensure_ascii=False)      
          
            
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="LexiconBuilder-GetSentimentDetectionLogRecordsForApplication: "+str(e)
            raise Exception(fullError)

class Payload(object):
    def __init__(self, j):
        try:
            self.__dict__ = json.loads(j)
        except Exception as e :
            e = sys.exc_info()[1]
            raise Exception(fullError)

class HSAPICryptography:
    publicKey="IMLBKTCBSXYGEKUA"
    
    def EncryptText(self,inputString):
        try:
            inputString_Adjusted = inputString.rjust(64)
            cipher = AES.new(self.publicKey,AES.MODE_ECB) # never use ECB in strong systems obviously
            encoded = base64.b64encode(cipher.encrypt(inputString_Adjusted))        
            return encoded
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="HSAPICryptography-EncryptText: "+str(e)
            raise Exception(fullError)
    
    def DecryptText(self,encryptedVal):
        try:
            cipher = AES.new(self.publicKey,AES.MODE_ECB)
            decoded = cipher.decrypt(base64.b64decode(encryptedVal))
            return decoded.strip().decode("utf-8")
        except Exception as e :
            e = sys.exc_info()[1]
            fullError="HSAPICryptography-DecryptText: "+str(e)
            raise Exception(fullError)      


class APIClientApplicationRegistry:
    adminSecretKey="ABGBALP11F"

    def RegisterClientAppplication(self,sqlCon,adminSecretKeyParam,applicationName):
        try:
            returnString=""

            if(adminSecretKeyParam.lower()==self.adminSecretKey.lower()):

                dbCursor = sqlCon.cursor()

                query="SELECT * FROM ClientApplicationRegistry WHERE LOWER(ApplicationName)=:applicationName"

                dbCursor.execute(query,{"applicationName": applicationName.lower()})

                resultRows=dbCursor.fetchall()

                if(len(resultRows)==0):

                    currentTimeStamp = str(datetime.datetime.now())

                    #create the application ID
                    md5StringVal = hashlib.md5()
                    md5StringVal.update(applicationName.lower().encode('utf-8'))
                    newApplicationId=md5StringVal.hexdigest()

                    #create the application token
                    cryptObj=HSAPICryptography()

                    encryptedVal=cryptObj.EncryptText((applicationName.lower()+"##"+currentTimeStamp))
                    newAPIToken=encryptedVal.decode("utf-8")

                    clientAppRecord=(applicationName,newApplicationId,newAPIToken,currentTimeStamp)

                    dbCursor.execute('INSERT INTO ClientApplicationRegistry(ApplicationName,ApplicationID,APIToken,EntryDate) VALUES (?,?,?,?)', clientAppRecord)

                    sqlCon.commit()

                    returnString="Success:Successfully registered the client application.Please contact the API admin team to receive the API token and the Application ID"

                else:

                    returnString="Error:The application is already registered.Please contact the API Admin team."



            else:
                returnString="Error:Invalid administrator secret key."


            return returnString


        except Exception as e :
            e = sys.exc_info()[1]
            fullError="APIClientApplicationRegistry-RegisterClientAppplication: "+str(e)
            raise Exception(fullError)


    def UpdateClientApplicationAPIToken(self,sqlCon,adminSecretKeyParam,applicationName):
        try:
            returnString=""

            if(adminSecretKeyParam.lower()==self.adminSecretKey.lower()):

                dbCursor = sqlCon.cursor()

                query="SELECT * FROM ClientApplicationRegistry WHERE LOWER(ApplicationName)=:applicationName"

                dbCursor.execute(query,{"applicationName": applicationName.lower()})

                sqlCon.commit()

                resultRows=dbCursor.fetchall()

                if(len(resultRows)==0):
                    returnString="No application named "+applicationName+" is available in our registry"

                else:

                    currentTimeStamp = str(datetime.datetime.now())

                    #create the application token
                    cryptObj=HSAPICryptography()

                    encryptedVal=cryptObj.EncryptText((applicationName.lower()+"##"+currentTimeStamp))
                    newAPIToken=encryptedVal.decode("utf-8")

                    query="UPDATE ClientApplicationRegistry SET APIToken=:apiToken,LastUpdatedDate=:lastUpdatedDate WHERE LOWER(ApplicationName)=:applicationName"

                    dbCursor.execute(query,{"apiToken":newAPIToken,
                                            "applicationName": applicationName.lower(),
                                            "lastUpdatedDate":currentTimeStamp })
                    sqlCon.commit()

                    returnString="Success:Successfully updated the client application's API Token.Please contact the API admin team to receive the API token "


            else:
                returnString="Error:Invalid administrator secret key."



            return returnString

        except Exception as e :
            e = sys.exc_info()[1]
            fullError="APIClientApplicationRegistry-UpdateClientApplication: "+str(e)
            raise Exception(fullError)

    def IsValidToken(self,sqlCon,applicationId,apiToken):
        try:
            dbCursor = sqlCon.cursor()

            query="SELECT * FROM ClientApplicationRegistry WHERE LOWER(ApplicationID)=:applicationID AND LOWER(APIToken)=:apiToken"

            dbCursor.execute(query,{"applicationID": applicationId.lower(),"apiToken":apiToken.lower()})

            resultRows=dbCursor.fetchall()

            if(len(resultRows)>0):
                return True

            return False


        except Exception as e :
            e = sys.exc_info()[1]
            fullError="APIClientApplicationRegistry-IsValidToken: "+str(e)
            raise Exception(fullError)



    def GetRegisteredClientApplications(self,sqlCon,adminSecretKeyParam):
        try:
            if(adminSecretKeyParam.lower()==self.adminSecretKey.lower()):

                dbCursor = sqlCon.cursor()

                query="SELECT * FROM ClientApplicationRegistry"

                dbCursor.execute(query)
                
                result = dbCursor.fetchall()

                constructorList=[]

                for elem in result:
                    dictionaryObj={"ClientApplicationRegistryID":elem[0],
                                   "ApplicationName":elem[1],
                                   "ApplicationID":elem[2],
                                   "APIToken":elem[3],
                                   "EntryDate":elem[4],
                                   "LastUpdatedDate":elem[5]
                                   }
                    constructorList.append(dictionaryObj)


                sqlCon.commit()


                return json.dumps(constructorList,ensure_ascii=False)

            else:
                returnString="Error:Invalid administrator secret key."

        except Exception as e :
            e = sys.exc_info()[1]
            fullError="APIClientApplicationRegistry-GetRegisteredClientApplications: "+str(e)
            raise Exception(fullError)

#---------------------End of class declarations--------------------------------------------------------------------------------

#---------------------------------------------API Call Definitions-------------------------------------------------------------------
@app.route('/',methods=['GET'])
def home():
    try:
        now=datetime.datetime.now()
        message = str(now)
        versionVal=str(sys.version_info[0])+"."+str(sys.version_info[1])+"\n"+str(sys.path)
        #return versionVal
        return render_template("HateSpeechDetectorHomePage.html",message=message)
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/home/v1: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')
        # return render_template("HateSpeechDetectorHomePage.html")
        # versionVal=str(sys.version_info[0])+"."+str(sys.version_info[1])+"\n"+str(sys.path)
        #return versionVal



@app.route('/api/hatespeechdetection/v1/ProcessTrainingAndTestData', methods=['POST'])
def ProcessTrainingAndTestData():
    try:
        trainingDataPath=postDataDeserialized.trainingDataPath
        testDataPath=postDataDeserialized.testDataPath
        pronounsListPath=postDataDeserialized.pronounsListPath
        predictionInitializer=PredictionInitializer()
        predictionInitializer.trainingDataPath=trainingDataPath
        predictionInitializer.testDataPath=testDataPath
        predictionInitializer.pronounsListPath=pronounsListPath
        predictionInitializer.SetTrainingAndTestingData(predictionInitializer.trainingDataPath,
                                                        predictionInitializer.testDataPath,
                                                        predictionInitializer.pronounsListPath)

        dataCleaner=DataCleaner()
        htmlCleanedTrainingPhrases=dataCleaner.RemoveHTMLElementsForList(predictionInitializer.trainingData_df)
        htmlCleanedTestPhrases=dataCleaner.RemoveHTMLElementsForList(predictionInitializer.testData_df)
        specialSymbolCleanedPhrases=dataCleaner.RemoveSpecialCharactersForList(htmlCleanedTrainingPhrases,
                                                                               predictionInitializer.pronounsList)
        specialSymbolCleanedTestPhrases=dataCleaner.RemoveSpecialCharactersForList(htmlCleanedTestPhrases,
                                                                                   predictionInitializer.pronounsList)
        speechCleaned_NonWord=dataCleaner.RemoveNonWordEntriesForList(specialSymbolCleanedPhrases)
        speechCleaned_NonWord_Test=dataCleaner.RemoveNonWordEntriesForList(specialSymbolCleanedTestPhrases)
        return Response(json.dumps(speechCleaned_NonWord),  mimetype='application/json')
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/ProcessTrainingAndTestData: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')
        # return render_template("HateSpeechDetectorHomePage.html")
        # versionVal=str(sys.version_info[0])+"."+str(sys.version_info[1])+"\n"+str(sys.path)
        #return versionVal
        # return '''<h1>Python Hate Speeh Detection API </h1><p>A prototype API to detect sentiment in posts,comments and phrases posted in Social Media.</p>'''



@app.route('/api/hatespeechdetection/v1/BuildAndSavePredictionModel', methods=['POST'])  
def BuildAndSavePredictionModel():
    try:
        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey
        trainingDataPath=postDataDeserialized.trainingDataPath
        testDataPath=postDataDeserialized.testDataPath
        pronounsListPath=postDataDeserialized.pronounsListPath        
        filePath='var/www/FlaskApps/GeneralSentimentDetector/Hate_speech_detection_model.h5'       
        apiClientReg=APIClientApplicationRegistry()
        
        if adminSecretKey==apiClientReg.adminSecretKey:
            
            predictionInitializer=PredictionInitializer()
            predictionInitializer.trainingDataPath=trainingDataPath
            predictionInitializer.testDataPath=testDataPath
            predictionInitializer.pronounsListPath=pronounsListPath
            predictionInitializer.SetTrainingAndTestingData(predictionInitializer.trainingDataPath,
                                                            predictionInitializer.testDataPath,
                                                            predictionInitializer.pronounsListPath)
               
            #Operation 1 : Cleaning the data 
            dataCleaner=DataCleaner()      
            htmlCleanedTraining_Df=dataCleaner.RemoveHTMLElementsForDf(predictionInitializer.trainingData_df)          
            unideCode_Training_Df = dataCleaner.ConvertToUnideCodeForDf(htmlCleanedTraining_Df)                    
            specialSymbolCleanedPhrases_Df=dataCleaner.RemoveSpecialCharactersForDf(unideCode_Training_Df,predictionInitializer.pronounsList)                  
            speechCleaned_NonWord_Df=dataCleaner.RemoveNonWordEntriesForDf(specialSymbolCleanedPhrases_Df)
                                 
            X_train, X_test, y_train, y_test = train_test_split(speechCleaned_NonWord_Df['Phrase'], 
                                                                    speechCleaned_NonWord_Df['IsHateSpeech'], 
                                                                    random_state = 0,test_size=0.2,shuffle=True)
            
            
            sp=SentimentPredictor()            
            modelBuildingStatus,count_vect,xTrainTfidf=sp.BuildPredictionModel(X_train,y_train)            
            resultMessage=modelBuildingStatus+":"+"Model has been built and saved successfully.Now you can use the 'PredictSentimentForPhrase' API call "+ "to predict sentiment in texts"+"\n"+ "You can also retrain the model using an updated dataset." 
        else:            
            resultMessage= "Please enter the correct admin key." +"\n"+"Model training and saving is allowed only for the API Admininstrator only."
        result={"Message":resultMessage}  
        return Response(json.dumps(result),  mimetype='application/json')         
    except Exception, e:
        e = sys.exc_info()[1]
        fullError="Error: "+"/api/hatespeechdetection/v1/BuildAndSavePredictionModel: "+str(e)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')

@app.route('/api/hatespeechdetection/v1/PredictSentimentForPhrase', methods=['POST'])       
def PredictSentimentForPhrase():
    try:
        start_time = time.time()       
        sp=SentimentPredictor()
        postData = request.data
        postDataDeserialized=Payload(postData)
        apiToken=postDataDeserialized.apiToken
        applicationId=postDataDeserialized.applicationId
        uploadedPhrase=postDataDeserialized.phraseToUpload

        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'
        
        sqlCon = sqlite3.connect(filePath)
       
        #Validate the token and the application client ID
        apiClientReg=APIClientApplicationRegistry()
              
        isValidToken=apiClientReg.IsValidToken(sqlCon,applicationId,apiToken)      
        
        if(isValidToken==True):

        #uploadedPhrase=uploadedPhrase.decode('utf-8')

            predictedSentiment=sp.ValidateSpeechForHateContent(uploadedPhrase)

            elapsed_time = time.time() - start_time


            #After the sentiment detection, save the lexicons and build up the lexicon repository
            lb=LexiconBuilder()
            filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'
            #filePath =':memory:'# os.path.abspath(filePath)
            sqlCon = sqlite3.connect(filePath)

            if(lb.ValidateLexiconIsNullOrEmpty(uploadedPhrase)==True and lb.ValidateLexiconSpecialCharacters(uploadedPhrase)==True):
                lb.SaveLexiconToRepository(sqlCon,uploadedPhrase,unidecode(uploadedPhrase),applicationId)

            #Save sentiment detection log
            lb.SaveSentimentDetectionLog(sqlCon,uploadedPhrase,unidecode(uploadedPhrase),predictedSentiment,elapsed_time,applicationId)
            sqlCon.close()

            predictedSentimentAsJson={"PhraseSentiment":predictedSentiment}            
            return Response(json.dumps(predictedSentimentAsJson),  mimetype='application/json')            
        else:

             sqlCon.close()
             outputMessage={"PhraseSentiment":"Error: Token validation failed.Please contact the API admin team."}            
             return Response(json.dumps(outputMessage),  mimetype='application/json')            
    except BaseException as e:
        #e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/PredictSentimentForPhrase: "+str(e)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')



@app.route('/api/hatespeechdetection/v1/GetLexicons', methods=['GET'])       
def GetLexicons():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'        
        sqlCon = sqlite3.connect(filePath)
        
        lb=LexiconBuilder()
        lexiconList=lb.GetAllLexicons(sqlCon)     
        sqlCon.close()
        return Response(lexiconList,mimetype='application/json')         
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/GetLexicons: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')


@app.route('/api/hatespeechdetection/v1/GetAllSentimentDetectionLogRecords', methods=['POST'])    
def GetAllSentimentDetectionLogRecords():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'        
        sqlCon = sqlite3.connect(filePath)
        
        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey
        lb=LexiconBuilder()
        trueAdminKey=lb.adminSecretKey
        dbCursor = sqlCon.cursor()
        lb=LexiconBuilder()
        lexiconList=lb.GetAllSentimentDetectionLogRecords(sqlCon,adminSecretKey)    
        sqlCon.close()
        return Response(lexiconList,mimetype='application/json')
         
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/GetAllSentimentDetectionLogRecords: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')
   

@app.route('/api/hatespeechdetection/v1/RegisterNewApplication', methods=['POST'])
def RegisterNewApplication():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'        
        sqlCon = sqlite3.connect(filePath)        
        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey
        applicationName=postDataDeserialized.applicationName       
        apiClientReg=APIClientApplicationRegistry()
        resultMsg=apiClientReg.RegisterClientAppplication(sqlCon,adminSecretKey,applicationName) 
        outputObj={"Message":resultMsg}
        sqlCon.close()
        return Response(json.dumps(outputObj),mimetype='application/json')
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/RegisterNewApplication: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')
        
@app.route('/api/hatespeechdetection/v1/UpdateClientApplicationAPIToken', methods=['POST'])
def UpdateClientApplicationAPIToken():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'        
        sqlCon = sqlite3.connect(filePath)        
        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey
        applicationId=postDataDeserialized.applicationId
        applicationName=postDataDeserialized.applicationName       
        apiClientReg=APIClientApplicationRegistry()
        resultMsg=apiClientReg.UpdateClientApplicationAPIToken(sqlCon,adminSecretKey,applicationName) 
        outputObj={"Message":resultMsg}
        sqlCon.close()
        return Response(json.dumps(outputObj),mimetype='application/json')
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/UpdateClientApplicationAPIToken: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')


@app.route('/api/hatespeechdetection/v1/GetRegisteredClientApplications', methods=['POST'])       
def GetRegisteredClientApplications():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'        
        sqlCon = sqlite3.connect(filePath)
        
        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey       
        apiClientReg=APIClientApplicationRegistry()
        clientAppList=apiClientReg.GetRegisteredClientApplications(sqlCon,adminSecretKey)       
        sqlCon.close()            
        return Response(clientAppList,mimetype='application/json')         
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/GetRegisteredClientApplications: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')


@app.route('/api/hatespeechdetection/v1/DeleteAllLexiconsOfApplication', methods=['POST'])
def DeleteAllLexiconsOfApplication():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'
        sqlCon = sqlite3.connect(filePath)
        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey
        applicationId=postDataDeserialized.applicationId
        apiClientReg=APIClientApplicationRegistry()
        
        if adminSecretKey==apiClientReg.adminSecretKey:        
            lb=LexiconBuilder()
            deleteResult=lb.DeleteAllLexiconsOfApplication(sqlCon,adminSecretKey,applicationId)
            sqlCon.close()
        else:
            deleteResult= "Please enter the correct admin key." +"\n"+"Lexicon deletion is allowed only for the API Admininstrator."
        return Response(deleteResult,mimetype='application/json')
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/GetAllSentimentDetectionLogRecords: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')


@app.route('/api/hatespeechdetection/v1/DeleteAllSentimentDetectionLogRecordsForApplication', methods=['POST'])
def DeleteAllSentimentDetectionLogRecordsForApplication():
    try:
        #sqlCon = sqlite3.connect('HateSpeechRepository.db')
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'
        sqlCon = sqlite3.connect(filePath)

        postData = request.data
        postDataDeserialized=Payload(postData)
        adminSecretKey=postDataDeserialized.adminSecretKey
        applicationId=postDataDeserialized.applicationId


        lb=LexiconBuilder()

        deleteResult=lb.DeleteAllSentimentDetectionLogRecordsForApplication(sqlCon,adminSecretKey,applicationId)


        sqlCon.close()


        return Response(deleteResult,mimetype='application/json')

    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/DeleteAllSentimentDetectionLogRecordsForApplication: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')

@app.route('/api/hatespeechdetection/v1/GetSchemaObjects', methods=['GET'])
def GetSchemaObjects():
    try:
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'
        sqlCon = sqlite3.connect(filePath)
        dbCursor = sqlCon.cursor()
        dbCursor.execute('''SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;''')
        result= dbCursor.fetchall()
        sqlCon.close()
        return Response(json.dumps(result),  mimetype='application/json')
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/GetSchemaObjects: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')


@app.route('/api/hatespeechdetection/v1/CreateSchemaObject', methods=['POST'])
def CreateSchemaObject():
    try:
        postData = request.data
        postDataDeserialized=Payload(postData)
        queryToExecute=postDataDeserialized.queryParam #Ex: 'CREATE TABLE SentimentDetectionLog(SentimentDetectionLogID INTEGER PRIMARY KEY,PhraseContentOriginal TEXT,PhraseContentUnidecode TEXT,Sentiment TEXT,LogTimeStamp TEXT,PredictionTOC TEXT, ApplicationID TEXT)'
        filePath='var/www/FlaskApps/GeneralSentimentDetector/HateSpeechRepository.db'
        sqlCon = sqlite3.connect(filePath)     
        dbCursor = sqlCon.cursor()      
        dbCursor.execute(queryToExecute)        
        sqlCon.close()
        returnString = "Success..."
        resultMessage={"Message":returnString}      
        finalResult=json.dumps(resultMessage,ensure_ascii=False)
        return Response(finalResult,  mimetype='application/json')
    except Exception as e:
        e = sys.exc_info()[1]
        fullError="/api/hatespeechdetection/v1/CreateSchemaObject: "+str(e.message)+"\n"
        return Response(json.dumps(fullError),  mimetype='application/json')

if __name__=="__main__":
   app.run()