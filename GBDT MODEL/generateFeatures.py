# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:50:25 2019

@author: Ashok
"""

import ngram
import pandas as pd
import numpy as np
#import pickle
from helpers import *
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
import dill as pickle
#from AlignmentFeatureGenerator import *




    

def process():

    read = False
    if not read:
        '''
        body_train = pd.read_csv("train_bodies_processed.csv", encoding='utf-8')
        stances_train = pd.read_csv("train_stances_processed.csv", encoding='utf-8')
        # training set
        train = pd.merge(stances_train, body_train, how='left', on='Body ID')
        
        train.head()
        targets = ['agree', 'disagree', 'discuss', 'unrelated']
        targets_dict = dict(zip(targets, range(len(targets))))
        train['target'] = map(lambda x: targets_dict[x], train['Stance'])
        print ('train.shape:')
        print (train.shape)
        n_train = train.shape[0]
        '''
        #sample starts
        
        sample_head = "Italy culls birds after five H5N8 avian flu outbreaks in October"
        sample_body = "ROME (Reuters) - Italy has had five outbreaks of highly pathogenic H5N8 avian flu in farms the central and northern parts of the country since the start of the month and about 880,000 chickens, ducks and turkeys will be culled, officials said on Wednesday.\
            The biggest outbreak of the H5N8 virus, which led to the death or killing of millions of birds in an outbreak in western Europe last winter, was at a large egg producing farm in the province of Ferrara.\
            The latest outbreak was confirmed on Oct. 6 and about 853,000 hens are due to be culled by Oct. 17, the IZSV zoological institute said.\
            Another involved 14,000 turkeys in the province of Brescia, which are due to be culled by Oct. 13.\
            A third involved 12,400 broiler chickens at a smaller farm in the province of Vicenza and two others were among a small number of hens, ducks, broilers and turkeys on family farms.\
            In those three cases, all the birds have been culled."
        sample_head_pd = pd.DataFrame([sample_head])
        sample_body_pd = pd.DataFrame([sample_body])
        sample_data_pd = pd.concat((sample_head_pd,sample_body_pd),axis = 1)
        sample_data_pd.columns = ['Headline','articleBody']
        sample_data_pd['URLs'] = np.nan
        sample_data_pd['Stance'] = np.nan
    
        
        #sample ends
        
        dataset = pd.read_csv('data.csv')

        dataset.isnull().sum()

        dataset = dataset[pd.notnull(dataset['Body'])]
        
        dataset.columns = ['URLs','Headline','articleBody','Stance']
        
        X_data = dataset.iloc[:,1:3]
        Y_data = dataset.iloc[:,3]
        
        from sklearn.cross_validation import train_test_split

        X_train,X_test,Y_train,Y_test = train_test_split(X_data, Y_data, test_size = 0.25, random_state = 0)
        

        train = pd.concat([X_train,Y_train],axis = 1)
        
        train.to_csv('gdbt_training_input.csv', index = False)
        
        
        X_test.to_csv('gdbt_testing_input.csv', index = False)
        Y_test = pd.DataFrame(Y_test)
        Y_test.to_csv('gdbt_testing_ouput.csv', index = False)
        
        targets = ['Fake', 'Real']
        targets_dict = dict(zip(targets, range(len(targets))))
        train['target'] = map(lambda x: targets_dict[x], train['Stance'])
        
        
        
        data = train
        
        
        # read test set, no 'Stance' column in test set -> target = NULL
        # concatenate training and test set
        test_flag = True
        
        if test_flag:
            '''
            body_test = pd.read_csv("test_bodies_processed.csv", encoding='utf-8')
            headline_test = pd.read_csv("test_stances_unlabeled.csv", encoding='utf-8')
            test = pd.merge(headline_test, body_test, how="left", on="Body ID")
            '''
            data = pd.concat((train, X_test)) # target = NaN for test set
            #print (data)
            print ('data.shape:')
            print (data.shape)

            train = data[~data['target'].isnull()]
            print (train)
            print ('train.shape:')
            print (train.shape)
            
            test = data[data['target'].isnull()]
            print (test)
            print ('test.shape:')
            print (test.shape)

        #data = data.iloc[:100, :]
        
        #return 1
        
        print ("generate unigram")
        data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))
        print(data.head())
        data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))

        print ("generate bigram")
        join_str = "_"
        data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: ngram.getBigram(x, join_str))
        data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: ngram.getBigram(x, join_str))
        
        print ("generate trigram")
        join_str = "_"
        data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
        data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
        
        with open('data.pkl', 'wb') as outfile:
            pickle.dump(data, outfile, -1)
            print ('dataframe saved in data.pkl')

    else:
        with open('data.pkl', 'rb') as infile:
            data = pickle.load(infile)
            print ('data loaded')
            print ('data.shape:')
            print (data.shape)
    #return 1

    # define feature generators
    countFG    = CountFeatureGenerator()
    tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG    = SentimentFeatureGenerator()
    #walignFG   = AlignmentFeatureGenerator()
    generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    #generators = [svdFG, word2vecFG, sentiFG]
    #generators = [tfidfFG]
    #generators = [countFG]
    #generators = [walignFG]
    
    
    #countFG.process(data)
    #countFG.read()
    
    #word2vecFG.process(data)
    
    #sentiFG.process(data)
    
    
    
    for g in generators:
        g.process(data)
    
    for g in generators:
        g.read('train')
    
    for g in generators:
        g.read('test')

    print ('done')


if __name__ == "__main__":
    
    process()
