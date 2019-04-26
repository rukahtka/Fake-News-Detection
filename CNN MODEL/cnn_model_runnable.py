# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:04:34 2019

@author: Ashok
"""

import numpy as np 
import pandas as pd 
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
import gensim
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import model_from_json




MAX_HEAD_SEQUENCE_LENGTH = 120
MAX_BODY_SEQUENCE_LENGTH = 6000
MAX_NB_WORDS1 = 49433
EMBEDDING_DIM = 300

#trainig_data_file = 'train.csv'
#testing_data_file = 'test.csv'

LABELS = ['fake', 'real']







def build_model(head_embeddings, body_embeddings, max_head_sequence, max_body_sequence, head_num_words, body_num_words, 
                embedding_dim, trainable = False):
    
    
    
        
    embedding_layer_head = Embedding(head_num_words,
                            embedding_dim,
                            weights=[head_embeddings],
                            input_length=MAX_HEAD_SEQUENCE_LENGTH,
                            trainable=trainable)
    
    
    embedding_layer_body = Embedding(body_num_words,
                            embedding_dim,
                            weights=[body_embeddings],
                            input_length=MAX_BODY_SEQUENCE_LENGTH,
                            trainable=trainable)
    
    #taking input, Headline and Body in embedded form
    
    headline = Input((max_head_sequence,))
    body = Input((max_body_sequence,))
    
    headline_tensor = embedding_layer_head(headline)
    body_tensor = embedding_layer_body(body)
    

    #declaration of convolution and max pooling layers
    
    c0 = Conv1D(filters = 256, kernel_size = 3, padding='same', activation='relu')
    c1 = Conv1D(filters = 256, kernel_size = 3, padding='same', activation='relu')
    c2 = Conv1D(filters = 512, kernel_size = 4, padding='same', activation='relu')
    c3 = Conv1D(filters = 512, kernel_size = 4, padding='same', activation='relu')
    c4 = Conv1D(filters = 768, kernel_size = 5, padding='same', activation='relu')
    
    p0 = MaxPooling1D()
    p1 = MaxPooling1D()
    p2 = MaxPooling1D()
    
    #applying convolution and max pooling to headline
    
    c0_o = c0(headline_tensor)
    c0_o = Dropout(0.5)(c0_o)
    p0_o = p0(c0_o)
    c1_o = c1(p0_o)
    c1_o = Dropout(0.5)(c1_o)
    p1_o = p1(c1_o)
    c2_o = c2(p1_o)
    c2_o = Dropout(0.5)(c2_o)
    p2_o = p2(c2_o)
    c3_o = c3(p2_o)
    c3_o = Dropout(0.5)(c3_o)
    #head_convolution = c4(c3_o)
    c4_o = c4(c3_o)
    head_convolution = Dropout(0.5)(c4_o)
    
    
    #applyting convolution and max pooling to body 
    
    c0_o = c0(body_tensor)
    c0_o = Dropout(0.5)(c0_o)
    p0_o = p0(c0_o)
    c1_o = c1(p0_o)
    c1_o = Dropout(0.5)(c1_o)
    p1_o = p1(c1_o)
    c2_o = c2(p1_o)
    c2_o = Dropout(0.5)(c2_o)
    p2_o = p2(c2_o)
    c3_o = c3(p2_o)
    c3_o = Dropout(0.5)(c3_o)
    #body_convolution = c4(c3_o)
    c4_o = c4(c3_o)
    body_convolution = Dropout(0.5)(c4_o)
    
    
    #cocatenating headline and body tensors
    
    convs = [head_convolution,body_convolution]
    feature_vector = concatenate(convs,axis=1)#(axis = -1)(convs)
    
    #applying fully connected hidden layers to tensor
    
    ann_x = Dense(256,activation='relu')(feature_vector)
    ann_x = Flatten()(ann_x)  
    ann_x = Dense(256,activation='relu')(ann_x)
                          #flattening the input
    ann_x = Dense(256,activation='relu')(ann_x)
    
    #final output layer
    
    pred = Dense(2,activation='softmax')(ann_x)
    
    model = Model(inputs = [headline,body], outputs = pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model




'''

dataset = pd.read_csv('data.csv')

dataset.isnull().sum()

dataset = dataset[pd.notnull(dataset['Body'])]


X_data = dataset.iloc[:,1:3]
Y_data = dataset.iloc[:,3]


from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_data, Y_data, test_size = 0.25, random_state = 0)
'''



def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, data_token, generate_missing=False):
    embeddings = data_token.apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)






def preprocess(texts, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    train_word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(train_word_index))
    
    data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
    
    print('Shape of data tensor:', data.shape)
    
    return data, train_word_index




if __name__ == "__main__":

    
    dataset = pd.read_csv('gdbt_training_input.csv', encoding = 'cp1252')
    dataset2 = pd.read_csv('gdbt_testing_input.csv', encoding = 'cp1252')
    dataset3 = pd.read_csv('gdbt_testing_ouput.csv', encoding = 'cp1252')
    
    head_train = dataset['Headline']
    body_train = dataset['articleBody']
    
    head_test = dataset2['Headline']
    body_test = dataset2['articleBody']
    
    Y_train = dataset['Stance']
    Y_test = dataset3['Stance']
    
    
    
    
    
    
    
    
    
    
    
    #loading Google pre-trained Word2Vec model
    '''
    from generate_vector import *
    gv = GoogleVec()
    gv.load()
    '''
    
    #loading news for training
    
    
    
    
    
    
    
    '''
    
    from secondClass import *
    news = News(stances = trainig_data_file, bodies = 'train_bodies.csv', vecs = gv)
    
    news1 = News(stances = testing_data_file, bodies = 'train_bodies.csv', vecs = gv)		
    
    #getting training data
    head_train,body_train,stance_train = news.sample(n=news.n_headlines)
    
    #getting testing data
    head_test,body_test,stance_test = news1.sample(n=news1.n_headlines)
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    
    
    #creting train headline tokens
    head_train = pd.DataFrame(head_train)
    head_tokens = head_train['Headline'].apply(tokenizer.tokenize)
    
    #creting test headline tokens
    head_test = pd.DataFrame(head_test)
    head_test_tokens = head_test['Headline'].apply(tokenizer.tokenize)
    
    
    
    #creting train body tokens
    body_train = pd.DataFrame(body_train)
    body_tokens = body_train['articleBody'].apply(tokenizer.tokenize)
    
    
    
    #creting test body tokens
    body_test = pd.DataFrame(body_test)
    body_test_tokens = body_test['articleBody'].apply(tokenizer.tokenize)
    
    
    
    #data = preprocess(head_train, MAX_NB_WORDS1, MAX_HEAD_SEQUENCE_LENGTH)
    
    #training headline token length
    
    all_training_headline_words = [word for tokens in head_tokens for word in tokens]
    training_headline_lengths = [len(tokens) for tokens in head_tokens]
    TRAINING_HEADLINE_VOCAB = sorted(list(set(all_training_headline_words)))
    print(len(all_training_headline_words), " words total, with a vocabulary size of ",len(TRAINING_HEADLINE_VOCAB))
    print("Max sentence length is " , max(training_headline_lengths))
    
    
    #testing headlines token length
    all_testing_headline_words = [word for tokens in head_test_tokens for word in tokens]
    testing_headline_lengths = [len(tokens) for tokens in head_test_tokens]
    TESTING_HEADLINE_VOCAB = sorted(list(set(all_testing_headline_words)))
    print(len(all_testing_headline_words), " test words total, with a vocabulary size of ",len(TESTING_HEADLINE_VOCAB))
    print("Max test words length is " , max(testing_headline_lengths))
    
    
    #training bodies token length
    all_training_bodies_words = [word for tokens in body_tokens for word in tokens]
    training_bodies_lengths = [len(tokens) for tokens in body_tokens]
    TRAINING_BODY_VOCAB = sorted(list(set(all_training_bodies_words)))
    print(len(all_training_bodies_words), " tarining words total, with a vocabulary size of ",len(TRAINING_BODY_VOCAB))
    print("Max sentence length is " , max(training_bodies_lengths))
    
    #testing bodies token length
    all_testing_bodies_words = [word for tokens in body_test_tokens for word in tokens]
    testing_bodies_lengths = [len(tokens) for tokens in body_test_tokens]
    TESTING_BODY_VOCAB = sorted(list(set(all_testing_bodies_words)))
    print(len(all_testing_bodies_words), " test words total, with a vocabulary size of ",len(TESTING_BODY_VOCAB))
    print("Max test words length is " , max(testing_bodies_lengths))
    
    
    
    #word2vec embedding function
    word2vec_path = "GoogleNews-vectors-negative300.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    
    
    
    #training_embeddings = get_word2vec_embeddings(word2vec, head_tokens, generate_missing=True)
    
    
    #get embeddings
    head_train_data, train_word_index = preprocess(head_train['Headline'],MAX_NB_WORDS1,MAX_HEAD_SEQUENCE_LENGTH)
    
    train_head_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
    for word,index in train_word_index.items():
        train_head_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
    print(train_head_embedding_weights.shape)
    
    
    #get Embeddings for bodies
    body_train_data, body_train_word_index = preprocess(body_train['articleBody'],MAX_NB_WORDS1,MAX_BODY_SEQUENCE_LENGTH)
    
    train_body_embedding_weights = np.zeros((len(body_train_word_index)+1, EMBEDDING_DIM))
    for word,index in body_train_word_index.items():
        train_body_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
    print(train_body_embedding_weights.shape)
    
    #model = build_model(train_head_embedding_weights, train_body_embedding_weights,MAX_HEAD_SEQUENCE_LENGTH, MAX_BODY_SEQUENCE_LENGTH,
    #                    len(train_word_index)+1, len(body_train_word_index)+1, EMBEDDING_DIM, False)
    
    #model.summary()
    
    
    stance_train = np.array(Y_train)
    
    from sklearn.preprocessing import OneHotEncoder
    stance_train = stance_train.reshape(2991,-1)
    oneHot = OneHotEncoder(categorical_features=[0])
    stance_train = oneHot.fit_transform(stance_train).toarray()
    
    
    #hist = model.fit([head_train_data, body_train_data], stance_train, epochs = 2)
    #hist = model.fit([head_train_data, body_train_data], stance_train, epochs = 2, validation_split=0.1)
    
    
    #to load and use pretrained saved model
    
    #to save model into harddisk
    '''
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    '''
    
    #To used the saved model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    
    #model loading ends
    
    
    
    
    '''
    import pydot
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    

    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(hist.history['acc'],  color='b', label='train')
    plt.plot(hist.history['val_acc'],  color='r', label='val')
    plt.title('CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    
    
    
    
    
    '''
    
    #test_head_data, a = preprocess(head_test[0],MAX_NB_WORDS1,MAX_HEAD_SEQUENCE_LENGTH)
    #test_body_data,b = preprocess(body_test[0],MAX_NB_WORDS1,MAX_BODY_SEQUENCE_LENGTH)
    
    tokenizer1 = Tokenizer(num_words = MAX_NB_WORDS1)
    tokenizer1.fit_on_texts(head_train['Headline'])
    
    tokenizer2 = Tokenizer(num_words = MAX_NB_WORDS1)
    tokenizer2.fit_on_texts(body_train['articleBody'])
    
    test_head_data = tokenizer1.texts_to_sequences(head_test['Headline'])
    test_head_data = pad_sequences(test_head_data, maxlen=MAX_HEAD_SEQUENCE_LENGTH)
    
    
    
    #sample running
    
    
    
    sample_head = "Spider burrowed through tourist's stomach and up into his chest"
    sample_body = "Fear not arachnophobes, the story of Bunbury's spiderman might not be all it seemed.\
Perth scientists have cast doubt over claims that a spider burrowed into a man's body during his first trip to Bali. The story went global on Thursday, generating hundreds of stories online.\
Earlier this month, Dylan Thomas headed to the holiday island and sought medical help after experiencing ""a really burning sensation like a searing feeling"" in his abdomen.\
Dylan Thomas says he had a spider crawl underneath his skin.\
Thomas said a specialist dermatologist was called in and later used tweezers to remove what was believed to be a ""tropical spider"".\
But it seems we may have all been caught in a web... of misinformation.\
Arachnologist Dr Volker Framenau said whatever the creature was, it was ""almost impossible"" for the culprit to have been a spider.\
""If you look at a spider, the fangs, the mouth parts they have, they are not able to burrow. They can't get through skin,"" he said.\
""We thought it may have been something like a mite, there are a few different parasitic mites out there, which can sometimes look a bit like a spider. I can't think of any spider which could do this to a person.""\
Dr Mark Harvey from the Western Australian Museum agreed and said he found the case ""bizarre"".\
""I must confess I was amazed because I've never heard of a spider being able to survive under the skin of a human, or indeed any mammal,"" he said.\
""Spiders need air to breathe, they have spiracles on the sides of their bodies where air comes into their system through a series of what we call book lungs. Being under the skin of somebody, I would have thought they wouldn't have enough air to survive.\
""Even if it was a mite, I've never seen anything like this. Even if it was an insect, I've never heard of an insect crawling under the skin like this, so it really is a remarkable case.""\
Dr Harvey said spiders were widely feared in the community and often were the subject of urban legends.\
""We hear about people going on holidays and having spiders lay eggs under the skin. Then [the baby spiders] burst out when they return from their holiday in the tropics,"" he said.\
""None of those are true, they're just made up stories.\
""They're not actually able to dig through the skin, that's why this case is so unusual. Some can burrow into soil, but they have to remove soil particles one at a time if they want to do that.""\
Something which is true, according to Dr Harvey, is that certain arachnids do ""live on humans"".\
""We all have mites living on our faces. They're follicle mites, but they're absolutely miniscule and you can't see them. We transmit them to our children when we have kids,"" he said.\
""They live in the bases of hair follicles on our faces and in some of the pores in our skin. Those mites are so small, you can't see them, and they're not going to cause a blemish on the skin like this lad has on his stomach.""\
Dr Framenau said that much of the confusion could be eliminated by keeping or catching the creepy crawly offender, dead or alive, and enlisting the help of experts.\
""It would be great if they collected it or took a photo of it,"" he said.\
""If you have been bitten by something, the best thing you can do is collect it and submit it to a museum for identification before these things go viral.""\
Dylan Thomas has been contacted for comment.\
- WA Today"


    sample_head = "'Nasa Confirms Earth Will Experience 6 Days of Total Darkness in December' Fake News Story Goes Viral"
    sample_body = "Thousands of people have been duped by a fake news story claiming that Nasa has forecast a total blackout of earth for six days in December.\
The story, entitled ""Nasa Confirms Earth Will Experience 6 Days of Total Darkness in December 2014!"" originated from Huzlers.com, a website well known for publishing fake stories with sensational headlines.\
The bogus report read: ""Nasa has confirmed that the Earth will experience 6 days of almost complete darkness and will happen from the dates Tuesday the 16 â€“ Monday the 22 in December. The world will remain, during these three days, without sunlight due to a solar storm, which will cause dust and space debris to become plentiful and thus, block 90% sunlight.\
""The head of Nasa Charles Bolden who made the announcement and asked everyone to remain calm. This will be the product of a solar storm, the largest in the last 250 years for a period of 216 hours total.\
""Despite the six days of darkness soon to come, officials say that the earth will not experience any major problems, since six days of darkness is nowhere near enough to cause major damage to anything.""\
Adding on, the article also carried a made-up quote from Nasa scientist Earl Godoy, saying: ""We will solely rely on artificial light for the six days, which is not a problem at all.""\
Many Twitter users believed the fake news report, and expressed their shock.\
We're going to have a complete 6 days of darkness due to a solar storm in Dec! SO NERVOUS ABOUT THIS! Ahhh. #ThePurge http://t.co/0L2Sis54hvâ€” Janella (@hijanellamarie) October 26, 2014 6 days of total darkness in December? ? http://t.co/eTN60TnXftâ€” Jammie Macaranas (@JammiePeach) October 26, 2014 ""NASA Confirms Earth will experience 6 Days of total DARKNESS in December 2014."" Me: pic.twitter.com/xZG1xaxqdwâ€” [Hiatus] TT (@sarangBCES) October 26, 2014 ""NASA Confirms Earth Will Experience 6 Days of Total Darkness in December 2014!"" omg what?â€” æŸ¥ç† (@Chxrliecutie) October 26, 2014 islam know what this means im scared ""NASA Confirms Earth Will Experience 6 Days of Total Darkness in December 2014! http://t.co/GQGeGLmElZ""â€” hiatus (@taobby) October 26, 2014\
The website has previously published a fake report about American rapper and actor Tupac Shakur, claiming that he is alive.\
RelatedHalloween 2014 on Friday the 13th for First Time in 666 Years Declared a HoaxShah Rukh Khan's Son Aryan and Aishwarya Rai Bachchan's Niece Navya's Leaked Sex Tape is FakeEbola Zombies: Victims 'Rising from the Dead' Fake News Story Goes Viral, Sparks Outrage on Social MediaEminem 'Quits Music After Checking Into Rehab Again For Heroin Addiction' is Hoax: Satirical Article Creates Stir on Social Media"

    test_head_sample  = tokenizer1.texts_to_sequences([sample_head])
    test_head_sample = pad_sequences(test_head_sample, maxlen=MAX_HEAD_SEQUENCE_LENGTH)
    
    
    
    
    
       
    test_body_sample = tokenizer2.texts_to_sequences([sample_body])
    test_body_sample = pad_sequences(test_body_sample, maxlen=MAX_BODY_SEQUENCE_LENGTH)
    
    y_sample = model.predict([test_head_sample,test_body_sample])
    y_pred_sample = np.argmax(y_sample,axis = 1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #sample ends
    
    
    test_body_data = tokenizer2.texts_to_sequences(body_test['articleBody'])
    test_body_data = pad_sequences(test_body_data, maxlen=MAX_BODY_SEQUENCE_LENGTH)
    
    
    
    
    
    
    
    
    
    
    #print(len(a)+1,'   ',len(b)+1,'   ',len(train_word_index)+1,'   ',len(body_train_word_index)+1)
    
    y = model.predict([test_head_data,test_body_data])
    
    y_pred = np.argmax(y,axis = 1)
    
    
    #np.array(Y_test).count(0)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.array(Y_test),y_pred)
    print(cm)
    
    
    
    
    #to save predictions
    '''
    predicted = [LABELS[int(a)] for a in y_pred]
    
    df_output = pd.DataFrame()
    df_output['Headline'] = dataset2['Headline']
    df_output['Body'] = dataset2['articleBody']
    df_output['Actual Stance'] = Y_test
    df_output['Stance'] = predicted
    df_output['prob_0'] = y[:, 0]
    df_output['prob_1'] = y[:, 1]
    #df_output['prob_2'] = pred_prob_y[:, 2]
    #df_output['prob_3'] = pred_prob_y[:, 3]
    #df_output.to_csv('submission.csv', index=False)
    df_output.to_csv('cnn_pred_prob_cor2.csv', index=False)
    
 '''

    
    
   ''' 
    #data combining for result
    
    tree_result = pd.read_csv('tree_pred_prob_cor2.csv', encoding = 'cp1252')
    cnn_result = pd.read_csv('cnn_pred_prob_cor2.csv', encoding = 'cp1252')
    
    prob0 = (tree_result['prob_0'] + cnn_result['prob_0'])/2
    prob1 = (tree_result['prob_1'] + cnn_result['prob_1'])/2
    
    prob0_arr = np.array(prob0)
    prob1_arr = np.array(prob1)
    
    final_output = np.column_stack((prob0, prob1))
    
    final_pred = np.argmax(final_output, axis = 1)
    
    final = pd.DataFrame(final_pred)
    final.to_csv('final_output.csv', index = False)
    '''
    
    #to print combined confusion matrix
    '''
    #confusion matrix for combined result
    cm = confusion_matrix(np.array(Y_test),final_pred)
    
    print(cm)
    
    '''
    
    
    
    '''
    
    import matplotlib.pyplot as plt
    plt.hist([0.973,0.92,0.975])
    
    
    x = np.arange(3)
    plt.hist(x,[0.9739,0.92,0.9759]) 
    plt.xticks(x,['XGBOOST','CNN','COMBINED'])
    
    
    '''
