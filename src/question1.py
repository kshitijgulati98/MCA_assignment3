#Assignment 3
#Question 1
#word2vec

import re 
import nltk
from nltk.corpus import abc
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def training_samples(tokens,window_size,vocab_size):
    training_samples=[]
    corpus_size=len(tokens)
    for i in range(corpus_size):
        neighbours= list(range(max(0, i - window_size), i)) + list(range(i + 1, min(corpus_size, i + window_size + 1)))
        onehot_target=np.zeros(vocab_size)
        onehot_target[tokens[i]-1]=1 #dictionary index values start from 1 and not 0
        onehot_contexts=[]
        for j in neighbours:
            onehot_context=np.zeros(vocab_size)
            onehot_context[tokens[j]-1]=1
            onehot_contexts.append(onehot_context)
        training_samples.append([onehot_target,onehot_contexts])
    return np.array(training_samples)

#hidden_layer_n -> no. of neurons in hidden_layer -> no of dimensions to represent each word
#alpha -> learning rate
def word2vec(training_samples,hidden_layer_n,vocab,vocab_size,epochs,alpha):
    weight_hidden=np.random.uniform(-1,1,(vocab_size,hidden_layer_n))
    weight_output=np.random.uniform(-1,1,(hidden_layer_n,vocab_size))
    for i in tqdm(range(epochs)):
        loss=0
        filename="epoch-"+str(i)+".png"
            target=training_samples[j][0]
            context=training_samples[j][1]
            #forwardprop
            hidden_layer=np.dot(weight_hidden.T,target)
            output_layer=np.dot(weight_output.T,hidden_layer)
            softmax=softmaxfunc(output_layer)
            
            error = np.sum([(softmax-k) for k in context], axis=0)
            #backprop
            
            dl_dhidden= np.outer(target, np.dot(weight_output, error.T))
            dl_doutput = np.outer(hidden_layer, error)
            weight_hidden -= (alpha * dl_dhidden)
            weight_output -= (alpha * dl_doutput)
                    
        print("epoch:",i)
        plot_func(vocab,weight_hidden,filename)
    
    
    return weight_hidden

#https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmaxfunc(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def word_vec(word,weight):
    i = vocab[word]
    word_vec= weight[i-1]
    return word_vec

#https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
def plot_func(vocab,weight,filename):
    words_ak = []
    embeddings_ak = []
    for word in list(vocab.keys()):
        embeddings_ak.append(word_vec(word,weight))
        words_ak.append(word)
    
    tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings_ak)
    tsne_plot_2d(filename,"words of abc corpus", embeddings_ak_2d,a=0.3)

def tsne_plot_2d(filename,label, embeddings, words=[], a=1):
    plt.figure(figsize=(16, 9))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c="red", alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    corpus=list(abc.words()) #the abc corpus of nltk
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(corpus)#tokenizing the corpus
    tokenized_corpus = tokenizer.texts_to_sequences(corpus)
    tokens=[] # removing the empty tokens("")
    for i in tokenized_corpus:
        if i!=[]:
            tokens.append(i[0])
    #tokens
    vocab=tokenizer.word_index  #vocabulary with all word indexes
    vocab_size = len(vocab) #size of vocabulary
    training_samples=training_samples(tokens,2,vocab_size)#create training samples
    word2vec(training_samples,10,vocab,vocab_size,10,0.05) 
    #10 epochs with learning rate=0.05 with 10 neurons in the NN
