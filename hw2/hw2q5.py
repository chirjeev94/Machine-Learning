#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 01:11:54 2018

@author: chintan94
"""

import pandas as pd
import numpy as np
import gc
from nltk.corpus import stopwords
from collections import Counter

def unique_word_set(string_list):
    set_list = []
    for string in string_list:
        set_list.append(set(string.split(" ")))
    unique_words = set.union(*set_list)
    
    # If the line below fails, download stopwords
    # import nltk
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    unique_words = unique_words - stop_words
    w0 = pd.Series(0,index = unique_words)
    w0.to_csv("init_w.csv")
    return w0

def tfd(string):
     wc = Counter(string.split(" "))
     x = pd.Series(wc.values(),index = wc.keys())
     return x 

def update_w(old_w,string,label):
    string_vec = tfd(string)
    
    #This  will ignore all the new words in string_vec
    string_vec = pd.Series(string_vec,index=old_w.index).fillna(0.0)
    
    dot_product = old_w.dot(string_vec)
    
    if(dot_product >= 0 and label == 0):
        return (old_w - string_vec)
    elif (dot_product <= 0 and label == 1):
        return (old_w + string_vec)
    return old_w
   
def main():
    
    #Storing all the unique words in a files
    #Toggle it to true if it's first run 
    uw_comp_necessary = False
    
    if uw_comp_necessary:
        #no need to load the entire file
        df = pd.read_csv("reviews_tr.csv")
        string_list = df["text"]
        w0 = unique_word_set(string_list)
        del df
        gc.collect()
    else:
        w0 = pd.Series.from_csv("init_w.csv")    
    
    w_final = pd.Series(0.0,index = w0.index)

        
    #Storing all the ws
    w_list = []
    chunksize = 10 ** 5
    total_len = 0
    control = True
    write = True
        
    #Processing the file in chunck
    for chunk in pd.read_csv("reviews_tr.csv", chunksize=chunksize):        
        for index, row in chunk.iterrows():
            w0 = update_w(w0,row[1],row[0])
            w_final = w_final + w0
        #Shuffle and re-do
        chunk = chunk.iloc[np.random.permutation(len(chunk))]
        for index, row in chunk.iterrows():
            w0 = update_w(w0,row[1],row[0])
            w_final = w_final + w0    
        total_len += 2 * len(chunk)
        w_list.append(w_final/total_len)
        print "Processed " + str(total_len/2)
        
        if(control):
            print "Do you wish to continue?"
            ans = raw_input("Yes/No?")
            if ans == "No":
                break
            
    #Documenting all the w for testing
    j = 0
    if(write):
        for i in w_list:
            i.to_csv("w"+ str(j) +".csv")
            j = j + 1
    
    #Testing accuracy part to follow

if __name__ == '__main__':
    main()