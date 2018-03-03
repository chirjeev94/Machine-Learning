"""
Created on Wed Feb 28 01:11:54 2018

@author: chintan94
"""

import pandas as pd
import numpy as np
import gc
from nltk.corpus import stopwords
from collections import Counter
from time import time

total_len = 0
tf_idf_denom = 0
np_tf_idf_denom = 0
w0 = 0
idf_w_0 = 0
np_w0 = 0
np_idf_w_0 = 0

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
    w0 = pd.Series(0, index=unique_words)
    w0.to_csv("init_w.csv")
    return w0


def faster_tfd(chunk):
    global w0,np_tf_idf_denom
    chunk["text"] = chunk["text"].str.split()
    chunk["text"] = chunk.text.apply(lambda x: pd.Series(pd.Series.value_counts(pd.Series(x)),index=w0.index).fillna(0.0).as_matrix())
    np_tf_idf_denom = np_tf_idf_denom + np.sum(chunk["text"])
    return chunk

def perceptron_map_tf(row):
    global np_w0, np_tf_idf_denom, w0
    dot_product = np_w0.dot(row[1])
    if (dot_product >= 0 and row[0] == 0):
        np_w0 = np_w0 - row[1]
    elif (dot_product <= 0 and row[0] == 1):
        np_w0 = np_w0 + row[1]


def perceptron_map_tf_idf(row):
    global np_idf_w_0, np_idf_w_0, np_tf_idf_denom, w0
    dot_product = np_idf_w_0.dot(row[1])
    if (dot_product >= 0 and row[0] == 0):
        np_idf_w_0 = np_idf_w_0 - row[1]
    elif (dot_product <= 0 and row[0] == 1):
        np_idf_w_0 = np_idf_w_0 + row[1]


def main():
    global tf_idf_denom, total_len, w0, idf_w_0, np_idf_w_0, np_tf_idf_denom, np_w0
    # Storing all the unique words in a files
    # Toggle it to true if it's first run
    uw_comp_necessary = False

    if uw_comp_necessary:
        # no need to load the entire file
        df = pd.read_csv("reviews_tr.csv")
        string_list = df["text"]
        w0 = unique_word_set(string_list)
        del df
        gc.collect()
    else:
        w0 = pd.Series.from_csv("init_w.csv")

    tf_idf_denom = pd.Series(0.0, index=w0.index)
    idf_w_0 = pd.Series(0.0, index=w0.index)
    np_w0 = w0.as_matrix()
    np_idf_w_0 = idf_w_0.as_matrix()
    np_tf_idf_denom = tf_idf_denom.as_matrix()

    w_list = []
    idf_w_list = []
    chunksize = 10 ** 5
    control = False
    write = True
    tf_idf_flag = False

    # Processing the file in chunck
    start = time()
    for chunk in pd.read_csv("reviews_tr.csv", chunksize=chunksize):

        total_len += len(chunk)
        # TFD
        chunk = faster_tfd(chunk)
        print "TFD Computed " + str(time() - start)
        chunk.apply(perceptron_map_tf, axis=1)
        print "Shuffling " + str(time() - start)
        chunk = chunk.iloc[np.random.permutation(len(chunk))]
        chunk.apply(perceptron_map_tf, axis=1)

        # TF_IDF
        print "TF_IDF Start " + str(time() - start)
        chunk.text.apply(lambda x:(total_len/2) * np.divide(x, np_tf_idf_denom, out=np.zeros_like(x), where=np_tf_idf_denom != 0))
        chunk.apply(perceptron_map_tf_idf, axis=1)
        print "TF_IDF_Shuffling " + str(time() - start)
        chunk = chunk.iloc[np.random.permutation(len(chunk))]
        chunk.apply(perceptron_map_tf_idf, axis=1)
        print "TF_IDF Part Over " + str(time() - start)

        if (total_len % 10 ** 5) == 0:
            w_list.append(np_w0 / (total_len * 2 + 1))
            idf_w_list.append(np_idf_w_0 / (1 + total_len * 2))
            print "Processed " + str(total_len) + " in " + str(time() - start)
            if control:
                print "Do you wish to continue?"
                ans = raw_input("Yes/No?")
                if ans == 'No':
                    break

    # Documenting all the w for testing
    # To get it back use this
    # w = pd.Series.from_csv("w0.csv")
    j = 0
    if (write):
        for i in w_list:
            temp = pd.Series(i,index=w0.index)
            temp.to_csv("w" + str(j) + ".csv")
            j = j + 1
        j = 0
        for i in idf_w_list:
            temp = pd.Series(i, index=w0.index)
            temp.to_csv("idf_w" + str(j) + ".csv")
            j = j + 1
    print('\a')
    # Testing accuracy part to follow

if __name__ == '__main__':
    main()
