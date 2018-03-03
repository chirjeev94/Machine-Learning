"""
Created on Wed Feb 28 01:11:54 2018

@author: chintan94
"""

import pandas as pd
import numpy as np
import gc
from nltk.corpus import stopwords
from time import time

total_len = 0
w0 = 0
np_w0 = 0
stop_words = set(stopwords.words('english'))

def unique_word_set(string_list):
    
    string_list = [[x for x in m.split(" ") if x not in stop_words] for m in string_list]
    set_list = [m + " " + n for l in string_list for m,n in zip(l[:-1], l[1:])]

    #Word_count used in preprocessing
    #wc = pd.Series(set_list).value_counts()
    wc = pd.Series.from_csv("biggram_word_count.csv")
    to_remove = set(wc[wc < 10].index)
    unique_words = set(set_list)
    unique_words = unique_words - to_remove
    w0 = pd.Series(0, index=unique_words)
    w0.to_csv("100_biggram_init_w.csv")
    return w0

def string_to_biggram(string):
    string = [x for x in string.split(" ") if x not in stop_words]
    return [m + " " + n for m,n in zip(string[:-1], string[1:])]

def biggram_generator(chunk):
    global w0,np_tf_idf_denom
    chunk["text"] = chunk["text"].map(string_to_biggram)
    chunk["text"] = chunk.text.apply(lambda x: pd.Series(pd.Series.value_counts(pd.Series(x)),index=w0.index).fillna(0.0).as_matrix())
    return chunk

def perceptron_map_tf(row):
    global np_w0, np_tf_idf_denom, w0
    dot_product = np_w0.dot(row[1])
    if (dot_product >= 0 and row[0] == 0):
        np_w0 = np_w0 - row[1]
    elif (dot_product <= 0 and row[0] == 1):
        np_w0 = np_w0 + row[1]

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
        w0 = pd.Series.from_csv("100_biggram_init_w.csv")

    np_w0 = w0.as_matrix()
    chunksize = 10 ** 4
    control = False

    # Processing the file in chunck
    start = time()
    j=0
    i=0
    for chunk in pd.read_csv("reviews_tr.csv", chunksize=chunksize):

        total_len += len(chunk)
        chunk = biggram_generator(chunk)
        print "Biggram TFD Computed " + str(time() - start)
        chunk.apply(perceptron_map_tf, axis=1)
        print "Shuffling " + str(time() - start)
        chunk = chunk.iloc[np.random.permutation(len(chunk))]
        chunk.apply(perceptron_map_tf, axis=1)

        i = i + 1
        if (i % 5) == 0:
            print "Writing time!!! i = " + str(i)
            temp = np_w0 / (total_len * 2 + 1)
            temp = pd.Series(temp, index=w0.index)
            temp.to_csv("biggram_w" + str(j) + ".csv")
            j = j + 1
            print "Processed " + str(total_len) + " in " + str(time() - start)
            if control:
                print "Do you wish to continue?"
                ans = raw_input("Yes/No?")
                if ans == 'No':
                    break
        del chunk
        gc.collect()
    # Documenting all the w for testing
    # To get it back use this
    # w = pd.Series.from_csv("w0.csv")
    # Testing accuracy part to follow

if __name__ == '__main__':
    main()
