#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:48:57 2018

@author: chintandoshi
"""


import pandas as pd
import numpy as np
import gc
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
w0 = 0
stop_words = set(stopwords.words('english'))

def faster_tfd(chunk):
    global w0
    chunk["text"] = chunk["text"].str.split()
    chunk["text"] = chunk.text.apply(lambda x: pd.Series(pd.Series.value_counts(pd.Series(x)),index=w0.index).fillna(0.0).as_matrix())
    return chunk


def string_to_biggram(string):
    string = [x for x in string.split(" ") if x not in stop_words]
    return [m + " " + n for m,n in zip(string[:-1], string[1:])]

def biggram_generator(chunk):
    global w0,np_tf_idf_denom
    chunk["text"] = chunk["text"].map(string_to_biggram)
    chunk["text"] = chunk.text.apply(lambda x: pd.Series(pd.Series.value_counts(pd.Series(x)),index=w0.index).fillna(0.0).as_matrix())
    return chunk


def biggram():
    global w0
    w0 = pd.Series.from_csv("100_biggram_init_w.csv")
    test_data = pd.read_csv("reviews_tr.csv")
    tf_w_list = []
    for i in range(0,14):
        tf_w_list.append(pd.Series.from_csv("biggram_w" + str(i) + ".csv"))

    sample1 = test_data.sample(n=15000)
    sample2 = test_data.sample(n=15000)
    sample3 = test_data.sample(n=15000)

    sample1 = biggram_generator(sample1)
    sample2 = biggram_generator(sample2)
    sample3 = biggram_generator(sample3)

    accu1 = []
    accu2 = []
    accu3 = []

    sample1["label"] = sample1["label"].replace(to_replace=0, value=-1)
    sample2["label"] = sample2["label"].replace(to_replace=0, value=-1)
    sample3["label"] = sample3["label"].replace(to_replace=0, value=-1)
    for i in tf_w_list:
        temp = sample1["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample1["label"]
        accu1.append(float(len([m for m in temp if m > 0])) / float(len(sample1)))
        temp = sample2["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample2["label"]
        accu2.append(float(len([m for m in temp if m > 0])) / float(len(sample2)))
        temp = sample3["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample3["label"]
        accu3.append(float(len([m for m in temp if m > 0])) / float(len(sample3)))
        accu = [(x + y + z)/3.0 for x, y, z in zip(accu1, accu2, accu3)]
    print "Done"
    del sample1,sample2,sample3
    gc.collect()
    return accu


def tf_idf():
    global w0
    w0 = pd.Series.from_csv("new_init_w.csv")
    test_data = pd.read_csv("reviews_tr.csv")
    tf_w_list = []
    for i in range(1,30,2):
        tf_w_list.append(pd.Series.from_csv("idf_w" + str(i) + ".csv"))
    sample1 = test_data.sample(n=15000)
    sample2 = test_data.sample(n=15000)
    sample3 = test_data.sample(n=15000)

    sample1 = faster_tfd(sample1)
    sample2 = faster_tfd(sample2)
    sample3 = faster_tfd(sample3)

    accu1 = []
    accu2 = []
    accu3 = []

    sample1["label"] = sample1["label"].replace(to_replace=0, value=-1)
    sample2["label"] = sample2["label"].replace(to_replace=0, value=-1)
    sample3["label"] = sample3["label"].replace(to_replace=0, value=-1)
    for i in tf_w_list:
        temp = sample1["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample1["label"]
        accu1.append(float(len([m for m in temp if m > 0])) / float(len(sample1)))
        temp = sample2["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample2["label"]
        accu2.append(float(len([m for m in temp if m > 0])) / float(len(sample2)))
        temp = sample3["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample3["label"]
        accu3.append(float(len([m for m in temp if m > 0])) / float(len(sample3)))
        accu = [(x + y + z)/3.0 for x, y, z in zip(accu1, accu2, accu3)]
    print "Done"
    del sample1,sample2,sample3
    gc.collect()
    return accu


def tfd_test():
    global w0
    w0 = pd.Series.from_csv("new_init_w.csv")
    test_data = pd.read_csv("reviews_tr.csv")
    tf_w_list = []
    for i in range(1, 30, 2):
        tf_w_list.append(pd.Series.from_csv("w" + str(i) + ".csv"))

    sample1 = test_data.sample(n=15000)
    sample2 = test_data.sample(n=15000)
    sample3 = test_data.sample(n=15000)
    
    sample1 = faster_tfd(sample1)
    sample2 = faster_tfd(sample2)
    sample3 = faster_tfd(sample3)

    accu1 = []
    accu2 = []
    accu3 = []

    sample1["label"] = sample1["label"].replace(to_replace=0, value=-1)
    sample2["label"] = sample2["label"].replace(to_replace=0, value=-1)
    sample3["label"] = sample3["label"].replace(to_replace=0, value=-1)
    
    for i in tf_w_list:
        temp = sample1["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample1["label"]
        accu1.append(float(len([m for m in temp if m > 0])) / float(len(sample1)))
        temp = sample2["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample2["label"]
        accu2.append(float(len([m for m in temp if m > 0])) / float(len(sample2)))
        temp = sample3["text"].apply(lambda x: np.dot(x, i))
        temp = temp * sample3["label"]
        accu3.append(float(len([m for m in temp if m > 0])) / float(len(sample3)))
        accu = [(x + y + z)/3.0 for x, y, z in zip(accu1, accu2, accu3)]
    print "Done"
    del sample1,sample2,sample3
    gc.collect()
    return accu


def main():

    bigg = biggram()
    tfd = tfd_test()
    tfidf = tf_idf()

    plt.title("Test Accuracy -> Datasize")
    plt.plot(range(100000,800000,50000), bigg, label="Biggram",marker='o')
    plt.plot(range(100000,800000,50000), tfd, label="tf-idf",marker='o')
    plt.plot(range(100000,800000,50000), tfidf, label="Onegram",marker='o')
    plt.xlabel("Test data size->")
    plt.ylabel("Accuracy->")
    plt.legend()
    plt.gcf().set_size_inches(15,8)
    plt.legend()
    plt.show()
    plt.savefig("graph_hw2q5.png")



if __name__ == '__main__':
    main()
