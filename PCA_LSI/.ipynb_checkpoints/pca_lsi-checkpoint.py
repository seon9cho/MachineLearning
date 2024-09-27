# pca_lsi.py
"""Volume 3: PCA/LSI. Spec File.
<Name>
<Class>
<Date>
"""

import os
import string
import numpy as np
from math import log
from scipy import sparse
from sklearn import datasets
from scipy import linalg as la
from collections import Counter
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla


class PCA:
    def __init__(self, X, y=None, s=None, sparse=False, center=True):
        self.X = X - X.mean(axis=0) if center else X
        self.y = y
        if s == None:
            n, d = X.shape
            s = d
        if not sparse:
            U, sig, Vh = la.svd(self.X, full_matrices=False)
            self.sig = sig[:s]**2
            self.Vh = Vh[:s]
        else:
            _, sig, Vh = spla.svds(self.X, k=s, return_singular_vectors="vh")
            self.sig = sig**2
            self.Vh = Vh
        self.a = self.transform(self.X)
        self.proj_X = self.project(self.X)
    
    def transform(self, x):
        return self.Vh@x.T
    
    def project(self, x):
        return self.Vh.T@self.a

def prob1():
    """Recreates the plot shown 
    in Figure 1.4 by performing PCA on the iris dataset, 
    keeping the first two principal components."""
    
    iris = datasets.load_iris()
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    pca2_iris = PCA(X, y=y, s=2)
    A = np.vstack([pca2_iris.a, pca2_iris.y])
    setosa = A.T[A[2] == 0][:, :2]
    versicolor = A.T[A[2] == 1][:, :2]
    virginica = A.T[A[2] == 2][:, :2]

    plt.scatter(setosa[:, 0], setosa[:, 1], label="setosa", alpha=0.5)
    plt.scatter(versicolor[:, 0], versicolor[:, 1], label="versicolor", alpha=0.5)
    plt.scatter(virginica[:, 0], virginica[:, 1], label="virginica", alpha=0.5)
    plt.legend()
    plt.show()

     
def prob2(speech="./Addresses/1984-Reagan.txt"):
    """
    Uses LSI, applied to the word count matrix X, with the first 7 principal
    components to find the most similar and least similar speeches 
    
    Parameters:
        speech str: Path to speech eg: "./Addresses/1984-Reagan.txt"
    
    Returns:
        tuple of str: (Most similar speech, least similar speech) 
    """
    
    # Get list of filepaths to each text file in the folder.
    folder = "./Addresses/"
    paths = [folder+p for p in os.listdir(folder) if p.endswith(".txt")]
    
    # Helper function to get list of words in a string.
    def extractWords(text):
        ignore = string.punctuation + string.digits
        cleaned = "".join([t for t in text.strip() if t not in ignore])
        return cleaned.lower().split()
    
    # Initialize vocab set, then read each file and add to the vocab set.
    vocab = set()
    for p in paths:
        with open(p, 'r') as infile:
            for line in infile:
                vocab.update(extractWords(line))
                
                
    # load stopwords
    with open("stopwords.txt", 'r') as f:
        stops = set([w.strip().lower() for w in f.readlines()])
    
    # remove stopwords from vocabulary, create ordering
    vocab = {w:i for i, w in enumerate(vocab.difference(stops))}
    
    
    counts = []      # holds the entries of X
    doc_index = []   # holds the row index of X
    word_index = []  # holds the column index of X

    # Iterate through the documents.
    for doc, p in enumerate(paths):
        with open(p, 'r') as f:
            # Create the word counter.
            ctr = Counter()
            for line in f:
                ctr.update(extractWords(line))
            # Iterate through the word counter, store counts.
            for word, count in ctr.items():
                if word in vocab:
                    word_index.append(vocab[word])
                    counts.append(count)
                    doc_index.append(doc)
    
    # Create sparse matrix holding these word counts.
    X = sparse.csr_matrix((counts, [doc_index, word_index]),
                           shape=(len(paths), len(vocab)), dtype=np.float)
    
    pca_speech = PCA(X, s=7, sparse=True, center=False)
    speech_idx = paths.index(speech)
    X_hat = pca_speech.a.T
    xj = X_hat[speech_idx]
    similarities = []
    for i,x in enumerate(X_hat):
        if i != speech_idx:
            similarity = np.dot(xj, x) / (la.norm(xj)*la.norm(x))
            similarities.append(similarity)
    del paths[speech_idx]
    most = paths[np.argmax(similarities)]
    least = paths[np.argmin(similarities)]
    most = most.split('/')[-1].split('.')[0]
    least = least.split('/')[-1].split('.')[0]
    return most, least
    
    
def prob3(speech="./Addresses/1984-Reagan.txt"):
    """
    Uses LSI, applied to the globally weighted word count matrix A, with the 
    first 7 principal components to find the most similar and least similar speeches 
    
    Parameters:
        speech str: Path to speech eg: "./Addresses/1984-Reagan.txt"
    
    Returns:
        tuple of str: (Most similar speech, least similar speech) 
    """
    
    # Get list of filepaths to each text file in the folder.
    folder = "./Addresses/"
    paths = [folder+p for p in os.listdir(folder) if p.endswith(".txt")]
    
    # Helper function to get list of words in a string.
    def extractWords(text):
        ignore = string.punctuation + string.digits
        cleaned = "".join([t for t in text.strip() if t not in ignore])
        return cleaned.lower().split()
    
    # Initialize vocab set, then read each file and add to the vocab set.
    vocab = set()
    for p in paths:
        with open(p, 'r') as infile:
            for line in infile:
                vocab.update(extractWords(line))
                
                
    # load stopwords
    with open("stopwords.txt", 'r') as f:
        stops = set([w.strip().lower() for w in f.readlines()])
    
    # remove stopwords from vocabulary, create ordering
    vocab = {w:i for i, w in enumerate(vocab.difference(stops))}
    
    t = np.zeros(len(vocab))
    counts = []
    doc_index = []
    word_index = []
    
    # get doc-term counts and global term counts
    for doc, path in enumerate(paths):
        with open(path, 'r') as f:
            # create the word counter
            ctr = Counter()
            for line in f:
                words = extractWords(line)
                ctr.update(words)
            # iterate through the word counter, store counts
            for word, count in ctr.items():
                if word in vocab:
                    word_ind = vocab[word]
                    word_index.append(word_ind)
                    counts.append(count)
                    doc_index.append(doc)
                    t[word_ind] += count
    
    # Get global weights.
    g = np.ones(len(vocab))
    logM = log(len(paths))
    for count, word in zip(counts, word_index):
        p = count/float(t[word])
        g[word] += p*log(p+1)/logM
    
    # Get globally weighted counts.
    gwcounts = []
    for count, word in zip(counts, word_index):
        gwcounts.append(g[word]*log(count+1))
    
    # Create sparse matrix holding these globally weighted word counts
    A = sparse.csr_matrix((gwcounts, [doc_index,word_index]),
                          shape=(len(paths), len(vocab)), dtype=np.float)
    
    pca_speech = PCA(A, s=7, sparse=True, center=False)
    speech_idx = paths.index(speech)
    X_hat = pca_speech.a.T
    xj = X_hat[speech_idx]
    similarities = []
    for i,x in enumerate(X_hat):
        if i != speech_idx:
            similarity = np.dot(xj, x) / (la.norm(xj)*la.norm(x))
            similarities.append(similarity)
    del paths[speech_idx]
    most = paths[np.argmax(similarities)]
    least = paths[np.argmin(similarities)]
    most = most.split('/')[-1].split('.')[0]
    least = least.split('/')[-1].split('.')[0]
    return most, least
