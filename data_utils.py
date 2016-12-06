#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None, tablesize = 1000000):#TODO why 1000000?
        if not path:
            path = "./paragraph2vec_l"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens
        with open(self.path+ "/tokensTable", "r") as TableFile:
            self._tokens = pickle.load(TableFile)
        with open(self.path+'/tokensFreqTable', 'r') as TableFile:
            self._tokenfreq = pickle.load(TableFile)
        with open(self.path+'/wordCountTable', 'r') as TableFile:
            self._wordcount = pickle.load(TableFile)
        with open(self.path+'/revtokensTable', 'r') as TableFile:
            self._revtokens = pickle.load(TableFile)
        return self._tokens
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0        
        with open(self.path + "/data_p2v.txt", "r") as f:
            for line in f:
                splitted = line.strip().split()[1:]
                for w in splitted:
                    wordcount += 1
                    if not w in tokens:
                        tokens[w] = idx
                        revtokens += [w]
                        tokenfreq[w] = 1
                        idx += 1
                    else:
                        tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        with open(self.path+'/tokensTable', 'w') as TableFile:
            pickle.dump(self._tokens, TableFile)
        with open(self.path+'/tokensFreqTable', 'w') as TableFile:
            pickle.dump(self._tokenfreq, TableFile)
        with open(self.path+'/wordCountTable', 'w') as TableFile:
            pickle.dump(self._wordcount, TableFile)
        with open(self.path+'/revtokensTable', 'w') as TableFile:
            pickle.dump(self._revtokens, TableFile)
        return self._tokens

    def not_reject(self, w, rejectProb, tokens):
        return 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]

    def getContext(self, C=5):
        
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        paragraph_id = -1
        with open(self.path + "/data_p2v.txt", "r") as f: 
            for line in f:
                paragraph_id += 1
                C1 = random.randint(1, C)
                i = 0
                splitted = line.strip().split()[1:]
                for w in splitted:  
                    context = splitted[max(0, i - C1):i] 
                    if i+1 < len(splitted):
                        context += splitted[i+1:min(len(splitted), i + C1 + 1)]
                    context = [x for x in context if (x != w and self.not_reject(x, rejectProb, tokens))]
                    if self.not_reject(w, rejectProb, tokens):
                         context += [w] 
                    i += 1
                    yield paragraph_id, context

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable
        with open(self.path+'/sampleTable', "r") as TableFile:
            self._sampleTable = pickle.load(TableFile)
        return self._sampleTable
        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        i = 0
        for w in xrange(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        
        for i in xrange(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j
        with open(self.path+'/sampleTable', 'w') as TableFile:
            pickle.dump(self._sampleTable, TableFile)
        return self._sampleTable

    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-2 * self._wordcount#TODO only for skipgram

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        for i in xrange(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]

