#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random

class StanfordSentiment:
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            #path = "cs224d/datasets/stanfordSentimentTreebank"
            #path = "/home/lusine/NLP/sensegram/word2vec_c/word2vec1"
            path = "/data/NLP/sensegram/word2vec_c/word2vec1/aclImdb/train"

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
        for dir_name in ["/neg", "/pos", "/unsup"]:
            for filename in os.listdir(self.path + dir_name):         
                with open(self.path + dir_name + "/" + filename, "r") as f:
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
            print len(tokens)
            print wordcount
            pickle.dump(self._tokens, TableFile)
        with open(self.path+'/tokensFreqTable', 'w') as TableFile:
            pickle.dump(self._tokenfreq, TableFile)
        with open(self.path+'/wordCountTable', 'w') as TableFile:
            pickle.dump(self._wordcount, TableFile)
        with open(self.path+'/revtokensTable', 'w') as TableFile:
            pickle.dump(self._revtokens, TableFile)
        exit()
        return self._tokens
    
    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences
        sentences = []
        #with open(self.path + "/datasetSentences.txt", "r") as f:
        with open(self.path + "/data.txt", "r") as f:
        #with open(self.path + "/data.txt", "r") as f:
            #first = True
            for line in f:
                '''if first:
                    first = False
                    continue'''
                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                #sentences += [[w.lower().decode("utf-8").encode('latin1') for w in splitted]]
                sentences += [[w for w in splitted]]
            
                
        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            with open(self.path + "/data.txt", "r") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                self._numSentences = line_num
                print line_num
                return self._numSentences
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[w for w in s 
            if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
            for s in sentences * 30]#TODO *30??
            #minimal log-likelihood value that a token requires to be considered as a frequent sentence starter TODO THIS?

        allsentences = [s for s in allsentences if len(s) > 1]
        
        self._allsentences = allsentences
        
        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID] 
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    def not_reject(self, w, rejectProb, tokens):
        return 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]

    def getContext(self, C=5):
        
        #
        #with open(self.path + "/file", "r") as f: 
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        paragraph_id = -1
        for dir_name in ["/neg", "/pos", "/unsup"]:
            for filename in os.listdir(self.path + dir_name):         
                with open(self.path + dir_name + "/" + filename, "r") as f: 
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
                            i += 1#yield i? #TODO
                            yield paragraph_id, context

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in xrange(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            sent_labels[i] = labels[dictionary[full_sent]]
            
        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in xrange(3)]
        with open(self.path + "/datasetSplit.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable
        with open(self.path+'/sampleTable', "r") as TableFile:
            self._sampleTable = pickle.load(TableFile)
        return self._sampleTable
        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))#TODO
        #self.allSentences() TODO ?????
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
            if (i % 100000 == 0):
                print i, j
        with open(self.path+'/sampleTable', 'w') as TableFile:
            pickle.dump(self._sampleTable, TableFile)
        return self._sampleTable

    def rejectProb(self):#TODO
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

