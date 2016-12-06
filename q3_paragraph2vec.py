import numpy as np
import random
import nltk

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
from nltk.tokenize import PunktSentenceTokenizer


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, prev_idx,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    K = 25
    neg_idx = [dataset.sampleTokenIdx() for k in range(K)]
    negative_set = set(neg_idx)
    negative_set.discard(target)
    negative_set = negative_set.difference(prev_idx)
    negative_idx = list(negative_set)
    
    #s = outputVectors.shape[1]
    if not(negative_idx):
        #Z = np.zeros((s, 1))
        return None
        return 0, 0, Z, Z.T, [], Z.T, Z.T, []
    prev_idx += negative_idx
    target_vector = outputVectors[target]
    U = outputVectors[negative_idx]
    negative_idx.append(target)

    negative_idx.remove(target)#TODO

    prod_t = np.dot(target_vector, predicted)
    sig = sigmoid(prod_t)
    g = sig - 1
    sig_n = sigmoid(np.dot(U, predicted))

    h =  sig_n * U.T
    #n =  -np.log(1 - sig_n)

    #cost = list(n)
    #-np.log(sig)

    grad =  sig_n[:, np.newaxis] * predicted
    
    return [], 0, h, g * target_vector.T, negative_idx, grad, g * predicted, [target]


def skipgram(p_id, C, contextWords, tokens, inputVectors, outputVectors, prev_idx,
    dataset, word2vecCostAndGradient = negSamplingCostAndGradient):
    """ PV-DBOW model in paragraph2vec """

    #t = tokens[currentWord]
    predicted = inputVectors[p_id]
    #contextWords_set = set(contextWords)
    #contextWords_set.discard(currentWord)
    #contextWords = list(contextWords_set)

    p_list = map((lambda x: word2vecCostAndGradient(predicted, tokens[x], outputVectors, dataset, prev_idx)), contextWords)
    p_list = [p for p in p_list if p != None]

    k = 0
    #cost_stack = np.array([])
    #cost_target = 0
    gradIn_stack = np.array([])
    arr_neg_idx = np.array([])
    gradOut_stack = np.array([])
    gradIn_target_stack = np.array([])
    gradOut_target_stack = np.array([])
    target_indices = []
    for x in zip(*p_list):

        '''if (k == 0):
            list_x = []
            for w in x:
                if (w):
                    list_x += w
            cost_stack = np.asarray(list_x)
        if (k == 1):
            cost_target = sum(x) '''
        if (k == 2):
            gradIn_stack = np.concatenate([j for j in x if np.sum(j)], axis = 1)
        if (k == 3):
            gradIn_target_stack = np.vstack([j for j in x if np.sum(j)])
        if (k == 4):
            list_x = []
            for w in x:
                if (w):
                    list_x += w
            arr_neg_idx = np.asarray(list_x)
        if (k == 5):
            gradOut_stack = np.vstack([j for j in x if np.sum(j)])
        if (k == 6):
            gradOut_target_stack = np.vstack([j for j in x if np.sum(j)])
        if (k == 7):
            for w in x:
                if (w):
                    target_indices += w
            break
        k += 1
    
    #return cost_stack, cost_target, target_indices, gradIn_stack, gradIn_target_stack, arr_neg_idx, gradOut_stack, gradOut_target_stack
    return target_indices, gradIn_stack, gradIn_target_stack, arr_neg_idx, gradOut_stack, gradOut_target_stack

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, it, nParagraphs, word2vecCostAndGradient = negSamplingCostAndGradient):   
    prev_idx = []
    batchsize = 50
    #cost = 0.0
    inputVectors = wordVectors[:nParagraphs,:]
    outputVectors = wordVectors[nParagraphs:,:]
    #grad = np.zeros(wordVectors.shape)
    count = 0
    first = True
    I_in = np.array([])
    GradIn = np.array([])
    I_out = np.array([])
    GradOut = np.array([])
    GradOut_target = np.array([])
    T = []
    for i in it:
        C1 = random.randint(1, C)
        
        denom = 1
        p_id, context = i
        #t = tokens[centerword]
        idx_in, gin, gin_target, idx_out, gout, gout_target = word2vecModel(p_id, C1, context, tokens, inputVectors, outputVectors, prev_idx, dataset, word2vecCostAndGradient)
        #c, c_target, idx_in, gin, gin_target, idx_out, gout, gout_target = word2vecModel(p_id, C1, context, tokens, inputVectors, outputVectors, prev_idx, dataset, word2vecCostAndGradient)
        count += 1
        if (count < batchsize):
            if ((len(idx_in)) and (len(idx_out))):
                #cost += (sum(c) + c_target) / batchsize / denom
                if (first):
                    first = False
                    I_in = idx_in
                    I_out = idx_out
                    GradIn = (np.sum(gin, axis=1) + np.sum(gin_target, axis = 0)).reshape(-1, 1) / batchsize
                    GradOut = gout
                    GradOut_target = gout_target
                    T = [p_id]
                else:
                    I_in = np.concatenate([I_in, idx_in])
                    I_out = np.concatenate([I_out, idx_out])
                    
                    try:
                        GradIn = np.concatenate([GradIn, (np.sum(gin, axis=1) + np.sum(gin_target, axis = 0)).reshape(-1, 1) / batchsize], axis = 1)
                    except:
                        print idx_in
                        print idx_out
                        print centerword
                        print gin.shape
                        print gin_target.shape
                        print GradIn.shape
                        exit()
                    GradOut = np.concatenate([GradOut, gout])
                    GradOut_target = np.concatenate([GradOut_target, gout_target])
                    T += [p_id]
            else:
                continue
        else:
            return I_in, GradIn, I_out, GradOut, GradOut_target, T, False
            return cost, I_in, GradIn, I_out, GradOut, GradOut_target, T, False   
            #return cost, grad
    return I_in, GradIn, I_out, GradOut, GradOut_target, T, True
    return cost, I_in, GradIn, I_out, GradOut, GradOut_target, T, True
    #return cost, grad

if __name__ == "__main__":

    test_word2vec()
