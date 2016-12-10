import numpy as np
import random


from sigmoid import sigmoid, sigmoid_grad


def negSamplingCostAndGradient(predicted, targets, outputVectors, dataset, prev_idx,
    K=10, w2v = True):

    K = 25

    C_size =  len(targets)
    neg_idx = dataset.sampleTokenIdx(K * C_size)#K negative samples for each of the C_size context words
    
    negative_set = set(neg_idx)
    negative_set = negative_set.difference(targets)
    negative_set = negative_set.difference(prev_idx)
    negative_idx = list(negative_set)
    
    if not(negative_idx):
        return None

    prev_idx += negative_idx

    A = outputVectors[targets + negative_idx]
    lll = [1] * C_size + [0] * len(negative_idx)
    sig_A = sigmoid(np.dot(A, predicted)) - lll
    
    h_A = sig_A * A.T
    
    if w2v:
        grad_A =  sig_A[:, np.newaxis] * predicted
    else:
        grad = None
    return h_A, negative_idx, grad_A, targets

def word2vec_sgd_wrapper(tokens, wordVectors, dataset, it, nParagraphs, word2vecCostAndGradient = negSamplingCostAndGradient, w2v = True):   
    prev_idx = []
    batchsize = 50
    inputVectors = wordVectors[:nParagraphs,:]
    outputVectors = wordVectors[nParagraphs:,:]
    count = 0
    first = True
    I_in = np.array([])
    GradIn = np.array([])
    I_out = np.array([])
    GradOut = np.array([])
    T = []

    for i in it:
        
        denom = 1
        p_id, context = i
        context_indices = [tokens[w] for w in context]
        res = word2vecCostAndGradient(inputVectors[p_id], context_indices, outputVectors, dataset, prev_idx, w2v = w2v)    
        if res:
            gin, idx_out, gout, idx_in = res
        else: 
            continue

        count += 1
        if (count < batchsize):
            if ((len(idx_in)) and (len(idx_out))):
                if (first):
                    first = False
                    I_in = idx_in
                    I_out = idx_out

                    GradIn = (np.sum(gin, axis = 1)).reshape(-1, 1) / batchsize
                    if w2v:
                        GradOut = gout
                    T = [p_id]
                else:
                    I_in = np.concatenate([I_in, idx_in])
                    I_out = np.concatenate([I_out, idx_out])
                    
                    try:
                        GradIn = np.concatenate([GradIn, (np.sum(gin, axis = 1)).reshape(-1, 1) / batchsize], axis = 1)
                    except:
                        print idx_in
                        print idx_out
                        print gin.shape
                        print gin_target.shape
                        print GradIn.shape
                        exit()
                    if w2v:
                        GradOut = np.concatenate([GradOut, gout])
                    T += [p_id]
            else:
                continue
        else:
            return I_in, GradIn, I_out, GradOut, T, False
    return I_in, GradIn, I_out, GradOut, T, True

if __name__ == "__main__":

    test_word2vec()
