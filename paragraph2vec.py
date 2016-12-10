import numpy as np
import random


from sigmoid import sigmoid, sigmoid_grad


def negSamplingCostAndGradient(predicted, context_indices, c_sizes, outputVectors, dataset, K=10, w2v = True):
    K = 25
    C_size = sum(c_sizes)

    negative_idx = dataset.sampleTokenIdx(K * C_size)

    A = outputVectors[np.concatenate((context_indices, negative_idx))]

    lll = [1] * C_size + [0] * len(negative_idx)
    sig_A = sigmoid(np.dot(A, predicted)) - lll
    
    h_A = sig_A * A.T
    
    if w2v:
        grad_A =  sig_A[:, np.newaxis] * predicted
    else:
        grad_A = None
    return h_A, negative_idx, grad_A
    
def word2vec_sgd_wrapper(tokens, wordVectors, dataset, it, nParagraphs, word2vecCostAndGradient = negSamplingCostAndGradient, w2v = True):   
    
    inputVectors = wordVectors[:nParagraphs,:]
    outputVectors = wordVectors[nParagraphs:,:]
    I_in = np.array([])
    GradIn = np.array([])
    I_out = np.array([])
    GradOut = np.array([])
    T = []

    for i in it:
        
        denom = 1
        p_id, context_indices, c_sizes = i
        batchsize = len(c_sizes)

        gin, idx_out, gout = word2vecCostAndGradient(inputVectors[p_id], context_indices, c_sizes, outputVectors, dataset, w2v = w2v)    
        
        GradIn = np.sum(gin, axis = 1) / batchsize

        if w2v:
            GradOut = gout / batchsize

        return context_indices, c_sizes, GradIn, idx_out, GradOut, p_id, False

    return np.array([]), [], GradIn, I_out, GradOut, T, True
    
    

if __name__ == "__main__":

    test_word2vec()
