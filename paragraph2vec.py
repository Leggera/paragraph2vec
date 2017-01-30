import numpy as np
import random


from sigmoid import sigmoid, sigmoid_grad


def negSamplingCostAndGradient(predicted, targets, neg_idx, C_size, outputVectors,
    K=10, w2v = True):
    
    negative_idx = neg_idx[~np.in1d(neg_idx, targets)]

    num_neg = len(negative_idx)

    if not(num_neg):
        return None

    A = outputVectors[np.concatenate((targets, negative_idx))]
    lll = [1] * C_size + [0] * num_neg
    sig_A = sigmoid(np.dot(A, predicted)) - lll
    
    h_A = sig_A * A.T
    
    if w2v:
        grad_A = np.dot(sig_A.reshape(1, -1).T, predicted.reshape(1, -1))
    else:
        grad_A = None
    return h_A, negative_idx, grad_A, targets

def word2vec_sgd_wrapper(tokens, wordVectors, dataset, it, C, nParagraphs, word2vecCostAndGradient = negSamplingCostAndGradient, w2v = True):   
    prev_idx = []
    batchsize = 50
    inputVectors = wordVectors[:nParagraphs,:]
    outputVectors = wordVectors[nParagraphs:,:]
    count = 0
    first = True
    I_in = np.array([])
    GradIn = np.zeros((inputVectors.shape[1], batchsize))
    I_out = np.array([])
    GradOut = np.array([])
    T = []
    K = 25
    neg_idx = dataset.sampleTokenIdx(K * C * batchsize)

    for i in it:
        
        denom = 1
        
        if (count < batchsize):
            p_id, context = i
            context_indices = [tokens[w] for w in context]
            C_size =  len(context_indices)
            res = word2vecCostAndGradient(inputVectors[p_id], context_indices, neg_idx[count * C: count * C + C_size], C_size, outputVectors, prev_idx, w2v = w2v)    
            
            if res:
                gin, idx_out, gout, idx_in = res
            else: 
                continue
            
            if ((len(idx_in)) and (len(idx_out))):
                if (first):
                    first = False
                    I_in = idx_in
                    I_out = idx_out

                    GradIn[:, count] = np.sum(gin, axis = 1) / batchsize
                    if w2v:
                        GradOut = gout
                    T = [p_id]
                else:
                    I_in = np.concatenate([I_in, idx_in])
                    I_out = np.concatenate([I_out, idx_out])
                    
                    try:
                        GradIn[:, count] = np.sum(gin, axis = 1) / batchsize
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
                count += 1
            else:
                count += 1
                continue
        else:
            return I_in, GradIn, I_out, GradOut, T, False
    return I_in, GradIn, I_out, GradOut, T, True

if __name__ == "__main__":

    test_word2vec()
