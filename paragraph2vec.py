import numpy as np
import random


from sigmoid import sigmoid, sigmoid_grad


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, prev_idx,
    K=10, w2v = True):

    K = 25
    neg_idx = [dataset.sampleTokenIdx() for k in range(K)]
    negative_set = set(neg_idx)
    negative_set.discard(target)
    negative_set = negative_set.difference(prev_idx)
    negative_idx = list(negative_set)

    if not(negative_idx):
        return None

    prev_idx += negative_idx
    target_vector = outputVectors[target]
    U = outputVectors[negative_idx]
    negative_idx.append(target)

    negative_idx.remove(target)

    prod_t = np.dot(target_vector, predicted)
    sig = sigmoid(prod_t)
    g = sig - 1
    sig_n = sigmoid(np.dot(U, predicted))

    h =  sig_n * U.T
    
    if w2v:
        grad = sig_n[:, np.newaxis] * predicted
        grad_t = g * predicted
    else:
        grad = None
        grad_t = None
    
    return [], 0, h, g * target_vector.T, negative_idx, grad, grad_t, [target]


def skipgram(p_id, contextWords, tokens, inputVectors, outputVectors, prev_idx,
    dataset, word2vecCostAndGradient = negSamplingCostAndGradient, w2v = True):
    """ PV-DBOW model in paragraph2vec """

    predicted = inputVectors[p_id]

    p_list = map((lambda x: word2vecCostAndGradient(predicted, tokens[x], outputVectors, dataset, prev_idx, w2v = w2v)), contextWords)
    p_list = [p for p in p_list if p != None]

    k = 0
    gradIn_stack = np.array([])
    arr_neg_idx = np.array([])
    gradOut_stack = np.array([])
    gradIn_target_stack = np.array([])
    gradOut_target_stack = np.array([])
    target_indices = []
    for x in zip(*p_list):

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
        if ((k == 5) and w2v):
            gradOut_stack = np.vstack([j for j in x if np.sum(j)])
        if ((k == 6) and w2v):
            gradOut_target_stack = np.vstack([j for j in x if np.sum(j)])
        if (k == 7):
            for w in x:
                if (w):
                    target_indices += w
            break
        k += 1
    
    return target_indices, gradIn_stack, gradIn_target_stack, arr_neg_idx, gradOut_stack, gradOut_target_stack

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, it, nParagraphs, word2vecCostAndGradient = negSamplingCostAndGradient, w2v = True):   
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
    GradOut_target = np.array([])
    T = []

    for i in it:
        
        denom = 1
        p_id, context = i

        idx_in, gin, gin_target, idx_out, gout, gout_target = word2vecModel(p_id, context, tokens, inputVectors, outputVectors, prev_idx, dataset, word2vecCostAndGradient, w2v = w2v)

        count += 1
        if (count < batchsize):
            if ((len(idx_in)) and (len(idx_out))):
                if (first):
                    first = False
                    I_in = idx_in
                    I_out = idx_out
                    GradIn = (np.sum(gin, axis = 1) + np.sum(gin_target, axis = 0)).reshape(-1, 1) / batchsize
                    if w2v:
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
                    if w2v:
                        GradOut = np.concatenate([GradOut, gout])
                        GradOut_target = np.concatenate([GradOut_target, gout_target])
                    T += [p_id]
            else:
                continue
        else:
            return I_in, GradIn, I_out, GradOut, GradOut_target, T, False
    return I_in, GradIn, I_out, GradOut, GradOut_target, T, True

if __name__ == "__main__":

    test_word2vec()
