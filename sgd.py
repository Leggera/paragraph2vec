# Save parameters every a few SGD iterations as fail-safe

import glob
import random
import numpy as np
import os.path as op
import cPickle as pickle

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("./saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("./saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None
    
def save_params(iter, params):
    with open("./saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, dataset, C, N, step, iterations, postprocessing = None, useSaved = False, w2v = True):
    """ Stochastic Gradient Descent """
    # Implement the stochastic gradient descent method in this        
    # function.                                                       
    
    # Inputs:                                                         
    # - f: the function to optimize, it should take a single        
    #     argument and yield two outputs, a cost and the gradient  
    #     with respect to the arguments                            
    # - x0: the initial point to start SGD from                     
    # - step: the step size for SGD                                 
    # - iterations: total iterations to run SGD for                 
    # - postprocessing: postprocessing function for the parameters  
    #     if necessary. In the case of word2vec we will need to    
    #     normalize the word vectors to have unit length.          
    # - PRINT_EVERY: specifies every how many iterations to output 
 

    # Output:                                                         
    # - x: the parameter value after SGD finishes  
    
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
            
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    
    iter_ = 0
    for epoch in range(iterations):
        if w2v:
            word = dataset.getContext(C)
        else:
            word = dataset.getTestContext(C)
        it = iter(word)#iterator over every word in the corpora

        while True:
            
            idx_in, gin, idx_out, gout, gout_target, T, finished = f(x, dataset, it)

            

            if finished:
                break

            if (len(idx_in)) and (len(idx_out)):

                if w2v:
                    h = [i + N for i in idx_out]
                    try:
                        x[h, :] -= step * gout
                    except:
                        print h
                        print x[h, :].shape
                        print gout.shape
                        exit()

                    h = [i + N for i in idx_in]
                    x[h, :] -= step * gout_target
                x[T, :] -= step * gin.T
                x = postprocessing(x)
                iter_ += 1

                if iter_ % 10000 == 0:
                    print iter_

                if iter_ % ANNEAL_EVERY == 0:
                    step *= 0.5

        print "epoch " + str(epoch + 1) + " of "+ str(iterations)
    if useSaved:
        save_params(epoch + 1, x)
    return x

