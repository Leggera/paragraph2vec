# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 10000

import glob
import random
import numpy as np
import os.path as op
import cPickle as pickle

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("/data/saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("/data/saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None
    
def save_params(iter, params):
    with open("/data/saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, N, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
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
    
    #for iter_ in xrange(start_iter + 1, iterations + 1):
    iter_ = 0
    while True:
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.
        #count = 0
        ### YOUR CODE HERE
        #cost, idx_in, gin, idx_out, gout, gout_target, T = f(x)
        idx_in, gin, idx_out, gout, gout_target, T, finished = f(x)
        if finished:
            return x
        if (len(idx_in)) and (len(idx_out)):
            '''u, indices = np.unique(idx_out, return_index=True)
            t_v = np.in1d(u, idx_in)
            #print cost, gradient
            x[u[t_v]] -= step * gout[indices[t_v]]
            x[idx_in] -= step * gout_target
            gin_sum = np.sum(gin[:, indices], axis = 1) + np.sum(gin_target, axis = 0)
            x[idx_in] -= step * gin_sum'''
            h = [i + N for i in idx_out]
            try:
                x[h, :] -= step * gout
            except:
                print h
                print x[h, :].shape
                print gout.shape
                exit()
            #print x[h, :]
            h = [i + N for i in idx_in]
            x[h, :] -= step * gout_target
            #print x[h, :]
            x[T, :] -= step * gin.T
            #print x[T, :]        
            #print len(h)
            '''print len(T)#centerword amount
            print x[T, :].shape
            print gin.shape'''
            #print (np.sum(gin, axis=1) + np.sum(gin_target, axis = 0)).shape
            #print("ok")
            #exit()
            #print x
            x = postprocessing(x)
            ### END YOUR CODE
            #print cost
            '''if iter_ % PRINT_EVERY == 0:
                if not expcost:
                    expcost = cost
                else:
                    expcost = .95 * expcost + .05 * cost
                print "iter %d: %f" % (iter_, expcost)'''
            iter_ += 1
            if iter_ % SAVE_PARAMS_EVERY == 0 and useSaved:
                save_params(iter_, x)
                
            if iter_ % ANNEAL_EVERY == 0:
                step *= 0.5
            
    print x.shape
    return x

def sanity_check():
    quad = lambda x: (np.sum(x ** 2), x * 2)

    #print "Running sanity checks..."
    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100000)
    #print "test 1 result:", t1
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100000)
    #print "test 2 result:", t2
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100000)
    #print "test 3 result:", t3
    assert abs(t3) <= 1e-6
    
    #print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q3_sgd.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    print timeit.timeit('sanity_check()', setup = "from __main__ import sanity_check", number=4000);
    #your_sanity_checks();
