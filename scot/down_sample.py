

import numpy as np


def newsample(input_mod1,input_mod2,input_sol,p):
    ### Randomly select rows
    sol=input_sol.X.toarray()
    idx = np.random.randint(sol.shape[0],size=p)

    idy=[]
    for i in idx:
        idy.append(np.nonzero(sol[i,:])[0][0])
    idy_sort=sorted(idy)

    ## Take the idx and idy_sort of the 2D array
    new_sol = sol[np.ix_(idx,idy_sort)]
    new_mod1 = input_mod1.X.toarray()[idx,:]
    new_mod2 = input_mod2.X.toarray()[idy_sort,:]
    return new_mod1,new_mod2,new_sol






