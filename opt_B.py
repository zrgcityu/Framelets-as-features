import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import numpy as np



def l1_norm(A_list, point):
    temp = 0
    for A in A_list:
        temp += np.sum(np.abs(A@point))
    return temp

def get_ortho_matrix(A_list):
    n = A_list[0].shape[1]
    manifold = pymanopt.manifolds.Stiefel(n,n)
    
    
    @pymanopt.function.autograd(manifold)
    def cost(point):
        
        res = 0
        for i in range(len(A_list)):
            C = A_list[i]@point
            D = anp.tanh(1000*C)
            res += anp.trace((C.T)@D)
        return res
            
    problem = pymanopt.Problem(manifold, cost)
    
    optimizer = pymanopt.optimizers.ConjugateGradient()
    
    result = optimizer.run(problem)
    
    return result.point, l1_norm(A_list, result.point)

