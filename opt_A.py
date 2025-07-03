import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import time

def get_manifold(children_node_lists, R):
    
    J = len(children_node_lists)
    cnt = 0
    idx_map = {}
    manifold_list = []
    for j in range(J):
        for k in children_node_lists[j].keys():
            tot = len(children_node_lists[j][k])
            temp_manifold = pymanopt.manifolds.Stiefel(tot,R[j])
            idx_map[(j,k)] = cnt
            cnt += 1
            manifold_list.append(temp_manifold)
    
    return pymanopt.manifolds.Product(manifold_list), idx_map

def get_ortho_matrix(F,children_node_lists, R):
    
    n_f = F.shape[0]
    res_manifold, idx_map = get_manifold(children_node_lists, R)
    C = [{} for x in range(n_f)]
    for i in range(n_f):
        for j in range(F[0].shape[1]):
            C[i][j] = anp.zeros((1,1))
            C[i][j][0,0] = F[i,0,j]
    
    @pymanopt.function.autograd(res_manifold)
    def cost(*point): 
        
        J = len(children_node_lists)
        c = C
        for j in range(J):
            
            temp_c = [{} for x in range(n_f)]
            for k in children_node_lists[j].keys():
                idx = idx_map[(j,k)]
                for i in range(n_f):
                    
                    tot = len(children_node_lists[j][k])
                    
                    temp_C = None
                    
                    for l in range(tot):
                        if not l:
                            temp_C = c[i][children_node_lists[j][k][l]]
                        else:
                            temp_C = anp.concatenate((temp_C,c[i][children_node_lists[j][k][l]]),axis = 0)
                        
                    temp_c[i][k] = anp.dot(anp.transpose(point[idx]),temp_C).reshape(1,-1)
            c = temp_c
        res = 0.0
        for i in range(len(c)):
            res += anp.sum(anp.square(c[i][0]))
        return -1.0*res

    problem = pymanopt.Problem(res_manifold, cost)
    
    optimizer = pymanopt.optimizers.ConjugateGradient()
    
    s = time.time()
    result = optimizer.run(problem)
    t = time.time()
    print("TIME:",t-s)
    return result.point, idx_map