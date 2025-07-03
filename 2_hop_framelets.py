import numpy as np
import pickle
import subgraph_frame_utils as sfu
import time
import os
import scipy.sparse as sp
import opt_A
import torch
import networkx as nx
from queue import Queue

def bfs(u, one_hop, n):
    vis = [0 for x in range(n)]
    d = [0 for x in range(n)]
    
    

    q = Queue()
    q.put(u)
    vis[u] = 1

    
    while not q.empty():
        v = q.get()
        if d[v] == 3:
            break
        for i in range(len(one_hop[v])):
            w = one_hop[v][i]
            if vis[w]:
                continue
            vis[w] = 1
            d[w] = d[v] + 1
            q.put(w)
    
    res = []
    for i in range(n):
        if d[i] == 2:
            res.append(i)
    
    return res


def sparse_Laplacian_variance(A, D_inv_sqrt, v):

    temp = D_inv_sqrt.dot(v)
    temp = A.dot(temp)
    temp = D_inv_sqrt.dot(temp)

    temp = v-temp
    var = v.T.dot(temp)

    return np.diag(var)


def get_sparse_adj(dataset):
    adj_file_path = "2_hop_adj/"+dataset
    if os.path.exists(adj_file_path):
        with open(adj_file_path, "rb") as fp:
            adj_list = pickle.load(fp)
    else:
        input_file_path = 'data/'+dataset+'.npz'
        data = np.load(input_file_path)
        node_features = data['node_features']
        labels = data['node_labels']
        edges = data['edges']

        #print("!!!",labels.shape)
        #input()
        n = node_features.shape[0]
        m = edges.shape[0]
        
        one_hop = [[] for x in range(n)]
        row = []
        col = []
        num = []

        for i in range(m):
            one_hop[edges[i][0]].append(edges[i][1])
            one_hop[edges[i][1]].append(edges[i][0])
            row.append(edges[i][0])
            row.append(edges[i][1])
            col.append(edges[i][1])
            col.append(edges[i][0])
            num.append(1)
            num.append(1)
        
            
        adj = sp.csr_matrix((num, (row, col)), shape = (n, n))
        two_hop = [[] for x in range(n)]
        
        print("!!!_1")
        for i in range(n):
            print("###",i)
            res = bfs(i,one_hop,n)
            two_hop[i] = res
        
        print("!!!_2")

        row = []
        col = []
        num = []
        
        two_hop_dict = dict()

        for i in range(n):
            for j in range(len(two_hop[i])):
                row.append(i)
                col.append(two_hop[i][j])
                num.append(1.0)
                two_hop_dict[(i,two_hop[i][j])] = 1
        
        two_hop_matrix = sp.csr_matrix((num, (row, col)), shape = (n, n))

        adj_list = [adj, two_hop_matrix,two_hop_dict]
        with open(adj_file_path, "wb") as fp:
                pickle.dump(adj_list, fp)
    
    return adj_list


def get_framelets(name):
    dataset = name

    adj_list = get_sparse_adj(dataset)
    adj = adj_list[0]
    two_hop_matrix = adj_list[1]
    two_hop_dict = adj_list[2]
    n = adj.shape[0]
    
    print("###")
    G = nx.from_scipy_sparse_array(two_hop_matrix)
    print("###")
    #gen = nx.connected_components(G)
    #l = [len(c) for c in sorted(gen, key=len, reverse=True)]


    row_sum = np.array(adj.sum(1))
    row_sum=(row_sum==0)*1+row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    partitions, children_node_lists, R = sfu.get_partitions(n,two_hop_matrix,two_hop_dict)
    
    A_list, idx_map = sfu.get_pre_defined_A_filters(children_node_lists)
    A, B = sfu.get_B_filters(A_list, idx_map, children_node_lists, frame = True)
    PHI, PSI = sfu.frame(n, A, B, children_node_lists, R)
    print("!!!_3")
    J = len(A)

    idx_map = dict()
    cnt = 0
    var_list = []
    for j in range(J):
        for i in range(len(PSI[j])):
            temp = PSI[j][i].todense()
            temp = sfu.row_normalize(temp)
            #temp = sfu.min_max_normalize(temp)
            temp = temp.T
            m = temp.shape[1]
            var = sparse_Laplacian_variance(adj, D_inv_sqrt, temp)
            
            for k in range(m):
            
                idx_map[cnt + k] = (j,i,k)
                var_list.append((var[k], cnt + k))
            cnt += m

            del temp
    print("!!!_4")

    print("!!!",len(var_list))
    var_sorted = sorted(var_list,key = lambda x: np.abs(x[0]),reverse=True)
    sorted_idx = []
    sorted_var = []

    for i in range(len(var_sorted)):
        sorted_idx.append(var_sorted[i][1])
        sorted_var.append(var_sorted[i][0])
    
    """
    np.save('frame_var/'+dataset+'_var.npy',np.array(sorted_var))
    print("???")
    input()
    """

    framelet_num = 5000

    res_mat = np.zeros((framelet_num, n))
    cnt = 0
    for i in range(framelet_num):
            idx = sorted_idx[i]
            addr = idx_map[idx]

            a = addr[0]
            b = addr[1]
            c = addr[2]

            temp = PSI[a][b].todense()                       
            res_mat[i] = temp[c]
            del temp

    res_mat = res_mat.T
    
    np.save("2_hop_frame/"+dataset+"_high_var", res_mat)

    res_mat = np.zeros((framelet_num, n))
    cnt = 0
    for i in range(framelet_num):
            idx = sorted_idx[-i]
            addr = idx_map[idx]

            a = addr[0]
            b = addr[1]
            c = addr[2]

            temp = PSI[a][b].todense()                       
            res_mat[i] = temp[c]
            del temp

    res_mat = res_mat.T
    
    np.save("2_hop_frame/"+dataset+"_low_var", res_mat)


if __name__=='__main__':
    np.random.seed(42)
    get_framelets('tolokers')