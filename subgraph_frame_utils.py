from sknetwork.hierarchy import Ward, LouvainHierarchy, Paris, cut_balanced
import numpy as np
import scipy.io as io
import networkx as nx
import clustering as cl
import opt_A, opt_B
import torch
import scipy.sparse as sp

def my_save(f,idx,adj,name):
    
    data = {'f':f,'idx':np.array(idx)+1, 'A': adj}
    
    io.savemat(name+'.mat',data)

def Laplacian(adj):
    
    D = np.diag(np.array(np.sum(adj,axis = 1)).flatten())
    L = D - adj
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    return eigenvalues, eigenvectors, L

def n_Laplacian(adj):
    D = np.diag(np.power(np.array(np.sum(adj,axis = 1)).flatten(),-0.5))
    L = np.eye(adj.shape[0]) - D@adj@D
    
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors

def get_shortest_path_length(adj):
    adj = get_undirected_adj(adj)

    n = adj.shape[0]


    G = nx.from_numpy_array(adj)
    """
    E = [e for e in G.edges]
    print("!!!",len(E))
    print("!!!",E[0:20])
    print("!!!")
    input()
    """
    results = dict(nx.all_pairs_dijkstra_path_length(G))
    
    res = np.zeros((n,n))


    
    for a in results.keys():
        for b in  results[a].keys():
            res[a][b] = results[a][b]

    return res


def get_partitions(n, adj, adj_dict):
    

    print("******Generating partition tree******")
    
    cluster_size_bound = 8
    
    #method = LouvainHierarchy(random_state = 1)
    method = Ward()
    #method = Paris()
    partitions= []
    children_node_lists = []
    partitions.append([i for i in range(n)])
    prev_cluster_set = {}
    
    R = []
    
    for i in range(n):
        temp_set = set()
        temp_set.add(i)
        prev_cluster_set[i] = temp_set
    
    prev_adj = adj
    first_adj_dict = adj_dict
    prev_adj_size = n

    flag = 0
    is_first = 1


    while(prev_adj_size>cluster_size_bound+1):
    
        dendrogram = method.fit_transform(prev_adj)
        cluster_id = cut_balanced(dendrogram, cluster_size_bound)
        if flag:
            cluster_id = cl.postprocess_merge(prev_adj, cluster_id, cluster_size_bound)
            flag = 0
        
        cluster_num = max(cluster_id) + 1
        print("Number of clusters at current level:",cluster_num)
        
        
        
        temp_cluster_set = {}                    #\mathcal{S}_{j,k}
        temp_cluster_list = {}                   #\mathcal{C}_{j,k}
        
        for j in range(len(cluster_id)):
            temp_id = cluster_id[j]
            if temp_id not in temp_cluster_set.keys():
                temp_cluster_set[temp_id] = prev_cluster_set[j]
                temp_cluster_list[temp_id] = [j]
            else:
                temp_cluster_set[temp_id] = temp_cluster_set[temp_id].union(prev_cluster_set[j])
                temp_cluster_list[temp_id].append(j)
        
        
        #min_size = 50000
        
        #for k in temp_cluster_list.keys():
        #    min_size = min(min_size, len(temp_cluster_list[k]))
        
        R.append(1)
        
        children_node_lists.append(temp_cluster_list)
        
        
        prev_cluster_set = temp_cluster_set
        temp_adj = np.zeros((cluster_num,cluster_num))
        
        if is_first:
            is_first = 0
            for j in range(cluster_num):
                for k in range(j+1,cluster_num):
                    
                    edge_weight = 0.0
                    for p in temp_cluster_list[j]:
                        for q in temp_cluster_list[k]:
                            if (p,q) in first_adj_dict.keys():
                                edge_weight += first_adj_dict[(p,q)]
                        

                    

                    temp_adj[j][k] += edge_weight
                    temp_adj[k][j] += edge_weight
        else:

            for j in range(cluster_num):
                for k in range(j+1,cluster_num):
                    
                    edge_weight = 0.0
                    for p in temp_cluster_list[j]:
                        for q in temp_cluster_list[k]:
                            edge_weight += prev_adj[p][q]
                        

                    

                    temp_adj[j][k] += edge_weight
                    temp_adj[k][j] += edge_weight
        
        #adjs.append(temp_adj)
        prev_adj = temp_adj
        prev_adj_size = prev_adj.shape[0]

        temp_partition = [0 for j in range(n)]
        for j in range(cluster_num):
            temp_list = list(temp_cluster_set[j])
            for p in temp_list:
                temp_partition[p] = j
        
        partitions.append(temp_partition)
    
    if prev_adj.shape[0]>1:
        temp_cluster_list = {}
        temp_cluster_list[0] = [x for x in range(prev_adj.shape[0])];
        children_node_lists.append(temp_cluster_list)
        
        #min_size = 10000
        
        #for k in temp_cluster_list.keys():
        #    min_size = min(min_size, len(temp_cluster_list[k]))
        
        R.append(1)
    
    print("*******Generation completed******")
    return partitions, children_node_lists, R

def get_clusters_stats(children_node_lists):
    J = len(children_node_lists)
    all_cluster_size = [{} for x in range(J)]
    
    for j in range(J):
        if j==0:
            for k in children_node_lists[j].keys():
                children_id = children_node_lists[j][k]
                all_cluster_size[j][k] = len(children_id)
        else:
            for k in children_node_lists[j].keys():
                all_cluster_size[j][k] = 0
                children_id = children_node_lists[j][k]
                for l in children_id:
                    all_cluster_size[j][k] += all_cluster_size[j-1][l]
    
    min_stats = []
    max_stats = []
    for j in range(J):
        m = 10000
        M = 0
        for k in children_node_lists[j].keys():
            m = min(m,all_cluster_size[j][k])
            M = max(M,all_cluster_size[j][k])
        min_stats.append(m)
        max_stats.append(M)

    print("###",min_stats,max_stats)

def merge_cluster(children_node_lists):
    j = 1
    for k in children_node_lists[j].keys():
        temp_list = []
        children_id = children_node_lists[j][k]
        for l in children_id:
            temp_list = temp_list + children_node_lists[j-1][l]
        children_node_lists[j][k] = temp_list
    
    children_node_lists.pop(0)
    return children_node_lists
                    




def parent_nodes(rooted_T_edges, num_nodes):
    p = [0 for x in range(num_nodes)]
    for e in rooted_T_edges:
        p[e[1]] = e[0]
    
    return p

def depth(u, p):
    if u == 0:
        return 0
    
    return depth(p[u],p) + 1

def nu(x):
    
    x = np.power(x,4)*(35-84*x+70*np.power(x,2)-20*np.power(x,3))
    
    return x

def alpha(x):
    
    temp = np.zeros(x.shape)
    
    temp[np.abs(x)<0.25] = 1
    
    
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)] = \
        np.cos(np.pi/2*nu(4*np.abs(x)-1))\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)]
        
    return temp

def beta_1(x):
    
    temp = np.zeros(x.shape)
    
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)] = \
        np.sin(np.pi/2*nu(4*np.abs(x)-1))\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.25,np.abs(x)<0.5),np.abs(np.abs(x)-0.25)<1e-6),np.abs(np.abs(x)-0.5)<1e-6)]
        
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)] = \
        np.power(np.cos(np.pi/2*nu(2*np.abs(x)-1)),2)\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)]
    
    return temp

def beta_2(x):
    temp = np.zeros(x.shape)
        
    temp[np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)] = \
        (np.cos(np.pi/2*nu(2*np.abs(x)-1))*np.sin(np.pi/2*nu(2*np.abs(x)-1)))\
            [np.logical_or(np.logical_or(np.logical_and(np.abs(x)> 0.5,np.abs(x)<1),np.abs(np.abs(x)-0.5)<1e-6),np.abs(np.abs(x)-1)<1e-6)]
    
    return temp

def spectral_frame(B, J = 3): # J = 2
    
    tot = B.shape[0]
    all_id = [x for x in range(B.shape[1])]
    
    temp_B = np.zeros((3*J*B.shape[1],B.shape[1]))
    
    x = np.linspace(0,1,tot)
    cnt = 0;
    for j in range(1,J+1):
        temp_x = x/np.power(2,j-1)
        
        temp_vec_1 = alpha(temp_x)
        temp_vec_2 = beta_1(temp_x)
        temp_vec_3 = beta_2(temp_x)
        
        
        for k in range(B.shape[1]):
            u = B[:,k]
            if j == 1:
                a = np.expand_dims(temp_vec_1*u, axis=0)
                temp_B[np.ix_([cnt],all_id)] = np.dot(a,B)
                cnt += 1
            b_1 = np.expand_dims(temp_vec_2*u, axis=0)
            temp_B[np.ix_([cnt],all_id)] = np.dot(b_1,B)
            cnt += 1
            b_2 = np.expand_dims(temp_vec_3*u, axis=0)
            temp_B[np.ix_([cnt],all_id)] = np.dot(b_2,B)
            cnt += 1
            
    return temp_B
    
def get_low_highpass(sub_adj, r, spectral = False):
    
    eigenvalues, eigenvectors = Laplacian(sub_adj)
        
    
    all_id = [x for x in range(sub_adj.shape[0])]
    selected_id = [x for x in range(r)]
    remaining_id = [x for x in range(r,sub_adj.shape[0])]
    
    A = np.transpose(eigenvectors[np.ix_(all_id,selected_id)])
    B = np.transpose(eigenvectors[np.ix_(all_id,remaining_id)])
    
    if spectral is True and B.shape[0]>= 4:
        temp_B = spectral_frame(B)
        B = temp_B
    
    return A, B

def get_filters(partitions,adjs,children_node_lists, R, spectral = False):
    A = []
    B = []
    J = len(children_node_lists)
    
    for  j in range(J):
        temp_A_set = {}
        temp_B_set = {}
        
        for k in children_node_lists[j].keys():
            children_id = children_node_lists[j][k]
            sub_adj = adjs[j][np.ix_(children_id,children_id)]
            
            temp_A, temp_B = get_low_highpass(sub_adj,R[j],spectral)
            
            temp_A_set[k] = temp_A
            temp_B_set[k] = temp_B
        
        A.append(temp_A_set)
        B.append(temp_B_set)
        
        
    return A,B

def transform(f,A,B,children_node_lists):
    c = []
    d = []
    temp_c = {}
    for i in range(f.shape[1]):
        temp_c[i] = np.zeros((1,1))
        temp_c[i][0,0] = f[0,i]
    c.append(temp_c)
    
    J = len(A)
    
    for j in range(J):
        temp_c = {}
        temp_d = {}
        for k in children_node_lists[j].keys():
            tot = len(children_node_lists[j][k])
            R = c[j][children_node_lists[j][k][0]].shape[1]
            
            temp_C = np.zeros((tot,R))
            
            for l in range(tot):
                temp_C[l,:] = c[j][children_node_lists[j][k][l]]
                
            temp_c[k] = np.dot(A[j][k],temp_C).reshape(1,-1)
            temp_d[k] = np.dot(B[j][k],temp_C).reshape(1,-1)
         
        c.append(temp_c)
        d.append(temp_d)
    
    return c[-1], d
            
def inverse(c,d,A,B,children_node_lists):
    
    J = len(d)
    
    for j in range(J-1,-1,-1):
        new_c = {}
        for k in children_node_lists[j].keys():
            tot = len(children_node_lists[j][k])
            r = A[j][k].shape[0]
            m = B[j][k].shape[0]
            # print("!!!",r,m)
            
            temp_c = np.concatenate(np.split(c[k],r,axis = 1), axis = 0)
            temp_d = np.concatenate(np.split(d[j][k],m,axis = 1), axis = 0)

            temp_c = np.dot(np.transpose(A[j][k]),temp_c)
            temp_d = np.dot(np.transpose(B[j][k]),temp_d)
            
            temp_C = temp_c + temp_d
            all_id = [x for x in range(temp_C.shape[1])]
            for l in range(tot):
                new_c[children_node_lists[j][k][l]] = temp_C[np.ix_([l],all_id)]
        
        c = new_c
    
    f = np.zeros((1,len(list(c.keys()))))
    for i in c.keys():
        f[0][i] = c[i][0][0]
    
    return f


def generate_whole_graph_signal(adj,name="",poly_deg = 9):
    G = nx.from_numpy_array(adj)

    T = nx.minimum_spanning_tree(G)
    rooted_T = nx.dfs_tree(T,source = 0)
    rooted_T_edges = list(rooted_T.edges())
    p = parent_nodes(rooted_T_edges, adj.shape[0])
    
    dep = []
    for i in range(adj.shape[0]):
        dep.append(depth(i,p))
    
    max_dep = max(dep)
    c = np.random.rand(poly_deg+1)
    x = np.linspace(-1,1,num = max_dep)
    vals = np.polynomial.chebyshev.chebval(x,c)
    
    f = np.zeros((1,adj.shape[0]))
    for i in range(adj.shape[0]):
        f[0,i]  = vals[dep[i]-1]
    # idx = [x for x in range(adj.shape[0])]
    # my_save(f,idx,adj,name)
    return f


def generate_whole_graph_sine_signal(adj):
    G = nx.from_numpy_array(adj)
    T = nx.minimum_spanning_tree(G)
    rooted_T = nx.dfs_tree(T,source = 0)
    rooted_T_edges = list(rooted_T.edges())
    p = parent_nodes(rooted_T_edges, adj.shape[0])
    
    dep = []
    for i in range(adj.shape[0]):
        dep.append(depth(i,p))
    
    max_dep = max(dep)
    print("@@@",max_dep)
    x = np.linspace(0,0.2,num = max_dep)
    vals = np.sin(np.power(0.01+2*x,-1))
    
    temp_f = np.zeros((1,adj.shape[0]))
    for i in range(adj.shape[0]):
        temp_f[0,i]  = vals[dep[i]-1]
    return temp_f



def thresholding(c, d, N):
    f = []
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            f.append((c[k][0][l],(1,k,l)))
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                f.append((d[j][k][0][l],(2,j,k,l)))
    
    
    
    temp_f = sorted(f,key = lambda x: np.abs(x[0]),reverse=True)
    
    
    # threshold = np.abs(temp_f[N][0])
    
    idx_set = set([x[1] for x in temp_f[0:N]])
        
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            if (1,k,l) not in idx_set:
                c[k][0][l] = 0
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                if (2,j,k,l) not in idx_set:
                    d[j][k][0][l] = 0

def hard_thresholding(c,d,sigma):
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            if abs(c[k][0][l])-sigma > 0:
                continue
            c[k][0][l] = 0
            
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                if abs(d[j][k][0][l])-sigma > 0:
                    continue
                d[j][k][0][l] = 0
                
    
    

def save_coeff(c, d, name = '1-1', is_mat = True):
    f = []
    
    for k in c.keys():
        for l in range(c[k].shape[1]):
            f.append((c[k][0][l]))
    
    for j in range(len(d)):
        for k in d[j].keys():
            for l in range(d[j][k].shape[1]):
                f.append((d[j][k][0][l]))
    
    if not is_mat:
        with open('coeff/'+name+'.npy','wb') as file:
            np.save(file,np.array(f))    
    else:
        data = {'f':np.array(f)}
        io.savemat(name+'.mat',data)

def complement(A): 
    u,s,v = np.linalg.svd(A)
    
    m = A.shape[1]
    
    B = u[:,m:]
    
    return B  # column vectors

def all_connected_graph_Laplacian_basis(n):
    adj = np.ones((n,n))
    for i in range(n):
        adj[i][i] = 0
    

def get_support(children_node_lists):
    
    J = len(children_node_lists)

    support_dict = {}
    for j in range(J):
        for k in children_node_lists[j].keys():
            if j == 0:
                support_dict[(j,k)] = list.copy(children_node_lists[j][k])
               
            else:
                support_dict[(j,k)] = []
                for l in children_node_lists[j][k]:
                    support_dict[(j,k)] = support_dict[(j,k)]+ support_dict[(j-1,l)]
                
                
                    
            support_dict[(j,k)].sort()
    
    return support_dict
                    
def check_support(support_dict, children_node_lists):

    J = len(children_node_lists)
    for j in range(J):
        temp_list = []
        for k in children_node_lists[j].keys():
            temp_list = temp_list + support_dict[(j,k)]
        temp_list.sort()
        print("@@@_support_sum",np.sum(np.array(temp_list)))
                    
            
def get_pre_defined_A_filters(children_node_lists):  # column vectors

    J = len(children_node_lists)
    cnt = 0
    idx_map = {}
    A_list = []
    for j in range(J):
        for k in children_node_lists[j].keys():
            tot = len(children_node_lists[j][k])
            temp_A = np.sqrt(1/tot)*np.ones((tot,1))
            
            idx_map[(j,k)] = cnt
            cnt += 1
            A_list.append(temp_A)
    
    return A_list, idx_map

def get_B_filters(A_list, idx_map, children_node_lists, frame = False):
    B = []
    A = []
    J = len(children_node_lists)
    
    for  j in range(J):
        temp_A_set = {}
        temp_B_set = {}
        
        for k in children_node_lists[j].keys():
            temp_A = A_list[idx_map[(j,k)]]
            temp_B = complement(temp_A)
            temp_A_set[k] = temp_A.T
            
            temp_B_set[k] = temp_B.T

            if not frame:
                temp_B_set[k] = temp_B.T
            else:
                temp_B_set[k] = get_dense_frame(temp_B).T
        
        A.append(temp_A_set)
        B.append(temp_B_set)
    
    return A,B

def get_undirected_adj(adj):

    adj = np.copy(adj)

    n = adj.shape[0]

    for i in range(n):
        adj[i][i] = 0
        for j in range(i+1,n):
            if adj[i][j] != adj[j][i]:
                adj[i][j] = 1
                adj[j][i] = 1
    
    return adj    


def l1_norm(x):
    return np.sum(np.abs(x))

def l2_norm(x):
    return np.sum(np.square(x))

def collect_norm(F, A, B, children_node_lists, N, is_all = True):
    d_norm_list = {}
    
    for i in range(F.shape[0]):
        c,d = transform(F[i], A, B, children_node_lists)
        for j in range(len(d)):
            for k in d[j].keys():
                if (j,k) not in d_norm_list.keys():
                    d_norm_list[(j,k)] = 0.0
                d_norm_list[(j,k)] += l2_norm(d[j][k])
    
    temp_list = []
    for k in d_norm_list.keys():
        temp_list.append((d_norm_list[k],k))
    
    res = sorted(temp_list, key = lambda x: x[0], reverse = True)
    
    if is_all:
        return d_norm_list.keys()
    return set([x[1] for x in res[0:N]])
    
def select_filters(A,B,F,partitions,adjs,children_node_lists, R, tree_node_id, frame = False):  # frame set to True to generate frames
    c = [[] for x in range(F.shape[0])]
    
    temp_c = [{} for x in range(F.shape[0])]
    for j in range(F.shape[0]):
        for i in range(F.shape[2]):
            temp_c[j][i] = np.zeros((1,1))
            temp_c[j][i][0,0] = F[j][0,i]
        c[j].append(temp_c[j])
    

    J = len(children_node_lists)
    
    for  j in range(J):
        temp_c = [{} for x in range(F.shape[0])]
        temp_d = [{} for x in range(F.shape[0])]
        temp_d_2 = [{} for x in range(F.shape[0])]
        for k in children_node_lists[j].keys():
            
            temp_A = A[j][k] 
            temp_B = B[j][k]
            temp_B_f = spectral_frame(B[j][k])
            sum_1 = 0.0
            
            a = 0
            
            C_1 = []
            C_2 = []
            for i in range(F.shape[0]):
                tot = len(children_node_lists[j][k])
                s = c[i][j][children_node_lists[j][k][0]].shape[1]
                temp_C = np.zeros((tot,s))
                for l in range(tot):
                    temp_C[l,:] = c[i][j][children_node_lists[j][k][l]]
                    
                temp_c[i][k] = np.dot(temp_A,temp_C).reshape(1,-1)
                temp_d[i][k] = np.dot(temp_B,temp_C).reshape(1,-1)
                temp_d_2[i][k] = np.dot(temp_B_f,temp_C).reshape(1,-1)
                
                if (j,k) in tree_node_id:
                    C_1.append(np.dot(temp_B,temp_C).T)
                    C_2.append(np.dot(temp_B_f,temp_C).T)
                
                sum_1 += l1_norm(temp_d[i][k])
                
                
                a += l2_norm(temp_d[i][k])
                
            
            if (j,k) in tree_node_id and tot != 1:
                if not frame:
                    #print("!!!",j,k,temp_B.shape,temp_C.shape,s,C_1[0].shape)
                    p1, res1 = opt_B.get_ortho_matrix(C_1)
                    B[j][k] = p1.T@temp_B
                else:
                    p1, res1 = opt_B.get_ortho_matrix(C_2)
                    B[j][k] = p1.T@temp_B_f
            else:
                if not frame:
                    B[j][k] = temp_B
                else:
                    B[j][k] = temp_B_f
        
        for i in range(F.shape[0]):
            c[i].append(temp_c[i])
        
       
    return A,B


def concat(M_list):
    cols = M_list[0].shape[1]
    rows = 0
    
    for i in range(len(M_list)):
        rows += M_list[i].shape[0]
    
    res = np.zeros((rows,cols))

    cnt = 0
    all_id = [x for x in range(cols)]
    for i in range((len(M_list))):
        temp_id = [x for x in range(cnt, cnt + M_list[i].shape[0])]
        res[np.ix_(temp_id, all_id)] = M_list[i]
        cnt += M_list[i].shape[0]
    
    return res

    

def frame(n,A,B,children_node_lists, R):


    J = len(A)
    pre_PHI = {x : None for x in range(n)}
    res_PSI = {x:[] for x in range(J-1,-1,-1)}

    #I = np.eye(n)
    
    
    for i in range(n):
        row = [0]
        col = [i]
        num = [1]
        pre_PHI[i] = sp.csr_matrix((num, (row, col)), shape = (1, n))
        #pre_PHI[i] = np.expand_dims(I[i,:],axis=0)

    temp_R = 1
    flag = 0
    for j in range(J):
        
        if j!=0:
            temp_R *= R[j-1]
        
        cur_PHI = {}
        for k in children_node_lists[j].keys():
            
            #print("@@@",j,k,flag)
            
            
            
            
            X_list = []
            for p in range(temp_R):
                temp_M_list = []
                
                for q in children_node_lists[j][k]:
                    #print("$$$",q,p,temp_R)
                    temp = pre_PHI[q].todense()
                    #print("###",temp.shape,temp[0,:].shape)
                    #temp_M_list.append(np.expand_dims(temp[p,:], axis = 0))
                    temp_M_list.append(temp)
                    del temp
                    #temp_M_list.append(np.expand_dims(pre_PHI[q][p,:], axis = 0))
                
                #print("@@@",len(temp_M_list),temp_M_list[0].shape)
                #input()
                X_list.append(concat(temp_M_list))
            
            temp_PHI_list = [A[j][k]@x for x in X_list]
            temp_PSI_list = [B[j][k]@x for x in X_list]

            #del temp_M_list
            del X_list



            temp_PHI = concat(temp_PHI_list)

            temp_PSI = concat(temp_PSI_list)
            
            del temp_PHI_list
            del temp_PSI_list
            """
            if (j,k) == (0,473):
                print("!!!",l2_norm(temp_PSI))
            if temp_PHI.shape[0] != 5:
                flag = 1
            """
            
            cur_PHI[k] = sp.csr_matrix(temp_PHI)
            res_PSI[j].append(sp.csr_matrix(temp_PSI))

            
            del temp_PHI
            del temp_PSI
        
        pre_PHI = cur_PHI
    
    return pre_PHI, res_PSI

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def foo(x):
    pre = x.shape[0]
    x = sp.csr_matrix(x,dtype=np.float32)
    cur = x.todense().shape[0]
    x = sparse_mx_to_torch_sparse_tensor(x)

    return x

def df_svd(n, p, B):

    L = np.zeros((n,n))
    
    idx_1 = [x for x in range(1,n)]
    idx_2 = [x for x in range(n-1)]
    L[np.ix_(idx_1,idx_2)] = np.eye(n-1)
    L[0][n-1] = 1
    U,_S,V = np.linalg.svd(B)
    S = np.zeros((n,p))
    print("&&&",_S.shape,B.shape)
    for i in range(p):
        S[i][i] = _S[i]
    M = S@V.T

    C_list = []
    pre = M
    #print("&&&",n,len(C_list),C_list[0].shape,B.shape,M.shape,U.shape,S.shape,V.shape)
    for i in range(n-1):
        C_list.append((L@pre).T)
        pre = L@pre
    
    #print("&&&",n,len(C_list),C_list[0].shape,B.shape,M.shape,U.shape,S.shape,V.shape)
    if len(C_list[0].shape)!=2:
        print("&&&",n,len(C_list),C_list[0].shape,B.shape,M.shape,U.shape,S.shape,V.shape)
        input()
    all_C = U@(concat(C_list).T)

    temp_list = [B.T, all_C.T]

    res = concat(temp_list).T # column vectors

    return res

def row_normalize(M):
    cnt = 0
    tot = 0
    for i in range(M.shape[0]):
        norm = np.linalg.norm(M[i])
        if np.abs(norm)<1e-8:
            tot += 1
            continue
        M[i] = M[i]/norm
        cnt += np.linalg.norm(M[i])

    #print("$$$",tot)
    return M

def min_max_normalize(M):
    
    for i in range(M.shape[0]):
        min_val = np.min(M[i])
        max_val = np.max(M[i])
        M[i] = (M[i] - min_val) / (max_val - min_val)

    return M

def random_unit_norm_coeff(n, p):
    coeff = np.random.randn(n,p)

    coeff = row_normalize(coeff.T).T

    return coeff # column vectors

def check_frame_tightness_and_constant(F):
    c = (F@F.T)[0][0]
    f_norm_dif = np.linalg.norm((1/c)*(F@F.T)-np.eye(F.shape[0]))
    inf_norm_dif = np.max(np.abs((1/c)*(F@F.T)-np.eye(F.shape[0])))

    if c<0 or f_norm_dif > 1e-8 or inf_norm_dif>1e-8:
        print("frame not tight",F.shape,c)
        input()


    return c

def get_dense_frame(B):

    n = B.shape[1]
    if n == 0 or n == 1:
        return B
    p = min(n,3)
    #p = n
    coeff = random_unit_norm_coeff(n,p)

    F = df_svd(n, p, coeff)
    c = check_frame_tightness_and_constant(F)
    B = (1/np.sqrt(c))*B@F

    return B
