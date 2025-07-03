from sortedcontainers import SortedDict


def find_cluster_to_merge(adj, cur_cluster, cluster_id, cluster_size):
    
    min_size = 10000
    new_key = 0
    for i in cur_cluster:
        for j in range(adj.shape[1]):
            if (adj[i][j]>0 or adj[j][i]>0) and cluster_id[i] != cluster_id[j]:
                
                if min_size > cluster_size[cluster_id[j]]:
                    min_size = cluster_size[cluster_id[j]]
                    new_key = (-1*cluster_size[cluster_id[j]],cluster_id[j])
    
    """
    if new_key == 0 and min_size == 10000:
        print("###",len(cur_cluster))
    """
    return new_key

def postprocess_merge(adj, cluster_id, lower_bound):
    
    cluster_num = max(cluster_id) + 1
    cluster = {x:[] for x in range(cluster_num)}
    cluster_size = {x:0 for x in range(cluster_num)}
    
    for i in range(len(cluster_id)):
        cluster[cluster_id[i]].append(i)
        cluster_size[cluster_id[i]] += 1 
        
    
    cluster = {(-1*len(value),key) : value for key, value in cluster.items()}
    sorted_cluster = SortedDict(cluster)
    
    
    idx = cluster_num
    while(sorted_cluster.peekitem()[0][0]*(-1)<lower_bound):
        key, val = sorted_cluster.popitem()
        new_key = find_cluster_to_merge(adj, val, cluster_id, cluster_size)

        if new_key == 0:
            continue
        new_val = sorted_cluster.pop(new_key)
        new_size = key[0] + new_key[0]
        sorted_cluster[(new_size,idx)] = val + new_val
        cluster_size[idx] = -1*(new_size)
        for i in val:
            cluster_id[i] = idx
        for i in new_val:
            cluster_id[i] = idx
        idx += 1
    
    new_id_cnt = 0
    new_id_map = dict()
    for i in range(len(cluster_id)):
        if cluster_id[i] not in new_id_map.keys():
            new_id_map[cluster_id[i]] = new_id_cnt
            cluster_id[i] = new_id_cnt
            new_id_cnt += 1
        else:
            cluster_id[i] = new_id_map[cluster_id[i]]
        
    return cluster_id
    
    
    
    
        
        
    
    
    