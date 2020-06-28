import json
import time

adj_table = dict()
data_dir = "../ConceptFlow/data/data/"


def get_path(post_ent, res_ent):
    '''
    get shortest paths from post entities to response entities using BFS
    only get path of response entities within 5-hop
    '''
    res_ent = list(set(res_ent))
    path = list()
    # for target in res_ent:
    current = dict()
    for p in post_ent:
        current[p] = [[p]]
    traversed = set()
    i = 0
    while i < 5:
        i += 1
        j = 0
        while j < len(res_ent):
            target = res_ent[j]
            if target in current:
                path.append(current[target])
                res_ent.remove(target)
            else:
                j += 1
        # if target in current:
        #     path.append(current[target])
        #     break
        if len(res_ent) == 0:
            break
        for c in current:
            traversed.add(c)
        new_nodes = dict()
        for c in current:
            for n in adj_table[c]:
                if n in traversed:
                    continue
                new_path = [old_path + [n] for old_path in current[c]]
                if n in new_nodes:
                    new_nodes[n] += new_path
                else:
                    new_nodes[n] = new_path
        current = new_nodes
    return path


def process_train():
    with open(data_dir + 'resource.txt') as f:
        d = json.loads(f.readline())
    csk_triples = d['csk_triples']
    id2entity = d['csk_entities']

    # get adjacent table
    for i in range(len(entity_list)):
        adj_table[i] = set()
    for triple in csk_triples:
        t = triple.split(',')
        sbj = t[0]
        obj = t[2][1:]
        if sbj not in entity_list or obj not in entity_list:
            continue
        id1 = entity2id[sbj]
        id2 = entity2id[obj]
        adj_table[id1].add(id2)
        adj_table[id2].add(id1)

    start_time = time.time()
    f_w = open(data_dir + 'trainset4bs_full.txt', 'w')
    with open(data_dir + 'trainset.txt') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print('processed %d samples, time used: %.2f' % (i, time.time() - start_time))
            # if i > 99999: break
            data = json.loads(line)
            post_ent = list()
            response_ent = [-1 for i in range(len(data['response']))]
            for i in range(len(data['post_triples'])):
                if data['post_triples'][i] > 0 and data['post'][i] in entity2id:
                    post_ent.append(entity2id[data['post'][i]])
            for i, w in enumerate(data['response']):
                if w in entity2id:
                    response_ent[i] = entity2id[w]
            path = get_path(post_ent, response_ent)
            # subgraph consists of zero-hop entities and entities from all shortest path for every golden entities
            subgraph = set(post_ent)
            edge_in_path = dict()
            for p in path:
                for pp in p:
                    prior = None
                    for e in pp:
                        if id2entity[e] in entity_list:
                            subgraph.add(e)
                        if prior:
                            head = max(prior, e)
                            tail = prior + e - head
                            if head in edge_in_path:
                                edge_in_path[head].add(tail)
                            else:
                                edge_in_path[head] = {tail}
                        prior = e
            subgraph = list(subgraph)
            edges = list()
            for i in range(len(subgraph)):
                for j in range(i + 1, len(subgraph)):
                    if subgraph[j] in adj_table[subgraph[i]]:
                        edges.append([subgraph[i], subgraph[j]])
            path_edges = []
            for node in edge_in_path:
                for tail in edge_in_path[node]:
                    path_edges.append([node, tail])
            n_data = {'post': data['post'], 'response': data['response'], 'post_ent': post_ent, 'response_ent': response_ent,
                      'graph_nodes': subgraph, 'graph_edges': edges, 'path_edges': path_edges}
            f_w.write(json.dumps(n_data) + '\n')
    f_w.close()


def process_test():
    f_w = open(data_dir + 'testset4bs.txt', 'w')
    with open(data_dir + 'testset.txt') as f:
        for line in f:
            data = json.loads(line)
            post_ent = list()
            response_ent = [-1 for i in range(len(data['response']))]
            for i in range(len(data['post_triples'])):
                if data['post_triples'][i] > 0 and data['post'][i] in entity2id:
                    post_ent.append(entity2id[data['post'][i]])
            for i, w in enumerate(data['response']):
                if w in entity2id:
                    response_ent[i] = entity2id[w]
            n_data = {'post': data['post'], 'response': data['response'], 'post_ent': post_ent, 'response_ent': response_ent}
            f_w.write(json.dumps(n_data) + '\n')


def main():
    global entity_list, entity2id
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    entity2id = dict()
    with open(data_dir + 'entity.txt') as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity2id[e] = len(entity_list)
            entity_list.append(e)
    process_train()


main()
