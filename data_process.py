import json

adj_table = dict()


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


# # test
# adj_table = {1: [2, 3, 4, 8], 2: [1, 5, 6], 3: [1, 7], 4: [1], 5: [2, 6, 8], 6: [2, 7, 8], 7: [3, 6], 8: [1, 5, 6]}
# path = get_path([1], [6])
# pass

data_dir = "../ConceptFlow/data/data/"
with open(data_dir + 'resource.txt') as f:
    d = json.loads(f.readline())
csk_triples = d['csk_triples']
csk_entities = d['csk_entities']
raw_vocab = d['vocab_dict']
entity2id = d['dict_csk_entities']

# get adjacent table
for i in range(len(csk_triples)):
    adj_table[i] = list()
for triple in csk_triples:
    t = triple.split(',')
    sbj = t[0]
    obj = t[2][1:]
    id1 = entity2id[sbj]
    id2 = entity2id[obj]
    adj_table[id1].append(id2)
    adj_table[id2].append(id1)

f_w = open(data_dir + 'trainset4bs.txt', 'w')
with open(data_dir + 'trainset.txt') as f:
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print('processed %d samples' % i)
        if i > 99999:
            break
        data = json.loads(line)
        post_ent = list()
        response_ent = list()
        for i in range(len(data['post_triples'])):
            if data['post_triples'][i] > 0:
                post_ent.append(entity2id[data['post'][i]])
        for w in data['response']:
            if w in entity2id:
                response_ent.append(entity2id[w])
        path = get_path(post_ent, response_ent)
        # subgraph consists of zero-hop entities and entities from all shortest path for every golden entities
        subgraph = set(post_ent)
        for p in path:
            for pp in p:
                for e in pp:
                    subgraph.add(e)
        subgraph = list(subgraph)
        edges = list()
        for i in range(len(subgraph)):
            for j in range(i + 1, len(subgraph)):
                if subgraph[j] in adj_table[subgraph[i]]:
                    edges.append([subgraph[i], subgraph[j]])
        n_data = {'post': data['post'], 'response': data['response'], 'post_ent': post_ent, 'response_ent': response_ent,
                  'graph_nodes': subgraph, 'graph_edges': edges}
        f_w.write(json.dumps(n_data) + '\n')
f_w.close()
