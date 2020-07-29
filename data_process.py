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


def process_train(src_path, dst_path):
    start_time = time.time()
    processed = 0
    f = open(data_dir + dst_path)
    for i in f:
        processed += 1
    f_w = open(data_dir + dst_path, 'a')
    print('process train to %s from line %d' % (dst_path, processed))
    three_hop = 0
    four_hop = 0
    five_hop = 0
    with open(data_dir + src_path) as f:
        for i, line in enumerate(f):
            if i < processed:
                continue
            if i % 1000 == 0:
                print('processed %d samples, time used: %.2f' % (i, time.time() - start_time))
            # if i > 999: break
            data = json.loads(line)
            post_ent = list()
            response_ent = [-1 for j in range(len(data['response']))]
            for j in range(len(data['post_triples'])):
                if data['post_triples'][j] > 0 and data['post'][j] in entity2id:
                    post_ent.append(entity2id[data['post'][j]])
            for j, w in enumerate(data['response']):
                if w in entity2id:
                    response_ent[j] = entity2id[w]
            path = get_path(post_ent, response_ent)

            # subgraph consists of zero-hop entities and entities from all shortest path for every golden entities
            paths = list()
            for p in path:
                for pp in p:
                    # if len(pp) > 3:
                    #     continue
                    #     memo['long_path'].append(pp)
                    #     if len(pp) == 4:
                    #         three_hop += 1
                    #     elif len(pp) == 5:
                    #         four_hop += 1
                    #     else:
                    #         five_hop += 1
                    paths.append(pp)
            subgraph = set()
            edge_list = dict()
            for path in paths:
                prior = None
                for e in path:
                    if prior:
                        head = max(prior, e)
                        tail = prior + e - head
                        if head not in edge_list:
                            edge_list[head] = set()
                        edge_list[head].add(tail)
                    prior = e
                    subgraph.add(e)
            subgraph = list(subgraph)
            edges = [[], []]
            for e in edge_list:
                for v in edge_list[e]:
                    edges[0] += [e, v]
                    edges[1] += [v, e]

            n_data = {'post': data['post'], 'response': data['response'], 'post_ent': post_ent, 'response_ent': response_ent,
                      'paths': paths, 'subgraph': subgraph, 'edges': edges
                      }
            f_w.write(json.dumps(n_data) + '\n')
    print(three_hop, four_hop, five_hop)
    f_w.close()


def process1():
    """
    Add 'subgraph' & 'edges' field according to 'paths'
    """
    with open(data_dir + 'testset4bs.txt') as f:
        datas = f.readlines()
    with open(data_dir + '__testset4bs.txt', 'w') as f:
        for data in datas:
            data = json.loads(data)
            edge_list = dict()
            paths = data['paths']
            subgraph = set()
            for path in paths:
                prior = None
                for e in path:
                    if prior:
                        head = max(prior, e)
                        tail = prior + e - head
                        if head not in edge_list:
                            edge_list[head] = set()
                        edge_list[head].add(tail)
                    prior = e
                    subgraph.add(e)
            subgraph = list(subgraph)
            edges = [[], []]
            for e in edge_list:
                for v in edge_list[e]:
                    edges[0] += [e, v]
                    edges[1] += [v, e]
            data['subgraph'] = subgraph
            data['edges'] = edges
            f.write(json.dumps(data) + '\n')


def process2(entity2id):
    """
    count the recall and precision of conceptflow graph
    """
    f = open(data_dir + 'resource.txt')
    d = json.loads(f.readline())
    f.close()
    csk_entities = d['csk_entities']
    response_ent = []
    recall = 0
    precision = 0
    total_graph_size = 0
    with open(data_dir + 'trainset4bs.txt') as f:
        for i, line in enumerate(f):
            if i > 99999:
                break
            data = json.loads(line)
            res_ent = set()
            for j in data['response_ent']:
                if j > 0:
                    res_ent.add(j)
            response_ent.append(res_ent)
    count = 0
    with open(data_dir + 'trainset.txt') as f:
        for i, line in enumerate(f):
            if i > 99999:
                break
            data = json.loads(line)
            graph = set()
            graph_size = 0
            # add zero hop
            post = data['post']
            for j, index in enumerate(data['post_triples']):
                if index > 0:
                    ent = post[j]
                    graph_size += 1
                    if ent not in entity2id:
                        continue
                    graph.add(entity2id[ent])
            # add one hop
            for oh in data['all_entities_one_hop']:
                if csk_entities[oh] in entity2id:
                    graph.add(entity2id[csk_entities[oh]])
            # add two hop
            for th in data['only_two']:
                if csk_entities[th] in entity2id:
                    graph.add(entity2id[csk_entities[th]])
            graph_size += len(data['all_entities_one_hop'])
            graph_size += len(data['only_two'])
            total_graph_size += graph_size
            res_ent = response_ent[i]
            if len(res_ent) == 0:
                continue
            hit = len(graph & res_ent)
            precision += hit / graph_size
            recall += hit / len(res_ent)
            count += 1
    precision /= count
    recall /= count
    total_graph_size /= i
    print("precision: %.4f recall: %.4f graph size: %.4f" % (precision, recall, total_graph_size))


def process3():     # not applicable
    """
    build the input for training graph decoder v3 from 'paths'
    the new 'graph_input' should have the size of (max_path_num, max_path_len, candidate_size)
    the 'output_mask' should have the same size as 'graph_input', in which 1 indicates valid input
    """
    with open(data_dir + '_trainset4bs.txt') as f:
        datas = f.readlines()
    with open(data_dir + '__trainset4bs.txt', 'w') as f:
        for data in datas:
            data = json.loads(data)
            paths = data['paths']
            post_ent = data['post_ent']
            graph_input = []
            output_mask = []
            max_path_len = max(len(p) for p in paths) + 1   # 1 <- EOP
            candidate_size = len(post_ent)
            for path in paths:
                for node in path:
                    if len(adj_table[node]) > candidate_size:
                        candidate_size = len(adj_table[node])
            for path in paths:
                path_candidate = []
                path_output_mask = []
                for i in range(max_path_len):
                    if i == 0:
                        candidate = [e for e in post_ent if e != path[0]]
                        path_candidate.append([path[0]] + candidate + [1] * (candidate_size - len(candidate) - 1))  # 1 is the padding token
                        path_output_mask.append([1] * (len(candidate) + 1) + [0] * (candidate_size - len(candidate) - 1))
                    elif i <= len(path):
                        ground_truth_ent = path[i] if i < len(path) else 0  # 0 is the EOP token
                        candidate = [e for e in adj_table[path[i-1]] if e != ground_truth_ent]
                        path_candidate.append([ground_truth_ent] + candidate + [1] * (candidate_size - len(candidate)))
                        path_output_mask.append([1] * (len(candidate) + 1) + [0] * (candidate_size - len(candidate) - 1))
                    else:
                        path_candidate.append([1] * candidate_size)
                        path_output_mask.append([0] * candidate_size)
                graph_input.append(path_candidate)
                output_mask.append(path_output_mask)
            data.pop('paths')
            data['graph_input'] = graph_input
            data['output_mask'] = output_mask
            f.write(json.dumps(data) + '\n')


def process4():
    """
    pre-compute the 'max_path_len' and 'max_candidate_size' for every sample
    """
    f_w = open(data_dir + '__trainset4bs.txt', 'w')
    with open(data_dir + 'trainset4bs500k.txt') as f:
        for line in f:
            data = json.loads(line)
            paths = data['paths']
            if len(paths) > 0:
                max_path_len = max(len(p) for p in paths) + 1   # 1 for EOP
                max_candidate_size = len(data['post_ent'])
                for path in paths:
                    for node in path:
                        if len(adj_table[node]) > max_candidate_size:
                            max_candidate_size = len(adj_table[node])
                data['max_path_len'] = max_path_len
                data['max_candidate_size'] = max_candidate_size
            else:
                data['max_path_len'] = 0
                data['max_candidate_size'] = 0
            f_w.write(json.dumps(data) + '\n')


def main():
    global entity_list, entity2id
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    entity2id = dict()
    with open(data_dir + 'entity.txt') as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity2id[e] = len(entity_list)
            entity_list.append(e)
    with open(data_dir + 'resource.txt') as f:
        d = json.loads(f.readline())
    csk_triples = d['csk_triples']

    # get adjacent table
    print("get adjacency table")
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
    print("done!")

    process4()
    # process_train('trainset.txt', 'trainset4bs_full.txt')


# main()
with open(data_dir + 'resource.txt') as f:
    data = f.read()
    data = json.loads(data)
    pass
