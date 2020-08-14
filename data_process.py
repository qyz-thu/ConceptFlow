import json
import time

adj_table = dict()
data_dir = "../ConceptFlow/data/data/"
beamsearch_size = 100
backup_dir = "../../../home3/qianyingzhuo/conceptflow_data/"


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


def process_train():
    start_time = time.time()
    f_w = open(data_dir + '_testset4bs.txt', 'w')
    # log = open(data_dir + 'checklist.txt', 'w')
    three_hop = 0
    four_hop = 0
    five_hop = 0
    with open(data_dir + 'testset.txt') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print('processed %d samples, time used: %.2f' % (i, time.time() - start_time))
            if i > 999: break
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
            edge_in_path = dict()
            subgraph = list()
            memo = {'id': 0, 'long_path': []}
            for p in path:
                for pp in p:
                    if len(pp) > 3:
                        continue
                    subgraph.append(pp)

            n_data = {'post': data['post'], 'response': data['response'], 'post_ent': post_ent, 'response_ent': response_ent,
                      'paths': subgraph,
                      # 'graph_edges': edges, 'path_edges': path_edges
                      }
            f_w.write(json.dumps(n_data) + '\n')
    print(three_hop, four_hop, five_hop)
    f_w.close()
    # log.close()


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


def process1():
    """
    Add 'subgraph' & 'edges' field according to 'paths'
    """
    with open(data_dir + 'testset4bs.txt') as f:
        datas = f.readlines()
    with open(data_dir + '_testset4bs.txt', 'w') as f:
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


def process2():
    """
    count the recall and precision of conceptflow graph
    """
    response_ent = []
    recall = 0
    precision = 0
    total_graph_size = 0
    one_recall = 0
    one_precision = 0
    two_recall = 0
    with open(data_dir + 'testset4bs.txt') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            res_ent = set()
            for j in data['response_ent']:
                if j > 0:
                    res_ent.add(j)
            response_ent.append(res_ent)
    count = 0
    with open(data_dir + 'testset.txt') as f:
        for i, line in enumerate(f):
            res_ent = response_ent[i]
            if len(res_ent) == 0:
                continue
            data = json.loads(line)
            graph = set()
            # add zero hop
            post = data['post']
            for j, index in enumerate(data['post_triples']):
                if index > 0:
                    ent = post[j]
                    if ent not in entity2id:
                        continue
                    graph.add(entity2id[ent])
            # add one hop
            for oh in data['all_entities_one_hop']:
                if csk_entities[oh] in entity2id:
                    graph.add(entity2id[csk_entities[oh]])
            one_hit = len(graph & res_ent)
            one_recall += one_hit / len(res_ent)
            one_precision += one_hit / len(graph)
            # add two hop
            two_hop = set()
            for th in data['only_two']:
                if csk_entities[th] in entity2id:
                    graph.add(entity2id[csk_entities[th]])
                    two_hop.add(entity2id[csk_entities[th]])
            graph_size = len(graph)
            total_graph_size += graph_size
            hit = len(graph & res_ent)
            precision += hit / graph_size
            recall += hit / len(res_ent)
            two_recall += len(two_hop & res_ent) / len(res_ent)
            count += 1
    precision /= count
    recall /= count
    total_graph_size /= count
    one_recall /= count
    one_precision /= count
    two_recall /= count
    print("precision: %.4f recall: %.4f graph size: %.4f" % (precision, recall, total_graph_size))
    print('one hop precision: %.4f recall: %.4f' % (one_precision, one_recall))
    print('two hop recall: %.4f' % two_recall)


def process6():
    """
    generate files with full two-hop concepts for filtering
    """
    with open(data_dir + 'testset.txt') as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print("processed %d lines" % i)
            # if i == 338400: break
            data = json.loads(line)
            post = data['post']
            new_data = {'post': post, 'response': data['response']}
            response_ent = [-1 for j in range(len(data['response']))]
            for j, w in enumerate(data['response']):
                if w in entity2id:
                    response_ent[j] = entity2id[w]

            subgraph = set()
            zero_hop = set()
            for i, e in enumerate(data['post_triples']):
                if e > 0:
                    zero_hop.add(entity2id[post[i]])
            one_hop = set([entity2id[csk_entities[e]] for e in data['all_entities_one_hop'] if csk_entities[e] in entity2id])
            central_graph = zero_hop | one_hop
            two_hop = set()
            # get two hop concepts
            for oh in one_hop:
                for e in adj_table[oh]:
                    if e not in central_graph:
                        two_hop.add(e)
            outer_size = len(two_hop)
            subgraph = list(subgraph)
            # print("subgraph size: %d" % len(subgraph))
            # new_data['subgraph'] = subgraph
            new_data['central_graph'] = list(central_graph)
            new_data['outer_graph'] = list(two_hop)
            # head, tail = [], []
            # for i in range(len(subgraph)):
            #     head.append(i)
            #     tail.append(i)
            #     for j in range(i + 1, len(subgraph)):
            #         if subgraph[i] in adj_table[subgraph[j]]:
            #             head += [i, j]
            #             tail += [j, i]
            # new_data['edges'] = [head, tail]
            new_data['outer_size'] = outer_size
            with open(data_dir + '_testset_filter.txt', 'a') as f_w:
                f_w.write(json.dumps(new_data) + '\n')


def process7():
    """
    calculate the recall of gat-filtered graph
    """
    response_ent = []
    recall = 0
    count = 0
    central_recall = 0
    outer_recall = 0
    with open(data_dir + 'testset4bs.txt') as f:
        for line in f:
            data = json.loads(line)
            res_ent = set()
            for e in data['response_ent']:
                if e > 0:
                    res_ent.add(e)
            response_ent.append(res_ent)
    central_graph = []
    with open(data_dir + 'testset.txt') as f:
        for i, line in enumerate(f):
            res_ent = response_ent[i]
            if len(res_ent) == 0:
                central_graph.append([])
                continue
            data = json.loads(line)
            graph = set()
            # add zero hop
            post = data['post']
            for j, index in enumerate(data['post_triples']):
                if index > 0:
                    ent = post[j]
                    if ent not in entity2id:
                        continue
                    graph.add(entity2id[ent])
            # add one hop
            for oh in data['all_entities_one_hop']:
                if csk_entities[oh] in entity2id:
                    graph.add(entity2id[csk_entities[oh]])
            central_graph.append(graph)
    with open(data_dir + 'g3_filtered_ent') as f:
        for i, line in enumerate(f):
            res_ent = response_ent[i]
            if len(res_ent) == 0:
                continue
            data = json.loads(line)
            central = central_graph[i]
            chit = len(central & res_ent)
            central_recall += chit / len(res_ent)
            two_hop = set(data['two_hop'])
            graph = two_hop | central
            hit = len(res_ent & graph)
            recall += hit / len(res_ent)
            outer_recall += len(two_hop & res_ent) / len(res_ent)
            count += 1
    recall /= count
    central_recall /= count
    outer_recall /= count
    print('recall for filtered two-hop: %.4f %.4f %.4f' % (recall, central_recall, outer_recall))


def main():
    global entity_list, entity2id, csk_entities
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    entity2id = dict()
    with open(data_dir + 'entity.txt') as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity2id[e] = len(entity_list)
            entity_list.append(e)
    with open(data_dir + 'relation.txt') as f:
        for line in f:
            entity2id[line.strip()] = len(entity2id)
    with open(data_dir + 'resource.txt') as f:
        d = json.loads(f.readline())
    csk_entities = d['csk_entities']
    kb_dict = d['dict_csk']

    # get adjacent table
    print("get adjacency table")
    for i in range(len(entity_list)):
        adj_table[i] = set()
    for e in kb_dict:
        id1 = entity2id[e]
        for triple in kb_dict[e]:
            tokens = triple.split(',')
            sbj = tokens[0]
            obj = tokens[2][1:]
            if sbj not in entity2id or obj not in entity2id:
                continue
            id2 = entity2id[sbj] + entity2id[obj] - id1
            adj_table[id1].add(id2)
    print('done!')

    # process_train()
    process7()


main()
