#coding:utf-8
import numpy as np
import json
import torch

        
def prepare_data(config):
    global csk_entities, csk_triples, kb_dict, dict_csk_entities, dict_csk_triples
    
    with open('%s/resource.txt' % config.data_dir) as f:
        d = json.loads(f.readline())
    
    csk_triples = d['csk_triples']
    csk_entities = d['csk_entities']
    raw_vocab = d['vocab_dict']
    kb_dict = d['dict_csk']
    dict_csk_entities = d['dict_csk_entities']
    dict_csk_triples = d['dict_csk_triples']
    
    data_train, data_test = [], []

    if config.is_train:
        with open('%s/trainset4bs.txt' % config.data_dir) as f:
            for idx, line in enumerate(f):
                if idx == 99999: break

                if idx % 100000 == 0:
                    print('read train file line %d' % idx)
                data_train.append(json.loads(line))
    
    with open('%s/testset4bs.txt' % config.data_dir) as f:
        for line in f:
            data_test.append(json.loads(line))
    
    return raw_vocab, data_train, data_test


def build_vocab(path, raw_vocab, config, trans='transE'):
    print("Creating word vocabulary...")
    vocab_list = ['_PAD', '_GO', '_EOS', '_UNK', ] + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    if len(vocab_list) > config.symbols:
        vocab_list = vocab_list[:config.symbols]

    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T'] 
    with open('%s/entity.txt' % path) as f:
        for line in f:
            entity_list.append(line.strip())
    
    print("Creating relation vocabulary...")
    relation_list = []
    with open('%s/relation.txt' % path) as f:
        for line in f:
            relation_list.append(line.strip())

    print('Creating adjacency table...')
    entity2id = dict()
    adj_table = dict()
    for i, e in enumerate(entity_list):
        entity2id[e] = i
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

    print("Loading word vectors...")
    vectors = {}
    with open('%s/glove.840B.300d.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("processing %d word vectors" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = vectors[word].split()
        else:
            vector = np.zeros((config.embed_units), dtype=np.float32) 
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
            
    print("Loading entity vectors...")
    entity_embed = []
    with open('%s/entity_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            entity_embed.append(s)

    print("Loading relation vectors...")
    relation_embed = []
    with open('%s/relation_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            relation_embed.append(s)

    entity_relation_embed = np.array(entity_embed + relation_embed, dtype=np.float32)
    entity_embed = np.array(entity_embed, dtype=np.float32)
    relation_embed = np.array(relation_embed, dtype=np.float32)

    word2id = dict()
    entity2id = dict()
    for word in vocab_list:
        word2id[word] = len(word2id)
    for entity in entity_list + relation_list:
        entity2id[entity] = len(entity2id)

    return word2id, entity2id, vocab_list, embed, entity_list, entity_embed, relation_list, relation_embed, entity_relation_embed, adj_table


def gen_batched_data(data, config, word2id, entity2id, is_inference=False):
    global csk_entities, csk_triples, kb_dict, dict_csk_entities, dict_csk_triples

    encoder_len = max([len(item['post']) for item in data]) + 1
    decoder_len = max([len(item['response']) for item in data]) + 1
    entity_len = max([len(item['paths']) for item in data])
    posts_id = np.full((len(data), encoder_len), 0, dtype=int)  # todo: change to np.zeros?
    responses_id = np.full((len(data), decoder_len), 0, dtype=int)
    post_ent = []
    post_ent_len = []
    response_ent = []
    responses_length = []
    subgraph = []
    subgraph_length = []
    paths = []
    edges = []
    match_entity = np.full((len(data), decoder_len), -1, dtype=int)

    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l - len(sent) - 1)

    next_id = 0
    for item in data:
        # posts
        for i, post_word in enumerate(padding(item['post'], encoder_len)):
            if post_word in word2id:
                posts_id[next_id, i] = word2id[post_word]
                
            else:
                posts_id[next_id, i] = word2id['_UNK']

        # responses
        for i, response_word in enumerate(padding(item['response'], decoder_len)):
            if response_word in word2id:
                responses_id[next_id, i] = word2id[response_word]
                
            else:
                responses_id[next_id, i] = word2id['_UNK']

        # responses_length
        responses_length.append(len(item['response']) + 1)

        # post&response entities
        post_ent.append(item['post_ent'])
        post_ent_len.append(len(item['post_ent']))
        response_ent.append(item['response_ent'] + [-1 for j in range(decoder_len - len(item['response_ent']))])

        # ground-truth path
        for i in range(len(item['paths'])):
            item['paths'][i].append(0)
        paths.append(item['paths'])

        # if not is_inference:
        subgraph_tmp = item['subgraph']
        subgraph_len_tmp = len(subgraph_tmp)
        # subgraph_tmp += [1] * (entity_len - len(subgraph_tmp))
        # for i in range(len(subgraph_tmp)):
        #     subgraph_len_tmp.append(len(subgraph_tmp[i]))
        #     subgraph_tmp[i] += [1] * (beamsearch_width - len(subgraph_tmp[i]))
        subgraph.append(subgraph_tmp)
        subgraph_length.append(subgraph_len_tmp)

        edges.append(item['edges'])

        g2l = dict()
        for i in range(len(subgraph_tmp)):
            g2l[subgraph_tmp[i]] = i

        for i in range(len(item['response_ent'])):
            if item['response_ent'][i] == -1:
                continue
            if item['response_ent'][i] not in g2l:
                continue
            else:
                match_entity[next_id, i] = g2l[item['response_ent'][i]]

        next_id += 1

    batched_data = {'post_text': np.array(posts_id),
                    'response_text': np.array(responses_id),
                    'subgraph': np.array(subgraph),
                    'subgraph_size': subgraph_length,
                    'paths': paths,
                    'edges': edges,
                    'responses_length': responses_length,
                    'post_ent': post_ent,
                    'post_ent_len': post_ent_len,
                    'response_ent': response_ent,
                    'match_entity': np.array(match_entity),
                    'word2id': word2id,
                    'entity2id': entity2id,
                    }
    
    return batched_data
