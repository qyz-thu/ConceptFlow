#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils
from torch.nn.parameter import Parameter
import dgl
from dgl.nn import GATConv

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


class GAT(nn.Module):
	def __init__(self, features):
		super(GAT, self).__init__()
		self.gat1 = GATConv(in_feats=features, out_feats=features, num_heads=1)
		self.gat2 = GATConv(in_feats=features, out_feats=features, num_heads=1)

	def forward(self, g, inputs):
		if g.number_of_edges() == 0:
			return inputs
		h = self.gat1(g, inputs)
		h = torch.relu(h)
		h = self.gat2(g, h)
		return h


class RGATLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super(RGATLayer, self).__init__()
        self.fc = nn.Linear(in_features=node_dim, out_features=out_dim, bias=False)
        self.relation_fc = nn.Linear(in_features=node_dim + edge_dim, out_features=node_dim)
        self.attn_fc = nn.Linear(in_features=2 * out_dim, out_features=1, bias=False)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.relation_fc.weight, gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain)

    def edge_attention(self, edges):
        # edge UDF for calculating 'z' for each edge
        end = self.relation_fc(torch.cat([edges.data['h'], edges.dst['z']], dim=1))
        a = self.attn_fc(torch.cat([edges.src['z'], end], dim=1))
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # edge UDF propagating 'z' and coefficient 'e'
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # node UDF executing weighted aggregation for output
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, input):
        g.ndata['z'] = self.fc(input)
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class RGAT(nn.Module):
    def __init__(self, ent_dim, rel_dim, layers=1, num_head=1):
        super(RGAT, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            heads = nn.ModuleList()
            for __ in range(num_head):
                heads.append(RGATLayer(ent_dim, rel_dim, ent_dim))
            self.layers.append(heads)

    def forward(self, graph, input):
        h = input
        for layer in self.layers:
            head_outs = [head(graph, h) for head in layer]
            h = torch.mean(torch.stack(head_outs), dim=0)
        return h


class ConceptFlow(nn.Module):
    def __init__(self, config, word_embed, entity_embed, adj_table):
        super(ConceptFlow, self).__init__()
        self.is_inference = False
        self.decode = config.decode
        # Encoder
        self.fact_scale = config.fact_scale
        self.trans_units = config.trans_units
        self.embed_units = config.embed_units
        self.units = config.units
        self.layers = config.layers
        self.gnn_layers = config.gnn_layers
        self.symbols = config.symbols
        self.bs_width = config.beam_search_width
        self.max_hop = config.max_hop

        self.adj_table = adj_table
        self.word_embedding = nn.Embedding(num_embeddings=word_embed.shape[0], embedding_dim=self.embed_units, padding_idx=0)
        self.word_embedding.weight = nn.Parameter(use_cuda(torch.Tensor(word_embed)))
        self.word_embedding.weight.requires_grad = True

        self.entity_embedding = nn.Embedding(num_embeddings=entity_embed.shape[0] + 7, embedding_dim=self.trans_units, padding_idx=0)
        entity_embed = torch.Tensor(entity_embed)

        entity_embed = torch.cat((torch.zeros(7, self.trans_units), entity_embed), 0)
        self.entity_embedding.weight = nn.Parameter(use_cuda(torch.Tensor(entity_embed)))
        self.entity_embedding.weight.requires_grad = True
        self.entity_linear = nn.Linear(in_features=self.trans_units, out_features=self.trans_units)

        self.softmax_d0 = nn.Softmax(dim=0)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.softmax_d2 = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

        self.text_encoder = nn.GRU(input_size=self.embed_units, hidden_size=self.units, num_layers=self.layers, batch_first=True)
        self.graph_decoder = nn.LSTM(input_size=self.units, hidden_size=self.units, num_layers=self.layers, batch_first=True)
        self.decoder = nn.GRU(input_size=self.units + self.embed_units, hidden_size=self.units, num_layers=self.layers, batch_first=True)

        self.attn_c_linear = nn.Linear(in_features=self.units, out_features=self.units, bias=False)
        self.context_linear = nn.Linear(in_features=2 * self.units + self.trans_units, out_features=self.units, bias=False)

        # matrix for graph attention
        self.graph_attn_linear = nn.Linear(in_features=self.trans_units, out_features=self.units, bias=False)
        # linear layer to convert z in graph decoder
        self.graph_convt_linear = nn.Linear(in_features=self.trans_units + self.units, out_features=self.units)
        self.graph_prob_linear = nn.Linear(in_features=self.units, out_features=self.trans_units, bias=False)
        self.relation_linear = nn.Linear(in_features=2 * self.trans_units, out_features=self.trans_units)
        self.bias = Parameter(torch.FloatTensor(1).zero_())

        # GAT
        self.GAT = GAT(self.trans_units)

        # Loss
        self.logits_linear = nn.Linear(in_features=self.units, out_features=self.symbols)
        self.selector_linear = nn.Linear(in_features=self.units, out_features=2)

    def forward(self, batch_data):
        query_text = batch_data['post_text']    # todo: rename query_text to post_text
        response_text = batch_data['response_text']
        responses_length = batch_data['responses_length']
        post_ent = batch_data['post_ent']
        post_ent_len = batch_data['post_ent_len']
        response_ent = batch_data['response_ent']
        subgraph = batch_data['subgraph']
        subgraph_len = batch_data['subgraph_size']
        match_entity = batch_data['match_entity']
        edges = batch_data['edges']

        if not self.is_inference:
            graph_input = batch_data['graph_input']
            graph_relation = batch_data['graph_relation']
            output_mask = batch_data['output_mask']
            max_path_num = output_mask.shape[1]
            max_path_len = output_mask.shape[2]
            path_num = batch_data['path_num']
            path_len = batch_data['path_len']
            max_candidate_size = batch_data['max_candidate_size']
            train_graph_nodes = batch_data['train_graph_nodes']
            train_graph_edges = batch_data['train_graph_edges']

        if self.is_inference:
            word2id = batch_data['word2id']
            entity2id = batch_data['entity2id']
            id2entity = dict()
            for key in entity2id.keys():
                id2entity[entity2id[key]] = key
        else:
            id2entity = None

        batch_size = query_text.shape[0]
        # numpy to tensor
        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        response_text = use_cuda(Variable(torch.from_numpy(response_text).type('torch.LongTensor'), requires_grad=False))
        responses_length = use_cuda(Variable(torch.Tensor(responses_length).type('torch.LongTensor'), requires_grad=False))
        query_mask = use_cuda((query_text != 0).type('torch.FloatTensor'))

        if not self.is_inference:
            graph_input = self.get_graph_input(graph_input, train_graph_nodes, train_graph_edges, graph_relation, batch_size, max_path_num, max_path_len, max_candidate_size)
            output_mask = use_cuda(Variable(torch.from_numpy(output_mask).type('torch.LongTensor'), requires_grad=False))
            graph_target = use_cuda(torch.FloatTensor(output_mask.size()).fill_(0))
            total_sample = torch.sum(output_mask)
            for i in range(batch_size):
                for j in range(path_num[i]):
                    for k in range(path_len[i][j]):
                        graph_target[i, j, k, 0].fill_(1)

        decoder_len = response_text.shape[1]
        responses_target = response_text
        responses_id = torch.cat((use_cuda(torch.ones([batch_size, 1]).type('torch.LongTensor')),torch.split(response_text, [decoder_len - 1, 1], 1)[0]), 1)

        # text encoder
        text_encoder_input = self.word_embedding(query_text)
        text_encoder_output, text_encoder_state = self.text_encoder(text_encoder_input, use_cuda(Variable(torch.zeros(self.layers, batch_size, self.units))))

        # graph decoder
        if not self.is_inference:
            graph_input = graph_input.contiguous().view(batch_size * max_path_num, max_path_len, -1, self.trans_units)    # todo: use reshape?
            text_hidden_state = text_encoder_state[self.layers - 1].unsqueeze(1)   # use the hidden states of the last layer in t=seq_len
            graph_context = use_cuda(torch.empty(0))
            for b in range(batch_size): # get graph_context of shape (bs * max_path_num, 1, D), D is the dimension of hidden states
                for n in range(max_path_num):
                    graph_context = torch.cat([graph_context, text_hidden_state[b:b+1, :, :]], dim=0)
            graph_decoder_state = self.init_hidden(self.layers, batch_size * max_path_num, self.units)
            for t in range(max_path_len):
                if t == 0:
                    h = graph_context
                else:
                    ground_truth = graph_input[:, t-1:t, 0]     # (bs * max_num, 1, trans_units)
                    graph_context = self.graph_convt_linear(torch.cat([graph_context, ground_truth], dim=2))
                    graph_output, graph_decoder_state = self.graph_decoder(graph_context, graph_decoder_state)
                    h = torch.cat([h, graph_output], dim=1)
            # size of h: (batch*max_num, max_len, D)
            logits = torch.matmul(graph_input, self.graph_prob_linear(h).unsqueeze(3)).reshape(batch_size, max_path_num, max_path_len, -1)
            logits += self.bias
            retrieval_loss = F.binary_cross_entropy_with_logits(logits, graph_target, weight=output_mask, reduction='sum')
            retrieval_loss /= total_sample
        else:
            subgraph, edges, subgraph_len = [], [], []
            match_entity = [[] for bs in range(batch_size)]
            for b in range(batch_size):

                for t in range(self.max_hop + 1):
                    if t == 0:  # select zero-hop
                        graph_context = text_encoder_state[self.layers-2: self.layers-1, b: b+1, :].transpose(0, 1)
                        graph_decoder_state = self.init_hidden(self.layers, 1, self.units)
                        graph_output, graph_decoder_state = self.graph_decoder(graph_context, graph_decoder_state)

                        # get entity representation from GAT layer
                        graph = dgl.DGLGraph()
                        head, tail = [], []
                        all_nodes = dict()
                        for node in post_ent[b]:
                            all_nodes[node] = len(all_nodes)
                        for i in range(len(post_ent[b])):
                            n1 = post_ent[b][i]
                            head.append(all_nodes[n1])
                            tail.append(all_nodes[n1])
                            for j in range(i + 1, len(all_nodes)):
                                n2 = post_ent[b][j]
                                if n1 in self.adj_table[n2]:
                                    head += [all_nodes[n1], all_nodes[n2]]
                                    tail += [all_nodes[n2], all_nodes[n1]]
                        graph.add_nodes(len(post_ent[b]))
                        graph.add_edges(head, tail)

                        candidate_embed = self.entity_embedding(use_cuda(torch.LongTensor(post_ent[b])))
                        gat_output = self.GAT(graph, candidate_embed).squeeze() # (N, trans_units), N is the size of candidates

                        logits = torch.matmul(gat_output, self.graph_prob_linear(graph_output.squeeze())).reshape([-1]) # (N)
                        logits += self.bias
                        prob = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                        sorted_prob = [[i, prob[i]] for i in range(len(prob))]
                        sorted_prob.sort(key=lambda x: x[1], reverse=True)
                        sorted_prob = sorted_prob[:self.bs_width]
                        next_ent = [post_ent[b][x[0]] for x in sorted_prob]

                        next_ent += [1] * (self.bs_width - len(next_ent))   # 1 for padding
                        all_paths = [[x] for x in next_ent]
                        paths_prob = [x[1] for x in sorted_prob]

                        # get graph context for all paths
                        # use the 'batch' dimension for multiple paths
                        all_graph_context, all_state1, all_state2 = use_cuda(torch.empty(0)), use_cuda(torch.empty(0)), use_cuda(torch.empty(0))
                        for i in range(len(next_ent)):
                            all_state1 = torch.cat([all_state1, graph_decoder_state[0]], dim=1)
                            all_state2 = torch.cat([all_state2, graph_decoder_state[1]], dim=1)
                            all_graph_context = torch.cat([all_graph_context, graph_context], dim=0)
                        all_graph_decoder_state = (all_state1, all_state2)
                        continue

                    embed = self.entity_embedding(use_cuda(torch.LongTensor(next_ent))).unsqueeze(1)
                    all_graph_context = self.graph_convt_linear(torch.cat([all_graph_context, embed], dim=2))
                    all_graph_output, all_graph_decoder_state = self.graph_decoder(all_graph_context, all_graph_decoder_state)
                    all_logits = use_cuda(torch.empty(0))

                    past_probs = []
                    path_candidates = []

                    # use all_paths to create graph for every path
                    for i, e in enumerate(next_ent):
                        if e != 0 and e != 1:
                            candidates = [x for x in self.adj_table[e] if x not in all_paths[i]] + [0] # global index of candidates
                            relations = [self.adj_table[e][x] if x != 0 else entity2id['RelatedTo'] for x in candidates]
                            path_candidates += [all_paths[i] + [x] for x in candidates]
                            graph_nodes = all_paths[i] + candidates
                            all_nodes = dict()
                            for x in graph_nodes:
                                all_nodes[x] = len(all_nodes)
                            graph = dgl.DGLGraph()
                            graph.add_nodes(len(graph_nodes))
                            head, tail = [], []
                            for j in range(len(graph_nodes)):
                                n1 = graph_nodes[j]
                                head.append(all_nodes[n1])
                                tail.append(all_nodes[n1])
                                for k in range(j + 1, len(graph_nodes)):
                                    n2 = graph_nodes[k]
                                    if n1 in self.adj_table[n2]:
                                        head += [all_nodes[n1], all_nodes[n2]]
                                        tail += [all_nodes[n1], all_nodes[n2]]
                            graph.add_edges(head, tail)
                            previous_embed = self.entity_embedding(use_cuda(torch.LongTensor(all_paths[i])))
                            node_embed = self.entity_embedding(use_cuda(torch.LongTensor(candidates)))
                            rel_embed = self.entity_embedding(use_cuda(torch.LongTensor(relations)))
                            new_embed = self.relation_linear(torch.cat([node_embed, rel_embed], dim=1))
                            gat_output = self.GAT(graph, torch.cat([previous_embed, new_embed], dim=0))
                            candidates_local = [all_nodes[x] for x in candidates]   # local index of candidates
                            candidate_embed = gat_output[candidates_local].squeeze()

                            past_probs += [paths_prob[i]] * len(candidates)
                            graph_output = all_graph_output.squeeze()[i]
                            logits = torch.matmul(candidate_embed, self.graph_prob_linear(graph_output)).reshape([-1])
                            logits += self.bias
                            all_logits = torch.cat([all_logits, logits], dim=0)
                    all_probs = torch.sigmoid(all_logits).detach().cpu().numpy().tolist()
                    all_probs = [all_probs[i] * past_probs[i] for i in range(len(all_probs))]
                    sorted_prob = [[i, all_probs[i]] for i in range(len(all_probs))]
                    sorted_prob.sort(key=lambda x: x[1], reverse=True)
                    sorted_prob = sorted_prob[: self.bs_width]
                    index = 0
                    new_ent = []
                    new_paths = []
                    new_prob = []
                    for i in range(self.bs_width):
                        if next_ent[i] == 0:
                            new_ent.append(0)
                            new_paths.append(all_paths[i])
                            new_prob.append(paths_prob[i])
                        else:
                            new_ent.append(path_candidates[sorted_prob[index][0]][-1])
                            new_paths.append(path_candidates[sorted_prob[index][0]])
                            new_prob.append(sorted_prob[index][1])
                            index += 1
                            if index >= len(sorted_prob):
                                break
                    next_ent = new_ent + [1] * (self.bs_width - len(new_ent))
                    all_paths = new_paths
                    paths_prob = new_prob
                    if sum(next_ent) == 0:
                        break

                # get subgraph from retrieved paths
                graph_nodes = set()
                graph_edges = [[], []]
                edge_list = dict()
                for path in all_paths:
                    prior = None
                    for node in path:
                        if node == 0:
                            break
                        graph_nodes.add(node)
                        if prior:
                            head = max(prior, node)
                            tail = prior + node - head
                            if head not in edge_list:
                                edge_list[head] = set()
                            edge_list[head].add(tail)
                        prior = node
                for head in edge_list:
                    for tail in edge_list[head]:
                        graph_edges[0] += [head, tail]
                        graph_edges[1] += [tail, head]
                graph_nodes = list(graph_nodes)
                graph_edges[0] += graph_nodes
                graph_edges[1] += graph_nodes
                subgraph.append(graph_nodes)
                subgraph_len.append(len(graph_nodes))
                edges.append(graph_edges)
                # process match_entity
                g2l = dict()
                for i in range(len(graph_nodes)):
                    g2l[graph_nodes[i]] = i
                for i in range(len(response_ent[b])):
                    index = g2l[response_ent[b][i]] if response_ent[b][i] in g2l else -1
                    match_entity[b].append(index)
        max_graph_size = max(subgraph_len)
        if self.is_inference:
            for b in range(batch_size):
                match_entity[b] += [-1] * (max_graph_size - len(match_entity[b]))

        # get recall & precision
        if self.is_inference:
            response_ent_num = 0
            found_num = 0
            for b in range(batch_size):
                entities = set()
                for i in range(len(subgraph[b])):
                    if subgraph[b][i] > 0:
                        entities.add(subgraph[b][i])
                for d in range(decoder_len):
                    if response_ent[b][d] == -1:
                        continue
                    response_ent_num += 1
                    if response_ent[b][d] in entities:
                        found_num += 1
            total_graph_size = sum(subgraph_len)
            recall = found_num / response_ent_num
            precision = found_num / total_graph_size

        if not self.decode:
            if self.is_inference:
                return recall, precision, total_graph_size
            return retrieval_loss

        # get subgraph representation
        graph_list = self.construct_graph(subgraph, edges)
        batched_graph = dgl.batch(graph_list)
        graph_embed = self.gnn(batched_graph, batched_graph.ndata['h'])
        # text decoder input
        decoder_input = self.word_embedding(responses_id)

        # attention
        c_attention_keys = self.attn_c_linear(text_encoder_output)
        c_attention_values = text_encoder_output

        decoder_state = text_encoder_state
        decoder_output = use_cuda(torch.empty(0))
        ce_alignments = use_cuda(torch.empty(0))

        context = use_cuda(torch.zeros([batch_size, self.units]))
        # train
        graph_mask = np.zeros([batch_size, graph_embed.shape[1]])
        for i in range(batch_size):
            graph_mask[i][0: subgraph_len[i]] = 1
        graph_mask = use_cuda(torch.from_numpy(graph_mask).type('torch.LongTensor'))

        ce_attention_keys = self.graph_attn_linear(graph_embed)
        ce_attention_values = graph_embed
        for t in range(decoder_len):
            decoder_input_t = torch.cat((decoder_input[:,t,:], context), 1).unsqueeze(1)
            decoder_output_t, decoder_state = self.decoder(decoder_input_t, decoder_state)

            context, ce_alignments_t = self.attention(c_attention_keys, c_attention_values, ce_attention_keys, ce_attention_values,
                decoder_output_t.squeeze(1), graph_mask)
            decoder_output_t = context.unsqueeze(1)
            decoder_output = torch.cat((decoder_output, decoder_output_t), 1)
            ce_alignments = torch.cat((ce_alignments, ce_alignments_t.unsqueeze(1)), 1)

        if self.is_inference:   # test
            word_index = use_cuda(torch.empty(0).type('torch.LongTensor'))
            decoder_input_t = self.word_embedding(use_cuda(torch.ones([batch_size]).type('torch.LongTensor')))
            context = use_cuda(torch.zeros([batch_size, self.units]))
            decoder_state = text_encoder_state
            selector = use_cuda(torch.empty(0).type('torch.LongTensor'))

            for t in range(decoder_len):
                decoder_input_t = torch.cat((decoder_input_t, context), 1).unsqueeze(1)
                decoder_output_t, decoder_state = self.decoder(decoder_input_t, decoder_state)

                context, ce_alignments_t = self.attention(c_attention_keys, c_attention_values, ce_attention_keys, ce_attention_values,
                                                              decoder_output_t.squeeze(1), graph_mask)
                decoder_output_t = context.unsqueeze(1)
                decoder_input_t, word_index_t, selector_t = self.inference(decoder_output_t, ce_alignments_t, word2id, subgraph, id2entity)
                word_index = torch.cat((word_index, word_index_t.unsqueeze(1)), 1)
                selector = torch.cat((selector, selector_t.unsqueeze(1)), 1)

        ### Total Loss
        decoder_mask = np.zeros([batch_size, decoder_len])
        for i in range(batch_size):
            decoder_mask[i][0:responses_length[i]] = 1
        decoder_mask = use_cuda(torch.from_numpy(decoder_mask).type('torch.LongTensor'))

        graph_entities = use_cuda(torch.zeros(batch_size, decoder_len, max_graph_size))
        # if not self.is_inference:
        for b in range(batch_size):
            for d in range(decoder_len):
                if match_entity[b][d] != -1:
                    graph_entities[b][d][match_entity[b][d]] = 1

        use_entities_local = torch.sum(graph_entities, [2])

        decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_entity, word_neg_num, local_neg_num = \
            self.total_loss(decoder_output, responses_target, decoder_mask, ce_alignments, use_entities_local, graph_entities)

        if self.is_inference:
            return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_entity, word_neg_num, local_neg_num, \
                   recall, precision, total_graph_size, word_index.detach().cpu().numpy().tolist()
        return decoder_loss, retrieval_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_entity, word_neg_num, local_neg_num

    def inference(self, decoder_output_t, ce_alignments_t, word2id, local_entity, id2entity):
        '''
        decoder_output_t: [batch_size, 1, self.units]
        ce_alignments_t: [batch_size, local_entity_len]
        '''
        batch_size = decoder_output_t.shape[0]

        logits = self.logits_linear(decoder_output_t.squeeze(1)) # (bs, num_symbols)

        selector = self.softmax_d1(self.selector_linear(decoder_output_t.squeeze(1)))   # (bs, 2)

        # get the probablities and indices of choosen tokens
        (word_prob, word_t) = torch.max(selector[:, 0].unsqueeze(1) * self.softmax_d1(logits), dim=1)
        (entity_prob, entity_index_t) = torch.max(selector[:, 1].unsqueeze(1) * ce_alignments_t, dim=1)

        selector[:,0] = selector[:,0] * word_prob
        selector[:,1] = selector[:,1] * entity_prob
        # selector[:, 0] = word_prob
        # selector[:, 1] = entity_prob
        selector = torch.argmax(selector, dim=1)

        entity_index_t = entity_index_t.cpu().numpy().tolist()
        word_t = word_t.cpu().numpy().tolist()

        word_local_entity_t = []
        word_only_two_entity_t = []
        word_index_final_t = []
        for i in range(batch_size):
            if selector[i] == 0:
                word_index_final_t.append(word_t[i])
            elif selector[i] == 1:
                local_entity_index_t = int(local_entity[i][entity_index_t[i]])
                local_entity_text = id2entity[local_entity_index_t]
                if local_entity_text not in word2id:
                    local_entity_text = '_UNK'
                word_index_final_t.append(word2id[local_entity_text])

        word_index_final_t = use_cuda(torch.LongTensor(word_index_final_t))
        decoder_input_t = self.word_embedding(word_index_final_t)

        return decoder_input_t, word_index_final_t, selector

    def total_loss(self, decoder_output, responses_target, decoder_mask, ce_alignments, use_entities_local, entity_targets_local):
        batch_size = decoder_output.shape[0]
        decoder_len = responses_target.shape[1]

        local_masks = use_cuda(decoder_mask.reshape([-1]).type("torch.FloatTensor"))
        local_masks_word = use_cuda((1 - use_entities_local).reshape([-1]).type("torch.FloatTensor")) * local_masks
        local_masks_local = use_cuda(use_entities_local.reshape([-1]).type("torch.FloatTensor"))
        logits = self.logits_linear(decoder_output) # (bs, decoder_len, num_symbols)

        word_prob = torch.gather(self.softmax_d2(logits), 2, responses_target.unsqueeze(2)).squeeze(2)  # (bs, decoder_len)

        selector_word, selector_local = torch.split(self.softmax_d2(self.selector_linear(decoder_output)), [1, 1], 2) # (bs, decoder_len, 1)
        selector_word = selector_word.squeeze(2)
        selector_local = selector_local.squeeze(2)
        entity_prob_local = torch.sum(ce_alignments * entity_targets_local, [2])

        ppx_prob = word_prob * (1 - use_entities_local) + entity_prob_local * use_entities_local
        ppx_word = word_prob * (1 - use_entities_local)
        ppx_local = entity_prob_local * use_entities_local

        final_prob = word_prob * selector_word * (1 - use_entities_local) + entity_prob_local * selector_local * use_entities_local

        final_loss = torch.sum(-torch.log(1e-12 + final_prob).reshape([-1]) * local_masks)

        sentence_ppx = torch.sum((-torch.log(1e-12 + ppx_prob).reshape([-1]) * local_masks).reshape([batch_size, -1]), 1)
        sentence_ppx_word = torch.sum((-torch.log(1e-12 + ppx_word).reshape([-1]) * local_masks_word).reshape([batch_size, -1]), 1)
        sentence_ppx_local = torch.sum((-torch.log(1e-12 + ppx_local).reshape([-1]) * local_masks_local).reshape([batch_size, -1]), 1)

        selector_loss = torch.sum(-torch.log(1e-12 + selector_local * use_entities_local + selector_word * (1 - use_entities_local)).reshape([-1]) * local_masks)

        loss = final_loss + selector_loss
        total_size = torch.sum(local_masks)
        total_size += 1e-12

        sum_word = torch.sum(use_cuda(((1 - use_entities_local) * use_cuda(decoder_mask.type("torch.FloatTensor"))).type("torch.FloatTensor")), 1)
        sum_local = torch.sum(use_cuda(use_entities_local.type("torch.FloatTensor")), 1)

        word_neg_mask = use_cuda((sum_word == 0).type("torch.FloatTensor"))
        local_neg_mask = use_cuda((sum_local == 0).type("torch.FloatTensor"))

        word_neg_num = torch.sum(word_neg_mask)
        local_neg_num = torch.sum(local_neg_mask)

        sum_word = sum_word + word_neg_mask
        sum_local = sum_local + local_neg_mask

        return loss/total_size, sentence_ppx/torch.sum(use_cuda(decoder_mask.type("torch.FloatTensor")), 1), \
               sentence_ppx_word/sum_word, sentence_ppx_local/sum_local, word_neg_num, local_neg_num

    def attention(self, c_attention_keys, c_attention_values, ce_attention_keys, ce_attention_values, decoder_state, graph_mask):
        batch_size = c_attention_keys.shape[0]

        c_query = decoder_state.reshape([-1, 1, self.units])
        ce_query = decoder_state.reshape([-1, 1, self.units])

        c_scores = torch.sum(c_attention_keys * c_query, 2)
        ce_scores = torch.sum(ce_attention_keys * ce_query, 2)

        c_alignments = self.softmax_d1(c_scores)
        ce_alignments = self.softmax_d1(ce_scores)
        ce_alignments = ce_alignments * use_cuda(graph_mask.type("torch.FloatTensor"))

        c_context = torch.sum(c_alignments.unsqueeze(2) * c_attention_values, 1)
        ce_context = torch.sum(ce_alignments.unsqueeze(2) * ce_attention_values, 1)

        context = self.context_linear(torch.cat((decoder_state, c_context, ce_context), 1))
        return context, ce_alignments

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))

    def get_graph_input(self, input, nodes, edges, relation, batch_size, max_num, max_len, max_candidate_size):
        graph_input = use_cuda(torch.empty(0))
        assert len(input) == batch_size
        graph_list = []
        for b in range(batch_size):
            for p in range(len(nodes[b])):
                batch_node = nodes[b][p]
                batch_edge = edges[b][p]
                batch_relation = relation[b][p]
                node_index = []     # global index of nodes in every graph
                edge_index = [[], []]      # local index of edges in every graph
                for l in range(len(batch_node)):
                    previous_node_embed = self.entity_embedding(use_cuda(torch.LongTensor(node_index)))
                    new_node_embed = self.entity_embedding(use_cuda(torch.LongTensor(batch_node[l])))
                    if l > 0:
                        relation_index = batch_relation[l - 1]
                        relation_embed = self.entity_embedding(use_cuda(torch.LongTensor(relation_index)))
                        new_node_embed = self.relation_linear(torch.cat([new_node_embed, relation_embed], dim=1))
                    node_embed = torch.cat([previous_node_embed, new_node_embed], dim=0)
                    node_index += batch_node[l]
                    edge_index[0] += batch_edge[l][0]
                    edge_index[1] += batch_edge[l][1]
                    graph = dgl.DGLGraph()
                    graph.add_nodes(len(node_index))
                    graph.add_edges(edge_index[0], edge_index[1])
                    graph.ndata['h'] = node_embed
                    graph_list.append(graph)
        batched_graph = dgl.batch(graph_list)
        graph_output = self.GAT(batched_graph, batched_graph.ndata['h'])
        batched_graph.ndata['h'] = graph_output
        graph_list = dgl.unbatch(batched_graph)
        index = 0
        for b in input:
            batch_input = use_cuda(torch.zeros(0))
            for p in b:
                path_input = use_cuda(torch.empty(0))
                for l in p:
                    node_feature = graph_list[index].ndata['h'][l].transpose(0, 1)   # (1, N, trans_units)
                    N = node_feature.shape[1]
                    padded_feature = torch.cat([node_feature, use_cuda(torch.zeros([1, max_candidate_size - N, self.trans_units]))], dim=1)
                    path_input = torch.cat([path_input, padded_feature], dim=0)
                    index += 1
                L = path_input.shape[0]
                path_padding = use_cuda(torch.zeros([max_len - L, max_candidate_size, self.trans_units]))
                path_input = torch.cat([path_input, path_padding], dim=0).unsqueeze(0)  # (1, max_len, max_candidate, trans_units)
                batch_input = torch.cat([batch_input, path_input], dim=0)
            M = batch_input.shape[0]
            batch_padding = use_cuda(torch.zeros([max_num - M, max_len, max_candidate_size, self.trans_units]))
            batch_input = torch.cat([batch_input, batch_padding], dim=0).unsqueeze(0)   # (1, max_num, max_len, max_candidate, trans_units)
            graph_input = torch.cat([graph_input, batch_input], dim=0)
        assert index == len(graph_list)
        return graph_input  # （bs, max_num, max_len, max_candidate, trans_units)

    def construct_graph(self, subgraphs, edges):
        graph_list = []
        for i in range(len(subgraphs)):
            g2l = dict()
            graph = dgl.DGLGraph()
            graph.add_nodes(len(subgraphs[i]))
            for index in range(len(subgraphs[i])):
                g2l[subgraphs[i][index]] = index
            edge_heads, edge_tails = [g2l[u] for u in edges[i][0]], [g2l[u] for u in edges[i][1]]
            graph.add_edges(edge_heads, edge_tails)
            node_embed = self.entity_embedding(use_cuda(torch.LongTensor(subgraphs[i])))
            graph.ndata['h'] = node_embed
            graph_list.append(graph)
        return graph_list

    def gnn(self, graph, input):
        gat_output = self.GAT(graph, input)
        graph.ndata['h'] = gat_output
        graph_list = dgl.unbatch(graph)

        gat_output = []
        for i in range(len(graph_list)):
            node_features = graph_list[i].ndata['h'].squeeze(1)
            gat_output.append(node_features)
        return nn_utils.rnn.pad_sequence(gat_output, batch_first=True, padding_value=0)
