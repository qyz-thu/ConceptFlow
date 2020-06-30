#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import utils as nn_utils

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var


class ConceptFlow(nn.Module):
    def __init__(self, config, word_embed, entity_embed, adj_table):
        super(ConceptFlow, self).__init__()
        self.is_inference = False
        # Encoder
        self.fact_scale = config.fact_scale
        self.pagerank_lambda = config.pagerank_lambda
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

        # self.lstm_drop = nn.Dropout(p=config.lstm_dropout)
        # self.linear_drop = nn.Dropout(p=config.linear_dropout)

        # self.node_encoder = nn.LSTM(input_size=self.embed_units, hidden_size=self.trans_units, batch_first=True, bidirectional=False)

        self.softmax_d1 = nn.Softmax(dim=1)
        self.softmax_d2 = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

        self.text_encoder = nn.GRU(input_size=self.embed_units, hidden_size=self.units, num_layers=self.layers, batch_first=True)
        self.decoder = nn.GRU(input_size=self.units + self.embed_units, hidden_size=self.units, num_layers=self.layers, batch_first=True)

        self.attn_c_linear = nn.Linear(in_features=self.units, out_features=self.units, bias=False)
        # self.attn_ce_linear = nn.Linear(in_features=self.trans_units, out_features=2 * self.units, bias=False)
        # self.attn_co_linear = nn.Linear(in_features=2 * self.trans_units, out_features=2 * self.units, bias=False)
        # self.attn_ct_linear = nn.Linear(in_features=self.trans_units, out_features=2 * self.units, bias=False)

        self.context_linear = nn.Linear(in_features=2 * self.units + self.trans_units, out_features=self.units, bias=False)

        # matrix for graph attention
        self.graph_attn_linear = nn.Linear(in_features=self.trans_units, out_features=self.units, bias=False)

        # create linear functions
        # self.k = 2 + 1
        # for i in range(self.gnn_layers):
        #     self.add_module('q2e_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
        #     self.add_module('d2e_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
        #     self.add_module('e2q_linear' + str(i), nn.Linear(in_features=self.k * self.trans_units, out_features=self.trans_units))
        #     self.add_module('e2d_linear' + str(i), nn.Linear(in_features=self.k * self.trans_units, out_features=self.trans_units))
        #     self.add_module('e2e_linear' + str(i), nn.Linear(in_features=self.k * self.trans_units, out_features=self.trans_units))
        #
        #     #use kb
        #     self.add_module('kb_head_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
        #     self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
        #     self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
        #
        # # one_two_triple
        # self.head_tail_linear = nn.Linear(in_features=self.trans_units * 2, out_features=self.trans_units)
        # self.one_two_entity_linear = nn.Linear(in_features=self.trans_units, out_features=self.trans_units)

        # Loss
        self.logits_linear = nn.Linear(in_features=self.units, out_features=self.symbols)
        self.selector_linear = nn.Linear(in_features=self.units, out_features=2)

    def forward(self, batch_data):
        query_text = batch_data['post_text']    # todo: rename query_text to post_text
        response_text = batch_data['response_text']
        responses_length = batch_data['responses_length']
        post_ent = batch_data['post_ent']
        response_ent = batch_data['response_ent']
        subgraph = batch_data['subgraph']
        subgraph_len = batch_data['subgraph_size']
        match_entity = batch_data['match_entity']
        # q2e_adj_mat = batch_data['q2e_adj_mat']
        # kb_adj_mat = batch_data['kb_adj_mat']
        # kb_fact_rel = batch_data['kb_fact_rel']
        # only_two_entity = batch_data['only_two_entity']
        # match_entity_only_two = batch_data['match_entity_only_two']
        # one_two_triples_id = batch_data['one_two_triples_id']
        # local_entity_length = batch_data['local_entity_length']
        # only_two_entity_length = batch_data['only_two_entity_length']

        if self.is_inference == True:
            word2id = batch_data['word2id']
            entity2id = batch_data['entity2id']
            id2entity = dict()
            for key in entity2id.keys():
                id2entity[entity2id[key]] = key
        else:
            id2entity = None

        if not self.is_inference:
            max_graph_size = subgraph.shape[1]
            subgraph = use_cuda(Variable(torch.from_numpy(subgraph).type('torch.LongTensor'), requires_grad=False))  # todo: remove Variable?
            match_entity = use_cuda(Variable(torch.from_numpy(match_entity).type('torch.LongTensor'), requires_grad=False))
            subgraph_emb = self.entity_linear(self.entity_embedding(subgraph))

        batch_size = query_text.shape[0]
        # _, max_only_two_entity = only_two_entity.shape
        # _, one_two_triple_num, one_two_triple_len, _ = one_two_triples_id.shape
        # _, max_fact = kb_fact_rel.shape

        # numpy to tensor
        # local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor'), requires_grad=False))
        # local_entity_mask = use_cuda((local_entity != 0).type('torch.FloatTensor'))
        # kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor'), requires_grad=False))
        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        response_text = use_cuda(Variable(torch.from_numpy(response_text).type('torch.LongTensor'), requires_grad=False))
        responses_length = use_cuda(Variable(torch.Tensor(responses_length).type('torch.LongTensor'), requires_grad=False))
        query_mask = use_cuda((query_text != 0).type('torch.FloatTensor'))
        # only_two_entity = use_cuda(Variable(torch.from_numpy(only_two_entity).type('torch.LongTensor'), requires_grad=False))
        # match_entity_only_two = use_cuda(Variable(torch.from_numpy(match_entity_only_two).type('torch.LongTensor'), requires_grad=False))
        # one_two_triples_id = use_cuda(Variable(torch.from_numpy(one_two_triples_id).type('torch.LongTensor'), requires_grad=False))

        # normalized adj matrix
        # pagerank_f = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=True))
        # q2e_adj_mat = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=False))
        # assert pagerank_f.requires_grad == True

        decoder_len = response_text.shape[1]
        responses_target = response_text
        responses_id = torch.cat((use_cuda(torch.ones([batch_size, 1]).type('torch.LongTensor')),torch.split(response_text, [decoder_len - 1, 1], 1)[0]), 1)
        # encode query
        # query_word_emb = self.word_embedding(query_text)
        # query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.trans_units))
        # query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)
        # query_rel_emb = query_node_emb

        # build kb_adj_matrix from sparse matrix
        # (e2f_batch, e2f_f, e2f_e, e2f_val), (f2e_batch, f2e_e, f2e_f, f2e_val) = kb_adj_mat
        # entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
        # entity2fact_val = torch.FloatTensor(e2f_val)
        # entity2fact_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size([batch_size, max_fact, max_local_entity])))
        # fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
        # fact2entity_val = torch.FloatTensor(f2e_val)
        # fact2entity_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val, torch.Size([batch_size, max_local_entity, max_fact])))
        #
        # local_fact_emb = self.entity_embedding(kb_fact_rel)
        # local_fact_emb = self.entity_linear(local_fact_emb)

        # attention fact2question
        # div = float(np.sqrt(self.trans_units))
        # fact2query_sim = torch.bmm(query_hidden_emb, local_fact_emb.transpose(1, 2)) / div
        # fact2query_sim = self.softmax_d1(fact2query_sim + (1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)
        #
        # fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2), dim=1)
        #
        # W = torch.sum(fact2query_att * local_fact_emb, dim=2) / div
        # W_max = torch.max(W, dim=1, keepdim=True)[0]
        # W_tilde = torch.exp(W - W_max)
        # e2f_softmax = self.sparse_bmm(entity2fact_mat.transpose(1, 2), W_tilde.unsqueeze(dim=2)).squeeze(dim=2)
        # e2f_softmax = torch.clamp(e2f_softmax, min=VERY_SMALL_NUMBER)
        # e2f_out_dim = use_cuda(Variable(torch.sum(entity2fact_mat.to_dense(), dim=1), requires_grad=False))

        # subgraph embedding
        # subgraph_emb = self.entity_embedding(subgraph)

        # label propagation on entities
        # for i in range(self.gnn_layers):
        #     # get linear transformation functions for each layer
        #     q2e_linear = getattr(self, 'q2e_linear' + str(i))
        #     d2e_linear = getattr(self, 'd2e_linear' + str(i))
        #     e2q_linear = getattr(self, 'e2q_linear' + str(i))
        #     e2d_linear = getattr(self, 'e2d_linear' + str(i))
        #     e2e_linear = getattr(self, 'e2e_linear' + str(i))
        #
        #     kb_self_linear = getattr(self, 'kb_self_linear' + str(i))
        #     kb_head_linear = getattr(self, 'kb_head_linear' + str(i))
        #     kb_tail_linear = getattr(self, 'kb_tail_linear' + str(i))
        #
        #     # start propagation
        #     next_local_entity_emb = local_entity_emb
        #
        #     # STEP 1: propagate from question, documents, and facts to entities
        #     # question -> entity
        #     q2e_emb = q2e_linear(self.linear_drop(query_node_emb)).expand(batch_size, max_local_entity, self.trans_units)
        #     next_local_entity_emb = torch.cat((next_local_entity_emb, q2e_emb), dim=2)
        #
        #     # fact -> entity
        #     e2f_emb = self.relu(kb_self_linear(local_fact_emb) + self.sparse_bmm(entity2fact_mat, kb_head_linear(self.linear_drop(local_entity_emb))))
        #     e2f_softmax_normalized = W_tilde.unsqueeze(dim=2) * self.sparse_bmm(entity2fact_mat, (pagerank_f / e2f_softmax).unsqueeze(dim=2))
        #     e2f_emb = e2f_emb * e2f_softmax_normalized
        #     f2e_emb = self.relu(kb_self_linear(local_entity_emb) + self.sparse_bmm(fact2entity_mat, kb_tail_linear(self.linear_drop(e2f_emb))))
        #
        #     pagerank_f = self.pagerank_lambda * self.sparse_bmm(fact2entity_mat, e2f_softmax_normalized).squeeze(dim=2) + (1 - self.pagerank_lambda) * pagerank_f
        #
        #     # STEP 2: combine embeddings from fact
        #     next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale * f2e_emb), dim=2)
        #
        #     # STEP 3: propagate from entities to update question, documents, and facts
        #     # entity -> query
        #     query_node_emb = torch.bmm(pagerank_f.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))
        #     # update entity
        #     local_entity_emb = self.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        text_encoder_input = self.word_embedding(query_text)
        text_encoder_output, text_encoder_state = self.text_encoder(text_encoder_input, use_cuda(Variable(torch.zeros(self.layers, batch_size, self.units))))

        # # encode_one_two_triple
        # one_two_triples_embedding = self.entity_embedding(one_two_triples_id).reshape([batch_size, one_two_triple_num, -1, 3 * self.trans_units])
        #
        # # Grow graph embedding
        # head, relation, tail = torch.split(one_two_triples_embedding, [self.trans_units] * 3, 3)
        # head_tail = torch.cat((head, tail), 3)
        # head_tail_transformed = torch.tanh(self.head_tail_linear(head_tail))
        #
        # relation_transformed = self.one_two_entity_linear(relation)
        #
        # e_weight = torch.sum(relation_transformed * head_tail_transformed, 3)
        # alpha_weight = self.softmax_d2(e_weight)
        #
        # one_two_embed = torch.sum(alpha_weight.unsqueeze(3) * head_tail, 2)


        # decoder
        decoder_input = self.word_embedding(responses_id)

        # attention
        c_attention_keys = self.attn_c_linear(text_encoder_output)
        c_attention_values = text_encoder_output

        # ce_attention_keys, ce_attention_values = torch.split(self.attn_ce_linear(local_entity_emb), [self.units, self.units], 2)
        #
        # co_attention_keys, co_attention_values = torch.split(self.attn_co_linear(one_two_embed), [self.units, self.units], 2)
        #
        # only_two_entity_embed = self.entity_linear(self.entity_embedding(only_two_entity))
        # ct_attention_keys, ct_attention_values = torch.split(self.attn_ct_linear(only_two_entity_embed), [self.units, self.units], 2)

        # ce_attention_keys, ce_attention_values, co_attention_keys, co_attention_values, ct_attention_keys = None, None, None, None, None

        decoder_state = text_encoder_state
        decoder_output = use_cuda(torch.empty(0))
        ce_alignments = use_cuda(torch.empty(0))

        # only_two_entity_mask = np.zeros([batch_size, only_two_entity.shape[1]])
        # for i in range(batch_size):
        #     only_two_entity_mask[i][0:only_two_entity_length[i]] = 1
        # only_two_entity_mask = use_cuda(torch.from_numpy(only_two_entity_mask).type('torch.LongTensor'))

        context = use_cuda(torch.zeros([batch_size, self.units]))
        if not self.is_inference:   # train
            graph_mask = np.zeros([batch_size, subgraph.shape[1]])
            for i in range(batch_size):
                graph_mask[i][0: subgraph_len[i]] = 1
            graph_mask = use_cuda(torch.from_numpy(graph_mask).type('torch.LongTensor'))

            ce_attention_keys = self.graph_attn_linear(subgraph_emb)
            ce_attention_values = subgraph_emb
            for t in range(decoder_len):
                decoder_input_t = torch.cat((decoder_input[:,t,:], context), 1).unsqueeze(1)
                decoder_output_t, decoder_state = self.decoder(decoder_input_t, decoder_state)

                context, ce_alignments_t = self.attention(c_attention_keys, c_attention_values, ce_attention_keys, ce_attention_values,
                    decoder_output_t.squeeze(1), graph_mask)
                decoder_output_t = context.unsqueeze(1)
                decoder_output = torch.cat((decoder_output, decoder_output_t), 1)
                ce_alignments = torch.cat((ce_alignments, ce_alignments_t.unsqueeze(1)), 1)

        else:   # test
            word_index = use_cuda(torch.empty(0).type('torch.LongTensor'))
            decoder_input_t = self.word_embedding(use_cuda(torch.ones([batch_size]).type('torch.LongTensor')))
            context = use_cuda(torch.zeros([batch_size, self.units]))
            # decoder_state = text_encoder_state
            selector = use_cuda(torch.empty(0).type('torch.LongTensor'))

            current_graph = post_ent
            outer = [[[n] for n in g] for g in post_ent]

            for t in range(decoder_len):
                decoder_input_t = torch.cat((decoder_input_t, context), 1).unsqueeze(1)
                decoder_output_t, decoder_state = self.decoder(decoder_input_t, decoder_state)
                # context, ce_alignments_t, co_alignments_t, ct_alignments_t = self.attention(c_attention_keys, c_attention_values, \
                #     ce_attention_keys, ce_attention_values, co_attention_keys, co_attention_values, ct_attention_keys, \
                #     decoder_output_t.squeeze(1), local_entity_mask, only_two_entity_mask)
                if t < self.max_hop:
                    ce_attention_keys, ce_attention_values, graph_mask = self.beam_search(decoder_output_t.squeeze(1), current_graph, outer)
                context, ce_alignments_t = self.attention(c_attention_keys, c_attention_values, ce_attention_keys, ce_attention_values,
                                                              decoder_output_t.squeeze(1), graph_mask)

                decoder_output_t = context.unsqueeze(1)

                decoder_input_t, word_index_t, selector_t = self.inference(decoder_output_t, ce_alignments_t, word2id, current_graph, id2entity)
                word_index = torch.cat((word_index, word_index_t.unsqueeze(1)), 1)
                selector = torch.cat((selector, selector_t.unsqueeze(1)), 1)
                if t > 0:
                    padding_len = ce_alignments_t.shape[1] - ce_alignments.shape[2]
                    ce_alignments = torch.cat((ce_alignments, use_cuda(torch.zeros([batch_size, t, padding_len]))), 2)
                decoder_output = torch.cat((decoder_output, decoder_output_t), 1)
                ce_alignments = torch.cat((ce_alignments, ce_alignments_t.unsqueeze(1)), 1)

            max_graph_size = max([len(c) for c in current_graph])

        ### Total Loss
        decoder_mask = np.zeros([batch_size, decoder_len])
        for i in range(batch_size):
            decoder_mask[i][0:responses_length[i]] = 1
        decoder_mask = use_cuda(torch.from_numpy(decoder_mask).type('torch.LongTensor'))

        graph_entities = use_cuda(torch.zeros(batch_size, decoder_len, max_graph_size))
        if not self.is_inference:
            for b in range(batch_size):
                for d in range(decoder_len):
                    if match_entity[b][d] != -1:
                        graph_entities[b][d][match_entity[b][d]] = 1
        else:
            response_ent_num = 0
            found_num = 0
            for b in range(batch_size):
                g2l = dict()
                for i in range(len(current_graph[b])):
                    if current_graph[b][i] > 0:
                        g2l[current_graph[b][i]] = i
                for d in range(decoder_len):
                    if response_ent[b][d] == -1:
                        continue
                    response_ent_num += 1
                    if response_ent[b][d] in g2l:
                        graph_entities[b][d][g2l[response_ent[b][d]]] = 1
                        found_num += 1

        use_entities_local = torch.sum(graph_entities, [2])
        #
        # one_hot_entities_only_two = use_cuda(torch.zeros(batch_size, decoder_len, max_only_two_entity))

        # decoder_loss, ppx_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, \
        #     word_neg_num, local_neg_num, only_two_neg_num \
        decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_entity, word_neg_num, local_neg_num = \
            self.total_loss(decoder_output, responses_target, decoder_mask, ce_alignments, use_entities_local, graph_entities)

        # if self.is_inference == True:
            # return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, \
            #     word_index.cpu().numpy().tolist(), word_neg_num, local_neg_num, only_two_neg_num, selector.cpu().numpy().tolist()
        # return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, only_two_neg_num
        if self.is_inference:
            return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_entity, word_neg_num, local_neg_num, found_num / response_ent_num
        return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_entity, word_neg_num, local_neg_num

    def inference(self, decoder_output_t, ce_alignments_t, word2id, local_entity, id2entity):
        '''
        decoder_output_t: [batch_size, 1, self.units]
        ce_alignments_t: [batch_size, local_entity_len]
        ct_alignments_t: [batch_size, only_two_entity_len]
        '''
        batch_size = decoder_output_t.shape[0]

        logits = self.logits_linear(decoder_output_t.squeeze(1)) # (bs, num_symbols)

        selector = self.softmax_d1(self.selector_linear(decoder_output_t.squeeze(1)))   # (bs, 2)

        # get the probablities and indices of choosen tokens
        (word_prob, word_t) = torch.max(selector[:, 0].unsqueeze(1) * self.softmax_d1(logits), dim=1)
        (entity_prob, entity_index_t) = torch.max(selector[:, 1].unsqueeze(1) * ce_alignments_t, dim=1)
        # (only_two_entity_prob, only_two_entity_l_index_t) = torch.max(selector[:,2].unsqueeze(1) * ct_alignments_t, dim=1)

        selector[:,0] = selector[:,0] * word_prob
        selector[:,1] = selector[:,1] * entity_prob
        # selector[:,2] = selector[:,2] * only_two_entity_prob
        selector = torch.argmax(selector, dim=1)

        entity_index_t = entity_index_t.cpu().numpy().tolist()
        # only_two_entity_l_index_t = only_two_entity_l_index_t.cpu().numpy().tolist()
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
            # if selector[i] == 2:
            #     only_two_entity_index_t = int(only_two_entity[i][only_two_entity_l_index_t[i]])
            #     only_two_entity_text = id2entity[only_two_entity_index_t]
            #     if only_two_entity_text not in word2id:
            #         only_two_entity_text = '_UNK'
            #     word_index_final_t.append(word2id[only_two_entity_text])

        word_index_final_t = use_cuda(torch.LongTensor(word_index_final_t))
        decoder_input_t = self.word_embedding(word_index_final_t)

        return decoder_input_t, word_index_final_t, selector

    def total_loss(self, decoder_output, responses_target, decoder_mask, ce_alignments, use_entities_local, entity_targets_local):
        batch_size = decoder_output.shape[0]
        decoder_len = responses_target.shape[1]

        local_masks = use_cuda(decoder_mask.reshape([-1]).type("torch.FloatTensor"))
        local_masks_word = use_cuda((1 - use_entities_local).reshape([-1]).type("torch.FloatTensor")) * local_masks
        local_masks_local = use_cuda(use_entities_local.reshape([-1]).type("torch.FloatTensor"))
        # local_masks_only_two = use_cuda(use_entities_only_two.reshape([-1]).type("torch.FloatTensor"))
        logits = self.logits_linear(decoder_output) # (bs, decoder_len, num_symbols)

        word_prob = torch.gather(self.softmax_d2(logits), 2, responses_target.unsqueeze(2)).squeeze(2)  # (bs, decoder_len)

        selector_word, selector_local = torch.split(self.softmax_d2(self.selector_linear(decoder_output)), [1, 1], 2) # (bs, decoder_len, 1)
        selector_word = selector_word.squeeze(2)
        selector_local = selector_local.squeeze(2)
        # selector_only_two = selector_only_two.squeeze(2)

        entity_prob_local = torch.sum(ce_alignments * entity_targets_local, [2])
        # entity_prob_only_two = torch.sum(ct_alignments * entity_targets_only_two, [2])

        ppx_prob = word_prob * (1 - use_entities_local) + entity_prob_local * use_entities_local
        ppx_word = word_prob * (1 - use_entities_local)
        ppx_local = entity_prob_local * use_entities_local
        # ppx_only_two = entity_prob_only_two * use_entities_only_two

        final_prob = word_prob * selector_word * (1 - use_entities_local) + entity_prob_local * selector_local * use_entities_local

        final_loss = torch.sum(-torch.log(1e-12 + final_prob).reshape([-1]) * local_masks)

        sentence_ppx = torch.sum((-torch.log(1e-12 + ppx_prob).reshape([-1]) * local_masks).reshape([batch_size, -1]), 1)
        sentence_ppx_word = torch.sum((-torch.log(1e-12 + ppx_word).reshape([-1]) * local_masks_word).reshape([batch_size, -1]), 1)
        sentence_ppx_local = torch.sum((-torch.log(1e-12 + ppx_local).reshape([-1]) * local_masks_local).reshape([batch_size, -1]), 1)
        # sentence_ppx_only_two = torch.sum((- torch.log(1e-12 + ppx_only_two).reshape([-1]) * local_masks_only_two).reshape([batch_size, -1]), 1)

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

        # return loss / total_size, 0, sentence_ppx / torch.sum(use_cuda(decoder_mask.type("torch.FloatTensor")), 1), \
        #     sentence_ppx_word / sum_word, sentence_ppx_local / sum_local, sentence_ppx_only_two / sum_only_two, word_neg_num, \
        #     local_neg_num, only_two_neg_num
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

    def beam_search(self, decoder_state, current_graph, outer):
        batch_size = decoder_state.shape[0]
        paths = []
        for i, d in enumerate(outer):
            path = dict()
            graph = set(current_graph[i])
            for o in d:
                out = o[-1]
                for node in self.adj_table[out]:
                    if node in graph:
                        continue
                    if node in path:
                        path[node].append(o + [node])
                    else:
                        path[node] = [o + [node]]
            paths.append(path)
        # choose the path with largest attention score for every nodes in outmost hop
        paths_embed = []
        nodes_list = []
        for d in paths:
            node_list = []
            # path_embed = use_cuda(torch.empty(0))
            path_embed = []
            for node in d:
                path = d[node]
                for i, p in enumerate(path):
                    node_list.append([node, p])
                    p_embed = torch.sum(self.entity_embedding(use_cuda(torch.Tensor(p).long())), 0).detach()
                    p_embed = p_embed.cpu().numpy().tolist()
                    # p_embed.unsqueeze_(dim=0)
                    # p_embed.unsqueeze_(dim=0)   # (1, 1, trans_unit)
                    # torch.cat((path_embed, p_embed), dim=1)     # (1, x, trans_unit)
                    path_embed.append(p_embed)
            paths_embed.append(path_embed)
            nodes_list.append(node_list)
        max_path_num = max([len(pp) for pp in paths_embed])
        padded_paths_embed = \
            [pp + [[0 for j in range(100)] for i in range((max_path_num - len(pp)))] for pp in paths_embed]
        paths_embed = use_cuda(torch.Tensor(padded_paths_embed))
        c_query = decoder_state.unsqueeze(1)
        scores = torch.sum(c_query * self.graph_attn_linear(paths_embed), dim=2)    # (bs, max_path_num)
        scores = scores.detach().cpu().numpy().tolist()
        for i in range(batch_size):
            node_list = nodes_list[i]
            candidates = dict()
            for j in range(len(node_list)):
                node = node_list[j][0]
                path = node_list[j][1]
                if node in candidates and scores[i][j] > candidates[node][0]:
                    candidates[node][0] = scores[i][j]
                else:
                    candidates[node] = [scores[i][j], path]
            candidate = [candidates[c] for c in candidates]
            candidate.sort(key=lambda c: c[0], reverse=True)
            new_nodes = candidate[:self.bs_width]
            outer[i] = [n[1] for n in new_nodes]
            current_graph[i] += [n[-1] for n in outer[i]]
        max_graph_size = max([len(g) for g in current_graph])
        padded_graph = [g + [1 for i in range(max_graph_size - len(g))] for g in current_graph]
        graph_mask = [[1 for i in range(len(g))] + [0 for i in range(max_graph_size - len(g))] for g in current_graph]
        graph_mask = use_cuda(torch.Tensor(graph_mask).long())
        graph_values = self.entity_embedding(use_cuda(torch.Tensor(padded_graph).long()))
        graph_keys = self.graph_attn_linear(graph_values)

        return graph_keys, graph_values, graph_mask

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))

    def sparse_bmm(self, X, Y):
        """Batch multiply X and Y where X is sparse, Y is dense.
        Args:
            X: Sparse tensor of size BxMxN. Consists of two tensors,
                I:3xZ indices, and V:1xZ values.
            Y: Dense tensor of size BxNxK.
        Returns:
            batched-matmul(X, Y): BxMxK
        """
        class LeftMMFixed(torch.autograd.Function):
            """
            Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
            This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
            """

            def __init__(self):
                super(LeftMMFixed, self).__init__()
                self.sparse_weights = None

            def forward(self, sparse_weights, x):
                if self.sparse_weights is None:
                    self.sparse_weights = sparse_weights
                return torch.mm(self.sparse_weights, x)

            def backward(self, grad_output):
                sparse_weights = self.sparse_weights
                return None, torch.mm(sparse_weights.t(), grad_output)

        I = X._indices()
        V = X._values()
        B, M, N = X.size()
        _, _, K = Y.size()
        Z = I.size()[1]
        lookup = Y[I[0, :], I[2, :], :]
        X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
        S = use_cuda(Variable(torch.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
        prod_op = LeftMMFixed()
        prod = prod_op(S, lookup)
        return prod.view(B, M, K)


