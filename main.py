#coding:utf-8
import numpy as np
import json
from model import ConceptFlow, use_cuda
from preprocession import prepare_data, build_vocab, gen_batched_data
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
import yaml
import os
import pynvml
import sys
import time
warnings.filterwarnings('ignore')

csk_triples, csk_entities, kb_dict = [], [], []
dict_csk_entities, dict_csk_triples = {}, {}


class Config():
    def __init__(self, path):
        self.config_path = path
        self._get_config()

    def _get_config(self):
        with open(self.config_path, "r") as setting:
            config = yaml.load(setting)
        self.is_train = config['is_train']
        self.test_model_path = config['test_model_path']
        self.embed_units = config['embed_units']
        self.symbols = config['symbols']
        self.units = config['units']
        self.layers = config['layers']
        self.batch_size = config['batch_size']
        self.data_dir = config['data_dir']
        self.num_epoch = config['num_epoch']
        self.lr_rate = config['lr_rate']
        self.lstm_dropout = config['lstm_dropout']
        self.linear_dropout = config['linear_dropout']
        self.max_gradient_norm = config['max_gradient_norm']
        self.trans_units = config['trans_units']
        self.gnn_layers = config['gnn_layers']
        self.head_num = config['gat_head_num']
        self.fact_dropout = config['fact_dropout']
        self.fact_scale = config['fact_scale']
        self.result_dir_name = config['result_dir_name']
        self.log_dir = config['log_dir']
        self.tb_path = config['tensorboard_path']
        self.model_save_name = config['model_save_name']
        self.generated_text_name = config['generated_text_name']
        self.beam_search_width = config['beam_search_width']
        self.max_hop = config['max_hop']
        self.max_candidate_size = config['max_candidate_size']
        self.to_generate = config['to_generate']
        self.decode = config['decode']

    def list_all_member(self):
        for name, value in vars(self).items():
            print('%s = %s' % (name, value))
        

def run(model, data_train, config, word2id, entity2id, is_inference=False):
    batched_data = gen_batched_data(data_train, config, word2id, entity2id, is_inference)

    return model(batched_data)


def train(config, model, data_train, data_test, word2id, entity2id, model_optimizer, writer):
    count = 0
    start_time = time.time()
    eval_count = 0
    for epoch in range(config.num_epoch):
        print("epoch: ", epoch)
        with open(config.log_dir, 'a') as f:
            f.write("epoch %d\n" % (epoch + 1))
        sentence_ppx_loss = 0
        sentence_ppx_word_loss = 0
        sentence_ppx_local_loss = 0
        word_cut = use_cuda(torch.Tensor([0]))
        local_cut = use_cuda(torch.Tensor([0]))

        for iteration in range(len(data_train) // config.batch_size):
            count += 1
            data = data_train[(iteration * config.batch_size):(iteration * config.batch_size + config.batch_size)]
            if not config.decode:
                retrieval_loss = run(model, data, config, word2id, entity2id)
                model_optimizer.zero_grad()
                retrieval_loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), config.max_gradient_norm)
                model_optimizer.step()
                writer.add_scalar('retrieval_loss', retrieval_loss.data, count)
                if count % 50 == 0:
                    print("iteration: %d, loss: %.4f" % (iteration, retrieval_loss.data))
                    print("time used %.2f" % (time.time() - start_time))
                continue
            decoder_loss, retrieval_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, word_neg_num, entity_neg_num = \
                run(model, data, config, word2id, entity2id)
            sentence_ppx_loss += torch.sum(sentence_ppx).data
            sentence_ppx_word_loss += torch.sum(sentence_ppx_word).data
            sentence_ppx_local_loss += torch.sum(sentence_ppx_local).data
            word_cut += word_neg_num
            local_cut += entity_neg_num

            model_optimizer.zero_grad()
            loss = decoder_loss + retrieval_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.max_gradient_norm)
            model_optimizer.step()
            writer.add_scalar('train_loss/decoding_loss', decoder_loss.data, count)
            writer.add_scalar('train_loss/retrieval_loss', retrieval_loss.data, count)
            if count % 50 == 0:
                print ("iteration:", iteration, "decode loss:", decoder_loss.data, "retr loss:", retrieval_loss.data)
                print("time used: %ds" % (time.time() - start_time))
                with open(config.log_dir, 'a') as f:
                    f.write("iteration: %d decode loss: %.4f retr loss: %.4f total loss: %.4f\n" %
                            (iteration, decoder_loss.data, retrieval_loss.data, loss.data))
            if count % 20000 == 0:
                eval_count += 1
                evaluate(model, data_test, config, word2id, entity2id, eval_count, writer)

        if config.decode:
            ppl = np.exp(sentence_ppx_loss.cpu() / len(data_train))
            word_ppl = np.exp(sentence_ppx_word_loss.cpu() / (len(data_train) - int(word_cut)))
            entity_ppl = np.exp(sentence_ppx_local_loss.cpu() / (len(data_train) - int(local_cut)))
            writer.add_scalar('train_ppl/ppl', ppl, epoch + 1)
            writer.add_scalar('train_ppl/word_ppl', word_ppl, epoch + 1)
            writer.add_scalar('train_ppl/entitty_ppl', entity_ppl, epoch + 1)
            print ("perplexity for epoch", epoch + 1, ":", ppl, " ppx_word: ", word_ppl, " ppx_entity: ", entity_ppl)
            with open(config.log_dir, 'a') as f:
                f.write("perplexity for epoch%d: %.2f word ppl: %.2f entity ppl: %.2f\n" % (epoch + 1, ppl, word_ppl, entity_ppl))

            # torch.save(model.state_dict(), config.model_save_name + '_epoch_' + str(epoch + 1) + '.pkl')
            ppx, ppx_word, ppx_entity, recall = evaluate(model, data_test, config, word2id, entity2id, epoch + 1, writer)
            ppx_f = open(config.result_dir_name,'a')
            ppx_f.write("test entity recall for epoch %d: %.4f\n" % (epoch + 1, recall))
            ppx_f.write("epoch " + str(epoch + 1) + " ppx: " + str(ppx) + " ppx_word: " + str(ppx_word) + " ppx_entity: " + \
                str(ppx_entity) + '\n')
            ppx_f.close()


def evaluate(model, data_test, config, word2id, entity2id, epoch, writer, is_test=False, model_path=None):
    eval_start_time = time.time()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    sentence_ppx_loss = 0
    sentence_ppx_word_loss = 0
    entity_recall = 0
    entity_precision = 0
    total_graph_size = 0
    sentence_ppx_local_loss = 0
    word_cut = use_cuda(torch.Tensor([0]))
    local_cut = use_cuda(torch.Tensor([0]))
    count = 0
    model.is_inference = True
    id2word = dict()
    for key in word2id.keys():
        id2word[word2id[key]] = key

    def write_batch_res_text(word_index, id2word, selector=None):
        w = open(config.generated_text_name + '_' + str(epoch) + '.txt', 'a')
        batch_size = len(word_index)
        decoder_len = len(word_index[0])
        text = []
        if True:
            for i in range(batch_size):
                tmp_dict = dict()
                tmp = []
                for j in range(decoder_len):
                    if word_index[i][j] == 2:
                        break
                    tmp.append(id2word[word_index[i][j]])
                tmp_dict['res_text'] = tmp
                text.append(tmp_dict)

        for line in text:
            w.write(json.dumps(line) + '\n')
        w.close()

    iter_time = len(data_test) // config.batch_size
    for iteration in range(iter_time):
        count += 1
        data = data_test[(iteration * config.batch_size):(iteration * config.batch_size + config.batch_size)]
        if not config.decode:
            recall, precision, graph_size = run(model, data, config, word2id, entity2id, model.is_inference)
            entity_recall += recall
            entity_precision += precision
            total_graph_size += graph_size / config.batch_size
            if count % 50 == 0:
                print('iteration for evaluate: %d time used: %.2f' % (count, time.time() - eval_start_time))
            continue
        decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, word_neg_num, entity_neg_num, recall, precision, graph_size, word_index = \
            run(model, data, config, word2id, entity2id, model.is_inference)
        sentence_ppx_loss += torch.sum(sentence_ppx).data
        entity_recall += recall
        entity_precision += precision
        total_graph_size += graph_size / config.batch_size
        sentence_ppx_word_loss += torch.sum(sentence_ppx_word).data
        sentence_ppx_local_loss += torch.sum(sentence_ppx_local).data
        word_cut += word_neg_num
        local_cut += entity_neg_num

        if config.to_generate:
            write_batch_res_text(word_index, id2word)

        writer.add_scalar('test_loss/', decoder_loss.data, count)
        if count % 50 == 0:
            print("iteration for evaluate:", count, "loss:", decoder_loss.data, "time used: ", time.time() - eval_start_time)
    entity_recall /= count
    entity_precision /= count
    total_graph_size /= count

    if config.decode:
        model.is_inference = False
        if model_path != None:
            print('perplexity on test set:', np.exp(sentence_ppx_loss.cpu() / len(data_test)), \
                np.exp(sentence_ppx_word_loss.cpu() / (len(data_test) - int(word_cut))),
                np.exp(sentence_ppx_local_loss.cpu() / (len(data_test) - int(local_cut))))
            exit()
        ppl = np.exp(sentence_ppx_loss.cpu() / len(data_test))
        word_ppl = np.exp(sentence_ppx_word_loss.cpu() / (len(data_test) - int(word_cut)))
        entity_ppl = np.exp(sentence_ppx_local_loss.cpu() / (len(data_test) - int(local_cut)))
        print('perplexity on test set:', ppl, "word ppl: ", word_ppl, 'entity ppl: ', entity_ppl)
        print("response entity recall: ", entity_recall, ' precision: ', entity_precision, "graph size:", total_graph_size)
        writer.add_scalar('test_ppl/ppl', ppl, epoch)
        writer.add_scalar('test_ppl/word_ppl', word_ppl, epoch)
        writer.add_scalar('test_ppl/entity_ppl', entity_ppl, epoch)
        writer.add_scalar('graph/recall', entity_recall, epoch)
        writer.add_scalar('graph/precision', entity_precision, epoch)
        writer.add_scalar('graph/graph_size', total_graph_size, epoch)
        with open(config.log_dir, 'a') as f:
            f.write("perplexity on testset: %.2f word ppl: %.2f entity ppl: %.2f\n" % (ppl, word_ppl, entity_ppl))
            f.write("response entity recall: %.2f; precision: %.2f\n" % (entity_recall, entity_precision))

        return np.exp(sentence_ppx_loss.cpu() / len(data_test)), np.exp(sentence_ppx_word_loss.cpu() / (len(data_test) - int(word_cut))), \
            np.exp(sentence_ppx_local_loss.cpu() / (len(data_test) - int(local_cut))), entity_recall
    else:
        writer.add_scalar('graph_recall', entity_recall, epoch)
        writer.add_scalar('graph/precision', entity_precision, epoch)
        writer.add_scalar('graph/graph_size', total_graph_size, epoch)
        return


def main():
    # choose gpu with sufficient memory
    pynvml.nvmlInit()
    device_index = -1
    device_count = pynvml.nvmlDeviceGetCount()
    max_space = 0
    for i in range(device_count):
        if i == 1:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.free > max_space:
            max_space = info.free
            device_index = i
    if max_space < 1e9:
        print("no gpu with sufficient memory currently.")
        sys.exit(0)
    print("Running on device %d" % device_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_index)

    config = Config('config.yml')
    config.list_all_member()
    raw_vocab, data_train, data_test = prepare_data(config)
    word2id, entity2id, vocab, embed, entity_vocab, entity_embed, relation_vocab, relation_embed, entity_relation_embed, adj_table \
        = build_vocab(config.data_dir, raw_vocab, config=config)
    model = use_cuda(ConceptFlow(config, embed, entity_relation_embed, adj_table))
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_rate)
    writer = SummaryWriter(config.tb_path)

    ppx_f = open(config.result_dir_name,'a')
    for name, value in vars(config).items():
        ppx_f.write('%s = %s' % (name, value) + '\n')

    if config.is_train == False:
        evaluate(model, data_test, config, word2id, entity2id, 0, writer, model_path=config.test_model_path)
        exit() 
    
    train(config, model, data_train, data_test, word2id, entity2id, model_optimizer, writer)

main()
