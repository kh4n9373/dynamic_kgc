import pickle
import os 
import random
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
import json

class data_sampler_CFRL(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.max_length = self.config.max_length
        self.task_length = self.config.task_length
        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        if self.config.model == 'bert':
            self.mask_token = '[MASK]' 
            model_path = self.config.bert_path
            tokenizer_from_pretrained = BertTokenizer.from_pretrained
        elif self.config.model == 'roberta':
            self.mask_token = '<mask>' 
            model_path = self.config.roberta_path
            tokenizer_from_pretrained = RobertaTokenizer.from_pretrained

        # tokenizer
        if config.pattern == 'marker':
            self.tokenizer = tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens)
            self.config.h_ids = self.tokenizer.get_vocab()[self.unused_tokens[0]]
            self.config.t_ids = self.tokenizer.get_vocab()[self.unused_tokens[2]]
        elif config.pattern == 'hardprompt' or config.pattern == 'cls':
            self.tokenizer = tokenizer_from_pretrained(model_path)
        elif config.pattern == 'softprompt' or config.pattern == 'hybridprompt':
            self.tokenizer =tokenizer_from_pretrained(model_path, \
            additional_special_tokens=[self.unused_token])
            self.config.prompt_token_ids = self.tokenizer.get_vocab()[self.unused_token]

        self.config.vocab_size = len(self.tokenizer)
        self.config.sep_token_ids = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.config.mask_token_ids = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.sep_token_ids, self.mask_token_ids =  self.config.sep_token_ids, self.config.mask_token_ids

        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_name)
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)

        self.config.num_of_relation = len(self.id2rel)

        # read data
        self.training_data = self._read_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_data = self._read_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_data = self._read_data(self.config.test_data, self._temp_datapath('test'))

        # read na data
        self.na_id = self.config.na_id
        self.training_na_data, _ = self._read_na_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_na_data, _ = self._read_na_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_na_data, self.filtered_test_na_data  = self._read_na_data(self.config.test_data, self._temp_datapath('test'))
        self.na_rel = self.id2rel[self.na_id]

        # read relation order
        rel_index = np.load(self.config.rel_index)
        rel_cluster_label = np.load(self.config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        # shuffle task order
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)        
        print(f'Task_order: {self.shuffle_index}')
        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
        self.history_filtered_test_na_data = []
        self.seen_descriptions = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)


    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()
        
        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        cur_na_training_data = []
        cur_na_valid_data = []
        cur_na_test_data = []
        cur_filtered_test_na_data = []


        for index in indexs:
            rel = self.id2rel[index]
            current_relations.append(rel)
            self.seen_relations.append(rel)

            cur_training_data[rel] = self.training_data[index]
            cur_na_training_data.extend(self.training_na_data[index])

            cur_valid_data[rel] = self.valid_data[index]
            cur_na_valid_data.extend(self.valid_na_data[index])

            cur_test_data[rel] = self.test_data[index]
            cur_na_test_data.extend(self.test_na_data[index])
            cur_filtered_test_na_data.extend(self.filtered_test_na_data[index])

            self.history_test_data[rel] = self.test_data[index]
            # fix_here 
            self.seen_descriptions[rel] = self.id2des[index]

        if self.na_rel not in self.seen_relations:
            self.seen_relations.append(self.na_rel)
            self.history_test_data[self.na_rel] = cur_na_test_data
            self.seen_descriptions[self.na_rel] = self.id2des[self.na_id]
        else:
            self.history_test_data[self.na_rel] += cur_na_test_data
        
        self.history_filtered_test_na_data.extend(cur_filtered_test_na_data)

        current_relations.append(self.na_rel)
        cur_training_data[self.na_rel] = cur_na_training_data
        cur_valid_data[self.na_rel] = cur_na_valid_data
        cur_test_data[self.na_rel] = cur_na_test_data
        
        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations, self.seen_descriptions,\
            cur_filtered_test_na_data, self.history_filtered_test_na_data

    def _temp_datapath(self, data_type):
        '''
            data_type = 'train'/'valid'/'test'
        '''
        temp_name = [data_type]
        file_name = '{}.pkl'.format('-'.join([str(x) for x in temp_name]))
        prompt_len = self.config.prompt_len * self.config.prompt_num
        if self.config.model == 'bert':
            tp1 = '_process_BERT_'
        elif self.config.model == 'roberta':
            tp1 = '_process_Roberta_'
        if self.config.task_name == 'FewRel':
            tp2 = 'CFRLFewRel/CFRLdata_10_100_10_'
        else:
            tp2 = 'CFRLTacred/CFRLdata_6_100_5_'
        if self.config.pattern == 'hardprompt':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1  + self.config.pattern)
        elif self.config.pattern == 'softprompt' or self.config.pattern == 'hybridprompt':                
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern + '_' + str(prompt_len) + 'token')
        elif self.config.pattern == 'cls':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern)            
        elif self.config.pattern == 'marker':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k),  \
            tp1 + self.config.pattern)   

        mid_dir += f"_numgenaugment{self.config.num_gen_augment}"
           
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        
        save_data_path = os.path.join(mid_dir, file_name)   
        return save_data_path     

    def _read_data(self, file, save_data_path):
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(save_data_path)
            return datas
        else:
            samples = []
            with open(file) as f:
                for i, line in enumerate(f):
                    sample = {}
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1
                        sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            sample['description'] = self.id2des[sample['relation']]
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]
                            sample['relation_definition'] = items[10:10+self.config.num_gen_augment] if len(items) >= 11 else None
                            samples.append(sample)

            read_data = [[] for i in range(self.config.num_of_relation)]
            for sample in samples:
                tokenized_sample = self.tokenize(sample)
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            with open(save_data_path, 'wb') as f:
                pickle.dump(read_data, f)
                print(save_data_path)
            return read_data

    def _read_na_data(self, file: str, save_data_path):
        # file = file.replace('train_0.txt', 'na_train.json').replace('valid_0.txt', 'na_valid.json').replace('test_0.txt', 'na_test.json')
        if "train_0" in file:
            file = os.path.dirname(file) + "/na_train.json"
        elif "valid_0" in file:
            file = os.path.dirname(file) + "/na_valid.json"
        elif "test_0" in file:
            file = os.path.dirname(file) + "/na_test.json"

        save_data_path = save_data_path.replace('train.pkl', 'na_train.pkl').replace('valid.pkl', 'na_valid.pkl').replace('test.pkl', 'na_test.pkl')
        save_data_path_2 = save_data_path.replace('.pkl', '_filtered_na.pkl')
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(save_data_path)

            with open(save_data_path_2, 'rb') as f:
                filtered_test_na_data = pickle.load(f)
                print(save_data_path_2)
            
            return datas, filtered_test_na_data
        else:
            print(f"Reading NA data from {file}")
            with open(file) as f:
                na_data = json.load(f)

            processed_na_data = [[] for i in range(self.config.num_of_relation)]
            filtered_test_na_data = [[] for i in range(self.config.num_of_relation)]

            for key in na_data.keys():
                # processed_na_data[int(key)] = []
                sample = {}
                for line in na_data[key]:
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1
                        if "test" in file:
                            filtered_test_na_data[int(key)].append(sample['relation'])
                        if "test" in file and sample['relation'] - 1 == self.na_id:
                            sample['relation'] = self.na_id

                        # sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            sample['description'] = self.id2des[sample['relation']]
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]

                            sample['relation_definition'] = [self.id2des[self.na_id]] if "train" in file else None

                            processed_na_data[int(key)].append(self.tokenize(sample))

            with open(save_data_path, 'wb') as f:
                pickle.dump(processed_na_data, f)
                print(save_data_path)

            with open(save_data_path_2, 'wb') as f:
                pickle.dump(filtered_test_na_data, f)
                print(save_data_path_2)

            return processed_na_data, filtered_test_na_data

    def tokenize(self, sample):
        tokenized_sample = {}
        tokenized_sample['relation'] = sample['relation']
        # tokenized_sample['index'] = sample['index']
        # tokenized_sample['oie'] = sample['oie']
        if self.config.pattern == 'hardprompt':
            ids, mask = self._tokenize_hardprompt(sample)
        elif self.config.pattern == 'softprompt':
            ids, mask = self._tokenize_softprompt(sample)   
        elif self.config.pattern == 'hybridprompt':
            ids, mask, rd_ids, rd_mask = self._tokenize_hybridprompt(sample)
            tokenized_sample['rd_ids'] = rd_ids
            tokenized_sample['rd_mask'] = rd_mask                    
        elif self.config.pattern == 'marker':
            ids, mask = self._tokenize_marker(sample)
        elif self.config.pattern == 'cls':
            ids, mask = self._tokenize_cls(sample)            
        tokenized_sample['ids'] = ids
        tokenized_sample['mask'] = mask    
        return tokenized_sample    


    def _read_relations(self, file):
        id2rel, rel2id = {}, {}
        with open(file) as f:
            for index, line in enumerate(f):
                rel = line.strip()
                id2rel[index] = rel
                rel2id[rel] = index
        return id2rel, rel2id
    
    def _read_descriptions(self, file):
            # id2rel, rel2id = {}, {}
            rel2des = {}
            id2des = {}
            with open(file, 'r', encoding = 'utf-8', errors='ignore') as f:
                for index, line in enumerate(f):
                    rel = line.strip()
                    x = rel.split('\t')
                    rel2des[x[1]] = x[2:]
                    id2des[int(x[0])] = x[2:]
            return rel2des, id2des  
    
    def _tokenize_softprompt(self, sample):
        '''
        X [v] [v] [v] [v]
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        prompt = raw_tokens + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len \
                             + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask           

    def _tokenize_hybridprompt(self, sample):
        '''
        X [v] e1 [v] [MASK] [v] e2 [v] 
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens + [self.unused_token] * prompt_len + h + [self.unused_token] * prompt_len \
               + [self.mask_token] + [self.unused_token] * prompt_len + t + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        # tokenize rd
        if sample['relation_definition']:
            _rd_ids, _rd_mask = [], []
            for rd in sample['relation_definition']:
                rd_ids = self.tokenizer.encode(rd,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.max_length)        
                # mask
                rd_mask = np.zeros(self.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
                rd_mask[:end_index + 1] = 1

                _rd_ids.append(rd_ids)
                _rd_mask.append(rd_mask)
        else:
            _rd_ids = []
            _rd_mask = []

        return ids, mask, _rd_ids, _rd_mask 

    def _tokenize_hardprompt(self, sample):
        '''
        X e1 [MASK] e2 
        '''
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens +  h + [self.mask_token] + t
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask

    def _tokenize_marker(self, sample):
        '''
        [unused]e[unused]
        '''
        raw_tokens = sample['tokens']
        h1, h2, t1, t2 =  sample['h'][2][0][0], sample['h'][2][0][-1], sample['t'][2][0][0], sample['t'][2][0][-1]
        new_tokens = []

        # add entities marker        
        for index, token in enumerate(raw_tokens):
            if index == h1:
                new_tokens.append(self.unused_tokens[0])
                new_tokens.append(token)
                if index == h2:
                    new_tokens.append(self.unused_tokens[1])
            elif index == h2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[1])
            elif index == t1:
                new_tokens.append(self.unused_tokens[2])
                new_tokens.append(token)
                if index == t2:
                    new_tokens.append(self.unused_tokens[3])
            elif index == t2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[3])
            else:
                new_tokens.append(token)
            
            ids = self.tokenizer.encode(' '.join(new_tokens),
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_length)
            
            # mask
            mask = np.zeros(self.max_length, dtype=np.int32)
            end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
            mask[:end_index + 1] = 1

        return ids, mask

    def _tokenize_cls(self, sample):
        '''
        [CLS] X
        '''
        raw_tokens = sample['tokens']
        ids = self.tokenizer.encode(' '.join(raw_tokens),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1

        return ids, mask


class unknown_na_data_sampler_CFRL(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.max_length = self.config.max_length
        self.task_length = self.config.task_length
        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        if self.config.model == 'bert':
            self.mask_token = '[MASK]' 
            model_path = self.config.bert_path
            tokenizer_from_pretrained = BertTokenizer.from_pretrained
        elif self.config.model == 'roberta':
            self.mask_token = '<mask>' 
            model_path = self.config.roberta_path
            tokenizer_from_pretrained = RobertaTokenizer.from_pretrained

        # tokenizer
        if config.pattern == 'marker':
            self.tokenizer = tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens)
            self.config.h_ids = self.tokenizer.get_vocab()[self.unused_tokens[0]]
            self.config.t_ids = self.tokenizer.get_vocab()[self.unused_tokens[2]]
        elif config.pattern == 'hardprompt' or config.pattern == 'cls':
            self.tokenizer = tokenizer_from_pretrained(model_path)
        elif config.pattern == 'softprompt' or config.pattern == 'hybridprompt':
            self.tokenizer =tokenizer_from_pretrained(model_path, \
            additional_special_tokens=[self.unused_token])
            self.config.prompt_token_ids = self.tokenizer.get_vocab()[self.unused_token]

        self.config.vocab_size = len(self.tokenizer)
        self.config.sep_token_ids = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.config.mask_token_ids = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.sep_token_ids, self.mask_token_ids =  self.config.sep_token_ids, self.config.mask_token_ids

        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_name)
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)

        self.config.num_of_relation = len(self.id2rel)

        # read data
        self.training_data = self._read_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_data = self._read_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_data = self._read_data(self.config.test_data, self._temp_datapath('test'))

        # read na data
        self.na_id = self.config.na_id
        self.training_na_data = self._read_na_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_na_data = self._read_na_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_na_data  = self._read_na_data(self.config.test_data, self._temp_datapath('test'))
        self.na_rel = self.id2rel[self.na_id]

        # read relation order
        rel_index = np.load(self.config.rel_index)
        rel_cluster_label = np.load(self.config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        # shuffle task order
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)        
        print(f'Task_order: {self.shuffle_index}')
        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}
        self.history_filtered_test_na_data = []
        self.seen_descriptions = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)


    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()
        
        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        cur_na_training_data = []
        cur_na_valid_data = []
        cur_na_test_data = []


        for index in indexs:
            rel = self.id2rel[index]
            current_relations.append(rel)
            self.seen_relations.append(rel)

            cur_training_data[rel] = self.training_data[index]
            cur_na_training_data.extend(self.training_na_data[index])

            cur_valid_data[rel] = self.valid_data[index]
            cur_na_valid_data.extend(self.valid_na_data[index])

            cur_test_data[rel] = self.test_data[index]
            cur_na_test_data.extend(self.test_na_data[index])

            self.history_test_data[rel] = self.test_data[index]
            # fix_here 
            self.seen_descriptions[rel] = self.id2des[index]

        if self.na_rel not in self.seen_relations:
            self.seen_relations.append(self.na_rel)
            self.history_test_data[self.na_rel] = cur_na_test_data
            self.seen_descriptions[self.na_rel] = self.id2des[self.na_id]
        else:
            self.history_test_data[self.na_rel] += cur_na_test_data
    

        current_relations.append(self.na_rel)
        cur_training_data[self.na_rel] = cur_na_training_data
        cur_valid_data[self.na_rel] = cur_na_valid_data
        cur_test_data[self.na_rel] = cur_na_test_data
        
        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations, self.seen_descriptions

    def _temp_datapath(self, data_type):
        '''
            data_type = 'train'/'valid'/'test'
        '''
        temp_name = [data_type]
        file_name = '{}.pkl'.format('-'.join([str(x) for x in temp_name]))
        prompt_len = self.config.prompt_len * self.config.prompt_num
        if self.config.model == 'bert':
            tp1 = '_process_BERT_'
        elif self.config.model == 'roberta':
            tp1 = '_process_Roberta_'
        if self.config.task_name == 'FewRel':
            tp2 = 'CFRLFewRel/CFRLdata_10_100_10_'
        else:
            tp2 = 'CFRLTacred/CFRLdata_6_100_5_'
        if self.config.pattern == 'hardprompt':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1  + self.config.pattern)
        elif self.config.pattern == 'softprompt' or self.config.pattern == 'hybridprompt':                
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern + '_' + str(prompt_len) + 'token')
        elif self.config.pattern == 'cls':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern)            
        elif self.config.pattern == 'marker':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k),  \
            tp1 + self.config.pattern)   

        mid_dir += f"_numgenaugment{self.config.num_gen_augment}"
           
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        
        save_data_path = os.path.join(mid_dir, file_name)   
        return save_data_path     

    def _read_data(self, file, save_data_path):
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(save_data_path)
            return datas
        else:
            samples = []
            with open(file) as f:
                for i, line in enumerate(f):
                    sample = {}
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1
                        sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            sample['description'] = self.id2des[sample['relation']]
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]
                            sample['relation_definition'] = items[10:10+self.config.num_gen_augment] if len(items) >= 11 else None
                            samples.append(sample)

            read_data = [[] for i in range(self.config.num_of_relation)]
            for sample in samples:
                tokenized_sample = self.tokenize(sample)
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            with open(save_data_path, 'wb') as f:
                pickle.dump(read_data, f)
                print(save_data_path)
            return read_data

    def _read_na_data(self, file: str, save_data_path):
        # file = file.replace('train_0.txt', 'na_train.json').replace('valid_0.txt', 'na_valid.json').replace('test_0.txt', 'na_test.json')
        if "train_0" in file:
            file = os.path.dirname(file) + "/na_train_uk.json"
        elif "valid_0" in file:
            file = os.path.dirname(file) + "/na_valid.json"
        elif "test_0" in file:
            file = os.path.dirname(file) + "/na_test.json"

        save_data_path = save_data_path.replace('train.pkl', 'na_train_uk.pkl').replace('valid.pkl', 'na_valid_uk.pkl').replace('test.pkl', 'na_test_uk.pkl')

        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(save_data_path)
            
            return datas
        else:
            print(f"Reading NA data from {file}")
            with open(file) as f:
                na_data = json.load(f)

            processed_na_data = [[] for i in range(self.config.num_of_relation)]

            for key in na_data.keys():
                # processed_na_data[int(key)] = []
                sample = {}
                for line in na_data[key]:
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1

                        if sample['relation'] - 1 == self.na_id:
                            continue

                        # sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            sample['description'] = self.id2des[sample['relation']]
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]

                            sample['relation_definition'] = [self.id2des[self.na_id]] if "train" in file else None

                            processed_na_data[int(key)].append(self.tokenize(sample))

            with open(save_data_path, 'wb') as f:
                pickle.dump(processed_na_data, f)
                print(save_data_path)

            return processed_na_data

    def tokenize(self, sample):
        tokenized_sample = {}
        tokenized_sample['relation'] = sample['relation']
        # tokenized_sample['index'] = sample['index']
        # tokenized_sample['oie'] = sample['oie']
        if self.config.pattern == 'hardprompt':
            ids, mask = self._tokenize_hardprompt(sample)
        elif self.config.pattern == 'softprompt':
            ids, mask = self._tokenize_softprompt(sample)   
        elif self.config.pattern == 'hybridprompt':
            ids, mask, rd_ids, rd_mask = self._tokenize_hybridprompt(sample)
            tokenized_sample['rd_ids'] = rd_ids
            tokenized_sample['rd_mask'] = rd_mask                    
        elif self.config.pattern == 'marker':
            ids, mask = self._tokenize_marker(sample)
        elif self.config.pattern == 'cls':
            ids, mask = self._tokenize_cls(sample)            
        tokenized_sample['ids'] = ids
        tokenized_sample['mask'] = mask    
        return tokenized_sample    


    def _read_relations(self, file):
        id2rel, rel2id = {}, {}
        with open(file) as f:
            for index, line in enumerate(f):
                rel = line.strip()
                id2rel[index] = rel
                rel2id[rel] = index
        return id2rel, rel2id
    
    def _read_descriptions(self, file):
            # id2rel, rel2id = {}, {}
            rel2des = {}
            id2des = {}
            with open(file, 'r', encoding = 'utf-8', errors='ignore') as f:
                for index, line in enumerate(f):
                    rel = line.strip()
                    x = rel.split('\t')
                    rel2des[x[1]] = x[2:]
                    id2des[int(x[0])] = x[2:]
            return rel2des, id2des  
    
    def _tokenize_softprompt(self, sample):
        '''
        X [v] [v] [v] [v]
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        prompt = raw_tokens + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len \
                             + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask           

    def _tokenize_hybridprompt(self, sample):
        '''
        X [v] e1 [v] [MASK] [v] e2 [v] 
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens + [self.unused_token] * prompt_len + h + [self.unused_token] * prompt_len \
               + [self.mask_token] + [self.unused_token] * prompt_len + t + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        # tokenize rd
        if sample['relation_definition']:
            _rd_ids, _rd_mask = [], []
            for rd in sample['relation_definition']:
                rd_ids = self.tokenizer.encode(rd,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=self.max_length)        
                # mask
                rd_mask = np.zeros(self.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
                rd_mask[:end_index + 1] = 1

                _rd_ids.append(rd_ids)
                _rd_mask.append(rd_mask)
        else:
            _rd_ids = []
            _rd_mask = []

        return ids, mask, _rd_ids, _rd_mask 

    def _tokenize_hardprompt(self, sample):
        '''
        X e1 [MASK] e2 
        '''
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens +  h + [self.mask_token] + t
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask

    def _tokenize_marker(self, sample):
        '''
        [unused]e[unused]
        '''
        raw_tokens = sample['tokens']
        h1, h2, t1, t2 =  sample['h'][2][0][0], sample['h'][2][0][-1], sample['t'][2][0][0], sample['t'][2][0][-1]
        new_tokens = []

        # add entities marker        
        for index, token in enumerate(raw_tokens):
            if index == h1:
                new_tokens.append(self.unused_tokens[0])
                new_tokens.append(token)
                if index == h2:
                    new_tokens.append(self.unused_tokens[1])
            elif index == h2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[1])
            elif index == t1:
                new_tokens.append(self.unused_tokens[2])
                new_tokens.append(token)
                if index == t2:
                    new_tokens.append(self.unused_tokens[3])
            elif index == t2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[3])
            else:
                new_tokens.append(token)
            
            ids = self.tokenizer.encode(' '.join(new_tokens),
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_length)
            
            # mask
            mask = np.zeros(self.max_length, dtype=np.int32)
            end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
            mask[:end_index + 1] = 1

        return ids, mask

    def _tokenize_cls(self, sample):
        '''
        [CLS] X
        '''
        raw_tokens = sample['tokens']
        ids = self.tokenizer.encode(' '.join(raw_tokens),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1

        return ids, mask
