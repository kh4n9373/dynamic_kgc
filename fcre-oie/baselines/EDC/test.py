# %%
import pickle
import os 
import random
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
import json

class data_sampler_CFRL(object):
    def __init__(self, base_seed):
        self.task_length = 8
        self.relation_name = "/home/aac/edc_3/datasets/fewrel/relation_name.txt"
        self.relation_description = "/home/aac/edc_3/datasets/fewrel/relation_description.txt"
        self.rel_cluster_label  = "/home/aac/edc_3/datasets/fewrel/rel_cluster_label_0.npy"
        self.test_data = "/home/aac/edc_3/datasets/fewrel/test_0.txt"
        # self.test_na_data = "/home/aac/edc_3/datasets/fewrel/na_test.json"
        self.rel_index = "/home/aac/edc_3/datasets/fewrel/rel_index.npy"
        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.relation_name)
        self.rel2des, self.id2des = self._read_descriptions(self.relation_description)

        self.num_of_relation = len(self.id2rel)
        print(self.num_of_relation)
        # read data
        # self.training_data = self._read_data(self.training_data)
        self.test_data = self._read_data(self.test_data)

        # read na data
        self.na_id = 41
        # self.training_na_data = self._read_na_data(self.training_data)
        # self.valid_na_data = self._read_na_data(self.valid_data, self._temp_datapath('valid'))
        # self.test_na_data = self._read_na_data(self.test_na_data)
        self.na_rel = self.id2rel[self.na_id]

        # read relation order
        rel_index = np.load(self.rel_index)
        print("rel_index", self.rel_index)
        rel_cluster_label = np.load(self.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        # shuffle task order
        self.seed = base_seed
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

            # cur_training_data[rel] = self.training_data[index]
            # cur_na_training_data.extend(self.training_na_data[index])

            # cur_valid_data[rel] = self.valid_data[index]
            # cur_na_valid_data.extend(self.valid_na_data[index])

            cur_test_data[rel] = self.test_data[index]
            # cur_na_test_data.extend(self.test_na_data[index])

            self.history_test_data[rel] = self.test_data[index]
            # fix_here 
            self.seen_descriptions[rel] = self.id2des[index]

        # if self.na_rel not in self.seen_relations:
        #     self.seen_relations.append(self.na_rel)
        #     self.history_test_data[self.na_rel] = cur_na_test_data
        #     self.seen_descriptions[self.na_rel] = self.id2des[self.na_id]
        # else:
        #     self.history_test_data[self.na_rel] += cur_na_test_data

        current_relations.append(self.na_rel)
        # cur_training_data[self.na_rel] = cur_na_training_data
        # cur_valid_data[self.na_rel] = cur_na_valid_data
        cur_test_data[self.na_rel] = cur_na_test_data
        
        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations, self.seen_descriptions
    

    def _read_data(self, file):
        # if os.path.isfile(save_data_path):
        #     with open(save_data_path, 'rb') as f:
        #         datas = pickle.load(f)
        #         print(save_data_path)
        #     return datas
        # else:
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
                        # sample['tokens'] = items[2].split()
                        sample['input_text'] = items[2]
                        sample['description'] = self.id2des[sample['relation']]
                        headent = items[3]
                        headidx = [[int(ix) for ix in items[4].split()]]
                        tailent = items[5]
                        tailidx = [[int(ix) for ix in items[6].split()]]
                        headid = items[7]
                        tailid = items[8]
                        sample['h'] = [headent, headid, headidx]
                        sample['t'] = [tailent, tailid, tailidx]
                        sample['relation_definition'] = items[10] if len(items) >= 11 else None
                        samples.append(sample)

        read_data = [[] for i in range(self.num_of_relation)]
        for sample in samples:
            tokenized_sample = sample
            read_data[tokenized_sample['relation']].append(tokenized_sample)
            # with open(save_data_path, 'wb') as f:
            #     pickle.dump(read_data, f)
            #     print(save_data_path)
        return read_data

    def _read_na_data(self, file: str):
        file = file.replace('train_0.txt', 'na_train.json').replace('valid_0.txt', 'na_valid.json').replace('test_0.txt', 'na_test.json')
        # save_data_path = save_data_path.replace('train.pkl', 'na_train.pkl').replace('valid.pkl', 'na_valid.pkl').replace('test.pkl', 'na_test.pkl')
        # if os.path.isfile(save_data_path):
        #     with open(save_data_path, 'rb') as f:
        #         datas = pickle.load(f)
        #         print(save_data_path)
        #     return datas
        # else:
        with open(file) as f:
            na_data = json.load(f)

        processed_na_data = [[] for i in range(self.num_of_relation)]

        for key in na_data.keys():
            # processed_na_data[int(key)] = []
            sample = {}
            for line in na_data[key]:
                items = line.strip().split('\t')
                if (len(items[0]) > 0):
                    sample['relation'] = int(items[0]) - 1
                    # sample['index'] = i
                    if items[1] != 'noNegativeAnswer':
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        # sample['tokens'] = items[2].split()
                        sample['input_text'] = items[2]
                        # sample['description'] = self.id2des[sample['relation']]
                        headent = items[3]
                        headidx = [[int(ix) for ix in items[4].split()]]
                        tailent = items[5]
                        tailidx = [[int(ix) for ix in items[6].split()]]
                        headid = items[7]
                        tailid = items[8]
                        sample['h'] = [headent, headid, headidx]
                        sample['t'] = [tailent, tailid, tailidx]

                        sample['relation_definition'] = self.id2des[self.na_id] if "train" in file else None

                        processed_na_data[int(key)].append(sample)


        # with open(save_data_path, 'wb') as f:
        #     pickle.dump(processed_na_data, f)
        #     print(save_data_path)
        return processed_na_data


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
                id2des[index] = x[2:]
        return rel2des, id2des  



# %%
def find_task(rel_id:int, cluster:dict):
    for task in cluster:
        if rel_id in cluster[task]:
            return task
# def check_entities(pred:list[str], gold:list[str]):
#     new_pred_h = set(pred[0].lower().replace("_", " ").split())
#     new_pred_t = set(pred[1].lower().replace("_", " ").split())
#     new_gold_h = set(gold[0].lower().split())
#     new_gold_t = set(gold[1].lower().split())
#     # print(new_pred_h)
#     if len(new_pred_h & new_gold_h) > 0  and len(new_pred_t & new_gold_t) > 0:
#         return True
#     elif len(new_pred_h & new_gold_t) > 0  and len(new_pred_t & new_gold_h) > 0:
#         return True
#     # print("new_pred_h",new_pred_h, len(new_pred_h & new_gold_h))
#     # print("new_gold_h",new_gold_h)
#     # print("pred", pred[0].lower().replace("_", " ") ,",", pred[1].lower().replace("_", " "))
#     # print("gold", gold[0].lower() ,",", gold[1].lower())
#     return False
# def check(entity_text:str) -> bool:
    # if entity_text.lower() in [head_entity.lower(), end_entity.lower()]:
    #     return False
    # if entity_text.lower() in ["lrb", "rrb"]:
    #     return False
    # if entity_text.lower() in head_entity.lower() or entity_text.lower() in end_entity.lower():
    #     return False
    # if head_entity.lower() in entity_text.lower() or end_entity.lower() in entity_text.lower():
    #     return False
    # return True

def check_entities(pred:list[str], gold:list[str]):
    new_pred_h = pred[0].lower()
    new_pred_t = pred[1].lower()
    new_gold_h = gold[0].lower()
    new_gold_t = gold[1].lower()
    # print(new_pred_h)

    if (new_pred_h in new_gold_h or new_gold_h in new_pred_h) and (new_pred_t in new_gold_t or new_gold_t in new_pred_t):
        return True
    elif (new_pred_h in new_gold_t or new_pred_t in new_pred_h) and (new_pred_t in new_gold_h or new_gold_h in new_pred_t):
        return True
    # print("new_pred_h",new_pred_h, len(new_pred_h & new_gold_h))
    # print("new_gold_h",new_gold_h)
    # print("pred", pred[0].lower().replace("_", " ") ,",", pred[1].lower().replace("_", " "))
    # print("gold", gold[0].lower() ,",", gold[1].lower())
    return False
def find_idx_same_entities(schema_list:list[list], gold:list):
    idx_lst = []
    for idx, schema in enumerate(schema_list):
        if schema is not None:
            if check_entities([schema[0], schema[2]], gold):
                idx_lst.append(idx)
    return idx_lst

# %%
total_round = 6
base_seed = 100
import torch
from sklearn.metrics import f1_score
random.seed(base_seed) 
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)   
from itertools import combinations
lst = []
dix = {}
for i in range(total_round):
    seed = base_seed + i * 100
    print('--------Round ', i)
    print('seed: ', seed)
    
    sampler = data_sampler_CFRL(seed)
    task_order = [int(i) for i in sampler.shuffle_index]
    phase_3_result_path = f"/home/aac/edc_3/phase_3_result_{task_order}_full_eti_fewrel.json"
    with open(phase_3_result_path, "r") as f:
        phase_3_result = json.load(f)
    sub_lst = []
    for step, (training_data, valid_data, test_data, current_relations,
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
        dix[str(step)] = [sampler.rel2id[rel] for rel in current_relations][:-1]
    break
dix

# %%
sampler = data_sampler_CFRL(seed)

# %%
sampler = data_sampler_CFRL(seed)
sampler.id2rel.keys()

# %%
total_round = 6
base_seed = 100
import torch
from sklearn.metrics import f1_score
random.seed(base_seed) 
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)   
from itertools import combinations
lst = []
for i in range(total_round):
    seed = base_seed + i * 100
    print('--------Round ', i)
    print('seed: ', seed)
    
    sampler = data_sampler_CFRL(seed)
    task_order = [int(i) for i in sampler.shuffle_index]
    phase_3_result_path = f"/home/aac/edc_3/phase_3_result_{task_order}_full_eti_fewrel.json"
    with open(phase_3_result_path, "r") as f:
        phase_3_result = json.load(f)
    sub_lst = []
    for step, (training_data, valid_data, test_data, current_relations,
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
        # print('step: ', step)
        full_pred_id = []
        full_gold_id = []
        for rel in seen_relations:  # Each `rel` for each iteration
            if rel != "NA or unknown":
                
                cluster = sampler.cluster_to_labels
                task = str(find_task(sampler.rel2id[rel], cluster))
                # print('task: ', task)
                # print('rel: ', rel)
                # print(str(sampler.rel2id[rel]))
                for gold_line, text in zip(historic_test_data[rel], phase_3_result[str(step)][task][str(sampler.rel2id[rel])]):
                    gold_head = gold_line['h'][0]
                    gold_tail = gold_line['t'][0]
                    gold = [gold_head, gold_tail]
                    entities = eval(text["entity_hint"].strip())
                    sub_gold_entities = list(combinations(entities, 2))
                    sub_gold = [sampler.na_id if i > 0 else sampler.rel2id[rel] for i in range(len(sub_gold_entities))]
                    sub_pred = [sampler.na_id  for i in range(len(sub_gold_entities)) ]
                    for k, go in enumerate(sub_gold_entities):   
                        idx_lst = find_idx_same_entities(text["schema_canonicalizaiton"], go) # loose in order of entities, and if entities have 2 relation
                        if len(idx_lst) > 0:
                            sub_pred[k] = sampler.rel2id[text["schema_canonicalizaiton"][idx_lst[0]][1]]
                            # print(sub_gold[k], sub_pred[k] , go, sampler.id2rel[sub_gold[k]], text["schema_canonicalizaiton"][idx_lst[0]] )
                    # n = len(text["schema_canonicalizaiton"])
                    # sub_gold = [sampler.na_id if i not in idx_lst else sampler.rel2id[rel] for i in range(n)]
                    # sub_pred = [sampler.rel2id[text["schema_canonicalizaiton"][i][1]] if text["schema_canonicalizaiton"][i] is not None else sampler.na_id for i in range(n)]
                    full_pred_id.extend(sub_pred)
                    full_gold_id.extend(sub_gold)
        
        unique_labels = set(full_pred_id + full_gold_id)
        if sampler.na_id in unique_labels:
            unique_labels.remove(sampler.na_id)
        
        unique_labels = list(unique_labels)

        # Calculate F1 score for each class separately
        # f1_per_class = f1_score(full_pred_id, full_gold_id, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(full_pred_id, full_gold_id, average='micro', labels=unique_labels)

        # Calculate macro-average F1 score
        # f1_macro = f1_score(preds, labels, average='macro', labels=unique_labels)

        # Calculate weighted-average F1 score
        # f1_weighted = f1_score(preds, labels, average='weighted', labels=unique_labels)

        # print("F1 score per class:", dict(zip(unique_labels, f1_per_class)))
        print(f"Task {step}", f1_micro)
        # print("Macro-average F1 score:", f1_macro)
        # print("Weighted-average F1 score:", f1_weighted)
        sub_lst.append(f1_micro)
        # compute f1 score 
        # merge list   
    lst.append(sub_lst)         
    if i == 1:
        break
import numpy as np

# # Sample lists
# list1 = [1, 2, 3, 4, 5]
# list2 = [6, 7, 8, 9, 10]

# # Compute mean and std for corresponding elements
means = []
std_devs = []

for a1, a2 in zip(lst[0], lst[1]):
    mean = np.mean([a1, a2])
    std_dev = np.std([a1*100, a2*100])
    std_dev = np.std([a1, a2])
    means.append(f"{mean*100:.2f}")
    std_devs.append(std_dev)

print("Means:", means)
print("Standard Deviations:", std_devs)

# %%
with open("/home/aac/edc_3/phase_3_result_[7, 3, 0, 5, 4, 1, 6, 2]_full_eti_fewrel.json", "r") as f:
    data = json.load(f)
data['0'].keys()

# %%


# %%
from sklearn.metrics import f1_score
import torch
import pandas as pd
import os
import shutil
import subprocess
import json
import random
import numpy as np
from itertools import combinations
from copy import deepcopy
from tqdm import tqdm
# from gliner import GLiNER
import ast
    
# combinations_ = list(combinations(entities, 2))[1:]        
total_round = 6
base_seed = 100

random.seed(base_seed) 
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)   
lst = []
for i in range(total_round):
    seed = base_seed + i * 100
    print('--------Round ', i)
    print('seed: ', seed)
    
    sampler = data_sampler_CFRL(seed)
    task_order = [int(i) for i in sampler.shuffle_index]
    phase_3_result_path = f"/home/aac/edc_3/phase_3_result_{task_order}_eti_tacred.json"
    with open(phase_3_result_path, "r") as f:
        phase_3_result = json.load(f, )
    sub_lst = []
    for step, (training_data, valid_data, test_data, current_relations,
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
        # print('step: ', step)
        full_pred_id = []
        full_gold_id = []
        for rel in seen_relations:  # Each `rel` for each iteration
            if rel != "NA or unknown":
                
                cluster = sampler.cluster_to_labels
                task = str(find_task(sampler.rel2id[rel], cluster))
                # print('task: ', task)
                # print('rel: ', rel)
                for gold_line, text in zip(historic_test_data[rel], phase_3_result[str(step)][task][str(sampler.rel2id[rel])]):
                    gold_head = gold_line['h'][0]
                    gold_tail = gold_line['t'][0]
                    # gold = [gold_head, gold_tail]
                    # print(text["entity_hint"].strip()[1:-1].split(", "))
                    gold = text["entity_hint"].strip()[1:-1].split(", ")
                    # print(gold)
                    idx_lst = find_idx_same_entities(text["schema_canonicalizaiton"], gold) # loose in order of entities, and if entities have 2 relation
                    # sub_pred = [sampler.na_id] * len(text["schema_canonicalizaiton"])
                    # sub_gold = [sampler.na_id] * len(text["schema_canonicalizaiton"])
                    # sub_gold = [sub_gold[i] if i not in idx_lst else sampler.rel2id[rel] for i in range(len(sub_gold))]
                    # sub_pred = [sampler.rel2id[text["schema_canonicalizaiton"][i][1]] if text["schema_canonicalizaiton"][i] is not None else sub_pred[i] for i in range(len(sub_pred))]
                    # full_pred_id.extend(sub_pred)
                    # full_gold_id.extend(sub_gold)
                    
                    sub_gold = sampler.rel2id[rel]
                    if len(idx_lst) > 0:
                        sub_pred = sampler.rel2id[text["schema_canonicalizaiton"][idx_lst[0]][1]] #[sampler.rel2id[text["schema_canonicalizaiton"][i][1]] if text["schema_canonicalizaiton"][i] is not None else sub_pred[i] for i in range(len(sub_pred))]
                    else:
                        sub_pred = sampler.na_id
                    full_pred_id.append(sub_pred)
                    full_gold_id.append(sub_gold)
        
        unique_labels = set(full_pred_id + full_gold_id)
        if sampler.na_id in unique_labels:
            unique_labels.remove(sampler.na_id)
        
        unique_labels = list(unique_labels)

        # Calculate F1 score for each class separately
        # f1_per_class = f1_score(full_pred_id, full_gold_id, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(full_pred_id, full_gold_id, average='micro', labels=unique_labels)

        # Calculate macro-average F1 score
        # f1_macro = f1_score(preds, labels, average='macro', labels=unique_labels)
        print(f"Task {step}", f1_micro)
        # print("Macro-average F1 score:", f1_macro)
        # print("Weighted-average F1 score:", f1_weighted)
        sub_lst.append(f1_micro)
        # compute f1 score 
        # merge list   
    lst.append(sub_lst)         
    if i == 1:
        break
import numpy as np

# # Sample lists
# list1 = [1, 2, 3, 4, 5]
# list2 = [6, 7, 8, 9, 10]

# # Compute mean and std for corresponding elements
means = []
std_devs = []

for a1, a2 in zip(lst[0], lst[1]):
    mean = np.mean([a1, a2])
    std_dev = np.std([a1, a2])
    means.append(f"{mean*100:.2f}")
    std_devs.append(std_dev)

print("Means:", means)
print("Standard Deviations:", std_devs)

# %%
from sklearn.metrics import f1_score
import torch
import pandas as pd
import os
import shutil
import subprocess
import json
import random
import numpy as np

        
        
total_round = 6
base_seed = 100

random.seed(base_seed) 
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)   

for i in range(total_round):
    seed = base_seed + i * 100
    print('--------Round ', i)
    print('seed: ', seed)
    
    sampler = data_sampler_CFRL(seed)
    task_order = [int(i) for i in sampler.shuffle_index]
    phase_3_result_path = f"/home/aac/edc_3/phase_3_result_{task_order}.json"
    with open(phase_3_result_path, "r") as f:
        phase_3_result = json.load(f)
    for step, (training_data, valid_data, test_data, current_relations,
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
        # print('step: ', step)
        full_pred_id = []
        full_gold_id = []
        for rel in seen_relations:  # Each `rel` for each iteration
            if rel != "NA or unknown":
                
                cluster = sampler.cluster_to_labels
                task = str(find_task(sampler.rel2id[rel], cluster))
                # print('task: ', task)
                # print('rel: ', rel)
                for gold_line, text in zip(historic_test_data[rel], phase_3_result[str(step)][task][str(sampler.rel2id[rel])]):
                    gold_head = gold_line['h'][0]
                    gold_tail = gold_line['t'][0]
                    gold = [gold_head, gold_tail]

                    idx_lst = find_idx_same_entities(text["schema_canonicalizaiton"], gold) # loose in order of entities, and if entities have 2 relation
                    # sub_pred = [sampler.na_id] * len(text["schema_canonicalizaiton"])
                    # sub_gold = [sampler.na_id] * len(text["schema_canonicalizaiton"])
                    # sub_gold = [sub_gold[i] if i not in idx_lst else sampler.rel2id[rel] for i in range(len(sub_gold))]
                    # sub_pred = [sampler.rel2id[text["schema_canonicalizaiton"][i][1]] if text["schema_canonicalizaiton"][i] is not None else sub_pred[i] for i in range(len(sub_pred))]
                    # full_pred_id.extend(sub_pred)
                    # full_gold_id.extend(sub_gold)
                    
                    sub_gold = sampler.rel2id[rel]
                    if len(idx_lst) > 0:
                        sub_pred = sampler.rel2id[text["schema_canonicalizaiton"][idx_lst[0]][1]] #[sampler.rel2id[text["schema_canonicalizaiton"][i][1]] if text["schema_canonicalizaiton"][i] is not None else sub_pred[i] for i in range(len(sub_pred))]
                    else:
                        sub_pred = sampler.na_id
                    full_pred_id.append(sub_pred)
                    full_gold_id.append(sub_gold)
        
        unique_labels = set(full_pred_id + full_gold_id)
        if sampler.na_id in unique_labels:
            unique_labels.remove(sampler.na_id)
        
        unique_labels = list(unique_labels)

        # Calculate F1 score for each class separately
        # f1_per_class = f1_score(full_pred_id, full_gold_id, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(full_pred_id, full_gold_id, average='micro', labels=unique_labels)

        # Calculate macro-average F1 score
        # f1_macro = f1_score(preds, labels, average='macro', labels=unique_labels)

        # Calculate weighted-average F1 score
        # f1_weighted = f1_score(preds, labels, average='weighted', labels=unique_labels)

        # print("F1 score per class:", dict(zip(unique_labels, f1_per_class)))
        print(f"Task {step}", f1_micro)
        # print("Macro-average F1 score:", f1_macro)
        # print("Weighted-average F1 score:", f1_weighted)

        # compute f1 score 
        # merge list            
    
    # break


