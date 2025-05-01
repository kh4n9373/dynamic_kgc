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
        # self.test_na_data = "/home/aac/edc_3/datasets/na_test.json"
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
        # cur_test_data[self.na_rel] = cur_na_test_data
        
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


import torch
import pandas as pd
import os
import shutil
import subprocess
import json
import random
import numpy as np

def find_task(rel_id:int, cluster:dict):
    for task in cluster:
        if rel_id in cluster[task]:
            return task
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

    # Initialize results dictionary
    phase_3_result_path = f"/home/aac/edc_3/phase_3_result_{task_order}_full_eti_fewrel.json" #modify
    schema_path = "/home/aac/edc_3/schemas/example_schema_fewrel.csv"

    # # If schema path exists, remove and create
    # if os.path.exists(schema_path):
    #     os.remove(schema_path)  # Remove schema if it exists

    if os.path.exists(phase_3_result_path):
        with open(phase_3_result_path, "r") as f:
            results_1_2 = json.load(f)
    else:
        results_1_2 = {}

    for step, (training_data, valid_data, test_data, current_relations,
               historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

        # Update schema with relations for each iteration
        schema = ""
        for rel in seen_relations:
            schema+=f"{rel}|{sampler.rel2des[rel][0]}\n"
        with open(schema_path, "w") as f:  # Append to schema
            f.write(schema)

        for rel in seen_relations:  # Each `rel` for each iteration
            if rel != "NA or unknown":
                line_to_write = ""
                rel_id = sampler.rel2id[rel]
                for line in historic_test_data[rel][:]:
                    line_to_write += f'{line["input_text"]}\n'

                # Write example text
                with open("/home/aac/edc_3/datasets/example_fewrel.txt", "w") as f:
                    f.write(line_to_write)

                folder_path = "/home/aac/edc_3/output_fewrel"

                # Check and remove folder if it exists
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Folder '{folder_path}' has been removed successfully.")
                else:
                    print(f"Folder '{folder_path}' does not exist.")

                env = os.environ.copy()
                env["OPENAI_KEY"] = # Replace with your actual API key
                env.update({
                    "OIE_LLM": "gpt-4o-mini",
                    "SD_LLM": "gpt-4o-mini",
                    "SC_LLM": "gpt-4o-mini",
                    "SC_EMBEDDER": "intfloat/e5-mistral-7b-instruct", #sentence-transformers/all-MiniLM-L6-v2
                    "DATASET": "example",
                    "HF_HOME": "/home/aac/models",
                    "CUDA_VISIBLE_DEVICES": "2"
                })
                
                task_key = str(find_task(rel_id, sampler.cluster_to_labels))
                # Define the command to run
                command = [
                    "python", "run.py",
                    "--oie_llm", env["OIE_LLM"],
                    "--oie_few_shot_example_file_path", f"./few_shot_examples/{env['DATASET']}/oie_few_shot_examples.txt",
                    "--sd_llm", env["SD_LLM"],
                    "--sd_few_shot_example_file_path", f"./few_shot_examples/{env['DATASET']}/sd_few_shot_examples.txt",
                    "--sc_llm", env["SC_LLM"],
                    "--sc_embedder", env["SC_EMBEDDER"],
                    "--input_text_file_path", f"./datasets/{env['DATASET']}_fewrel.txt",
                    "--target_schema_path", f"./schemas/{env['DATASET']}_schema_fewrel.csv",
                    "--output_dir", folder_path + f"/{env['DATASET']}_target_alignment",
                    "--logging_verbose",
                    "--task_id", f"{task_key}",
                    "--rel_id", f"{str(rel_id)}",
                    "--phase_1_2", "false",
                    "--phase_1_2_path", "/home/aac/edc_3/phase_1_2_result_full_eti_fewrel.json" # modify
                ]

                # Run the modified command inside the subprocess with the updated environment
                subprocess.run(command, env=env, check=True)

                # Load data from generated file
                result_file_path = folder_path + "/example_target_alignment/iter0/result_at_each_stage.json"
                if os.path.exists(result_file_path):
                    with open(result_file_path, "r") as f:
                        data = json.load(f)
                else:
                    print(f"Warning: {result_file_path} does not exist.")
                    data = {}

                # Fix duplicate key issue: Convert `task_order[step]` to a string key
                step_str = str(step)
                if step_str not in results_1_2:
                    results_1_2[step_str] = {}
                if task_key not in results_1_2[step_str]:
                    results_1_2[step_str][task_key] = {}

                results_1_2[step_str][task_key][str(rel_id)] = data

                # Write updated results back to JSON file
                with open(phase_3_result_path, "w") as f:
                    json.dump(results_1_2, f, indent=4)

        # break  # Exit after processing the first relation for each round
    if i ==1:
        break  # Exit after the first round