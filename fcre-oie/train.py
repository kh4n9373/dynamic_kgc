import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


from sampler import data_sampler_CFRL, unknown_na_data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment
from encoder import EncodingModel

from transformers import BertTokenizer
from losses import MutualInformationLoss, HardSoftMarginTripletLoss, HardMarginLoss
from sklearn.metrics import f1_score

class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # 👎 --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist
    def _cosine_similarity(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        cos = nn.CosineSimilarity(dim=1)
        sim = []
        for i in range(b):
            sim_i = cos(x2, x1[i])
            sim.append(torch.unsqueeze(sim_i, 0))
        sim = torch.cat(sim, 0)
        return sim
    

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        

    def train_model(self, encoder, training_data, seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        soft_margin_loss = HardSoftMarginTripletLoss()

        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):

                optimizer.zero_grad()

                # if batch_num == 5:
                #     break
                
                # new_labels = torch.repeat_interleave(labels, repeats=self.config.num_gen_augment)
                repeats = (labels != self.config.na_id).int()*self.config.num_gen_augment
                repeats[repeats == 0] = 1
                new_labels = torch.repeat_interleave(labels, repeats=repeats)


                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)
                    batch_instance = {'ids': [], 'mask': []} 

                    ids_list = []
                    mask_list = []
                    for l in labels:
                        label_item = l.item()
                        rel_id = self.id2rel[label_item]
                        ids_list.extend(seen_des[rel_id]['ids'])
                        mask_list.extend(seen_des[rel_id]['mask'])
                    batch_instance['ids'] = torch.tensor(ids_list).to(self.config.device)
                    batch_instance['mask'] = torch.tensor(mask_list).to(self.config.device) 
                
                hidden = encoder(instance) # b, dim
                # loss = self.moment.contrastive_loss(hidden, labels, is_memory)
                labels_des = encoder(batch_instance, is_des = True) # b*num_gen_augment, dim
                rd = encoder(instance, is_rd=True) # b*num_gen_augment, dim

                repeated_hidden = torch.repeat_interleave(hidden, repeats=repeats.to(hidden.device), dim=0) # b*num_gen_augment, dim

                # Use broadcasting to compute equality matrix in parallel
                new_matrix_labels_tensor = (new_labels.unsqueeze(1) == new_labels.unsqueeze(0)).float().to(self.config.device)

                # calculate loss factors per label
                label_weights = torch.ones(len(new_labels)).to(self.config.device)
                unique_labels, label_counts = torch.unique(new_labels, return_counts=True)
                label_weights = 1.0 / label_counts[torch.searchsorted(unique_labels, new_labels)]
                label_weights = label_weights / label_weights.sum() * len(new_labels)
                label_weights = label_weights.to(self.config.device) # (b)

                # print(f"repeat hidden size: {repeated_hidden.size()}\tlabels_des size: {labels_des.size()}\trd size: {rd.size()}\tlabel_weights size: {label_weights.size()}")

                # compute mutual information loss: hidden vs des
                if self.config.w1 != 0:
                    loss_retrieval = MutualInformationLoss(weights=label_weights)
                    s_d_mi_loss = loss_retrieval(repeated_hidden, labels_des, new_matrix_labels_tensor)
                else:
                    s_d_mi_loss = 0.0

                # compute mutual information loss: hidden vs rd
                if self.config.w2 != 0:
                    loss_retrieval = MutualInformationLoss(weights=label_weights)
                    s_c_mi_loss = loss_retrieval(repeated_hidden, rd, new_matrix_labels_tensor)
                else:
                    s_c_mi_loss = 0.0

                # compute mutual information loss: des vs rd
                if self.config.w3 != 0:
                    loss_retrieval = MutualInformationLoss(weights=label_weights)
                    d_c_mi_loss = loss_retrieval(labels_des, rd, new_matrix_labels_tensor)
                else:
                    d_c_mi_loss = 0.0

                # compute soft margin triplet loss: hidden vs hidden
                uniquie_labels = labels.unique()
                if len(uniquie_labels) > 1 and self.config.w4 != 0:
                    s_s_loss = soft_margin_loss(hidden, labels.to(self.config.device))
                else:
                    s_s_loss = 0.0

                loss = self.config.w1*s_d_mi_loss + self.config.w2*s_c_mi_loss + self.config.w3*d_c_mi_loss + self.config.w4*s_s_loss

                if loss == 0:
                    continue

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')             

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 4
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total

    def f1(self, preds, labels):
        unique_labels = set(preds + labels)
        if self.config.na_id in unique_labels:
            unique_labels.remove(self.config.na_id)
        
        unique_labels = list(unique_labels)

        # Calculate F1 score for each class separately
        f1_per_class = f1_score(preds, labels, average=None)

        # Calculate micro-average F1 score
        f1_micro = f1_score(preds, labels, average='micro', labels=unique_labels)

        # Calculate macro-average F1 score
        # f1_macro = f1_score(preds, labels, average='macro', labels=unique_labels)

        # Calculate weighted-average F1 score
        f1_weighted = f1_score(preds, labels, average='weighted', labels=unique_labels)

        print("F1 score per class:", dict(zip(unique_labels, f1_per_class)))
        print("Micro-average F1 score:", f1_micro)
        # print("Macro-average F1 score:", f1_macro)
        print("Weighted-average F1 score:", f1_weighted)

        return f1_micro


    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des, filtered_test_na_data: list[int]=None):
        """
        Args:
            encoder: Encoder
            seen_proto: seen prototypes. NxH tensor
            seen_relid: relation id of protoytpes
            test_data: test data
            rep_des: representation of seen relation description. N x H tensor

        Returns:

        """
        batch_size = 48
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        preds = []
        labels = []
        corrects = 0.0

        preds1 = []
        corrects1 = 0.0

        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations
            logits_des = self._cosine_similarity(fea, rep_des)  # (B, N)
           
            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            preds.extend(pred)
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size

            # by logits_des
            cur_index1 = torch.argmax(logits_des,dim=1)
            pred1 = []
            for i in range(cur_index1.size()[0]):
                pred1.append(seen_relid[int(cur_index1[i])])
            preds1.extend(pred1)
            pred1 = torch.tensor(pred1)
            correct1 = torch.eq(pred1, label).sum().item()
            acc1 = correct1/ batch_size
            corrects1 += correct1

            labels.extend(label.cpu().tolist())

        if filtered_test_na_data is not None:
            j = -1
            for i, label in enumerate(labels):
                if label == self.config.na_id:
                    j += 1
                    if filtered_test_na_data[j] == self.config.na_id_2:
                        preds[i] = self.config.na_id
                        preds1[i] = self.config.na_id

        print('')
        # return corrects / total, corrects1 / total, corrects2 / total
        return self.f1(preds, labels), self.f1(preds1, labels)

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        
        if len(sampler.filtered_test_na_data[0]) == 0:
            raise ValueError("filtered_test_na_data is empty")

        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number wo na
        cur_acc_wo_na, total_acc_wo_na = [], []
        cur_acc1_wo_na, total_acc1_wo_na = [], []

        cur_acc_num_wo_na, total_acc_num_wo_na = [], []
        cur_acc_num1_wo_na, total_acc_num1_wo_na = [], []

        # step is continual task number w na
        cur_acc_w_na, total_acc_w_na = [], []
        cur_acc1_w_na, total_acc1_w_na = [], []

        cur_acc_num_w_na, total_acc_num_w_na = [], []
        cur_acc_num1_w_na, total_acc_num1_w_na = [], []

        # step is continual task number with filtered na
        cur_acc_w_filtered_na, total_acc_w_filtered_na = [], []
        cur_acc1_w_filtered_na, total_acc1_w_filtered_na = [], []

        cur_acc_num_w_filtered_na, total_acc_num_w_filtered_na = [], []
        cur_acc_num1_w_filtered_na, total_acc_num1_w_filtered_na = [], []


        memory_samples = {}
        data_generation = []
        seen_des = {}


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])
        
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions, \
            cur_filtered_test_na_data, history_filtered_test_na_data) in enumerate(sampler):

            for rel in current_relations:
                
                if rel in seen_des and rel == self.id2rel[self.config.na_id]:
                    continue

                for augment_des in seen_descriptions[rel]:
                    ids = self.tokenizer.encode(augment_des,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.config.max_length)        
                    # mask
                    mask = np.zeros(self.config.max_length, dtype=np.int32)
                    end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                    mask[:end_index + 1] = 1 
                    if rel not in seen_des:
                        seen_des[rel] = {}
                        seen_des[rel]['ids'] = [ids]
                        seen_des[rel]['mask'] = [mask]
                    else:
                        seen_des[rel]['ids'].append(ids)
                        seen_des[rel]['mask'].append(mask)

            print(f"seen_des: {seen_des.keys()}")

            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize, seen_des)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True) 
                self.train_model(encoder, memory_data_initialize, seen_des, is_memory=True)

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]
            list_seen_des = []
            for i in range(len(seen_proto)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            with torch.no_grad():
                rep_des = []
                for i in range(len(list_seen_des)):
                    mean_hidden = []
                    for j in range(len(list_seen_des[i]['ids'])):
                        sample = {
                            'ids' : torch.tensor([list_seen_des[i]['ids'][j]]).to(self.config.device),
                            'mask' : torch.tensor([list_seen_des[i]['mask'][j]]).to(self.config.device)
                        }
                        hidden = encoder(sample, is_des=True)
                        hidden = hidden.detach().cpu().data
                        mean_hidden.append(hidden)
                        # calculate mean_hidden list of 3 elements to 1 elements
                    mean_hidden = torch.stack(mean_hidden, dim=0)  # Shape: (num_samples, hidden_dim)
                    mean_hidden = torch.mean(mean_hidden, dim=0) 
                    rep_des.append(mean_hidden)
                rep_des = torch.cat(rep_des, dim=0)
            

            # Eval current task and history task wo na
            test_data_initialize_cur_wo_na, test_data_initialize_seen_wo_na = [], []
            for rel in current_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_cur_wo_na += test_data[rel]
                
            for rel in seen_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_seen_wo_na += historic_test_data[rel]
            
            ac1_wo_na, ac1_des_wo_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_wo_na,rep_des)
            ac2_wo_na, ac2_des_wo_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_wo_na,rep_des)

            # Eval current task and history task w na
            test_data_initialize_cur_w_na, test_data_initialize_seen_w_na = [], []
            for rel in current_relations:
                test_data_initialize_cur_w_na += test_data[rel]
                
            for rel in seen_relations:
                test_data_initialize_seen_w_na += historic_test_data[rel]
            
            ac1_w_na, ac1_des_w_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_w_na,rep_des)
            ac2_w_na, ac2_des_w_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_w_na,rep_des)

            # Eval current task and history task with filtered na
            test_data_initialize_cur_w_na, test_data_initialize_seen_w_na = [], []
            for rel in current_relations:
                test_data_initialize_cur_w_na += test_data[rel]
                
            for rel in seen_relations:
                test_data_initialize_seen_w_na += historic_test_data[rel]
            
            ac1_w_filtered_na, ac1_des_w_filtered_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_w_na,rep_des, cur_filtered_test_na_data)
            ac2_w_filtered_na, ac2_des_w_filtered_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_w_na,rep_des, history_filtered_test_na_data)
            
            # wo na
            cur_acc_num_wo_na.append(ac1_wo_na)
            total_acc_num_wo_na.append(ac2_wo_na)
            cur_acc_wo_na.append('{:.4f}'.format(ac1_wo_na))
            total_acc_wo_na.append('{:.4f}'.format(ac2_wo_na))
            print('cur_acc_wo_na: ', cur_acc_wo_na)
            print('his_acc_wo_na: ', total_acc_wo_na)

            cur_acc_num1_wo_na.append(ac1_des_wo_na)
            total_acc_num1_wo_na.append(ac2_des_wo_na)
            cur_acc1_wo_na.append('{:.4f}'.format(ac1_des_wo_na))
            total_acc1_wo_na.append('{:.4f}'.format(ac2_des_wo_na))
            print('cur_acc des_wo_na: ', cur_acc1_wo_na)
            print('his_acc des_wo_na: ', total_acc1_wo_na)

            # w na
            cur_acc_num_w_na.append(ac1_w_na)
            total_acc_num_w_na.append(ac2_w_na)
            cur_acc_w_na.append('{:.4f}'.format(ac1_w_na))
            total_acc_w_na.append('{:.4f}'.format(ac2_w_na))
            print('cur_acc_w_na: ', cur_acc_w_na)
            print('his_acc_w_na: ', total_acc_w_na)

            cur_acc_num1_w_na.append(ac1_des_w_na)
            total_acc_num1_w_na.append(ac2_des_w_na)
            cur_acc1_w_na.append('{:.4f}'.format(ac1_des_w_na))
            total_acc1_w_na.append('{:.4f}'.format(ac2_des_w_na))
            print('cur_acc des_w_na: ', cur_acc1_w_na)
            print('his_acc des_w_na: ', total_acc1_w_na)

            # w filtered na
            cur_acc_num_w_filtered_na.append(ac1_w_filtered_na)
            total_acc_num_w_filtered_na.append(ac2_w_filtered_na)
            cur_acc_w_filtered_na.append('{:.4f}'.format(ac1_w_filtered_na))
            total_acc_w_filtered_na.append('{:.4f}'.format(ac2_w_filtered_na))
            print('cur_acc_w_filtered_na: ', cur_acc_w_filtered_na)
            print('his_acc_w_filtered_na: ', total_acc_w_filtered_na)

            cur_acc_num1_w_filtered_na.append(ac1_des_w_filtered_na)
            total_acc_num1_w_filtered_na.append(ac2_des_w_filtered_na)
            cur_acc1_w_filtered_na.append('{:.4f}'.format(ac1_des_w_filtered_na))
            total_acc1_w_filtered_na.append('{:.4f}'.format(ac2_des_w_filtered_na))
            print('cur_acc des_w_filtered_na: ', cur_acc1_w_filtered_na)
            print('his_acc des_w_filtered_na: ', total_acc1_w_filtered_na)


        torch.cuda.empty_cache()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        return (total_acc_num_wo_na, total_acc_num1_wo_na), (total_acc_num_w_na, total_acc_num1_w_na), (total_acc_num_w_filtered_na, total_acc_num1_w_filtered_na)

    def train_unknown_only(self):
        # sampler 
        sampler = unknown_na_data_sampler_CFRL(config=self.config, seed=self.config.seed)

        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number wo na
        cur_acc_wo_na, total_acc_wo_na = [], []
        cur_acc1_wo_na, total_acc1_wo_na = [], []

        cur_acc_num_wo_na, total_acc_num_wo_na = [], []
        cur_acc_num1_wo_na, total_acc_num1_wo_na = [], []

        # step is continual task number w na
        cur_acc_w_na, total_acc_w_na = [], []
        cur_acc1_w_na, total_acc1_w_na = [], []

        cur_acc_num_w_na, total_acc_num_w_na = [], []
        cur_acc_num1_w_na, total_acc_num1_w_na = [], []

        # step is continual task number with filtered na
        cur_acc_w_filtered_na, total_acc_w_filtered_na = [], []
        cur_acc1_w_filtered_na, total_acc1_w_filtered_na = [], []

        cur_acc_num_w_filtered_na, total_acc_num_w_filtered_na = [], []
        cur_acc_num1_w_filtered_na, total_acc_num1_w_filtered_na = [], []


        memory_samples = {}
        data_generation = []
        seen_des = {}


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])
        
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

            for rel in current_relations:
                
                if rel in seen_des and rel == self.id2rel[self.config.na_id]:
                    continue

                for augment_des in seen_descriptions[rel]:
                    ids = self.tokenizer.encode(augment_des,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.config.max_length)        
                    # mask
                    mask = np.zeros(self.config.max_length, dtype=np.int32)
                    end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                    mask[:end_index + 1] = 1 
                    if rel not in seen_des:
                        seen_des[rel] = {}
                        seen_des[rel]['ids'] = [ids]
                        seen_des[rel]['mask'] = [mask]
                    else:
                        seen_des[rel]['ids'].append(ids)
                        seen_des[rel]['mask'].append(mask)

            print(f"seen_des: {seen_des.keys()}")

            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize, seen_des)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True) 
                self.train_model(encoder, memory_data_initialize, seen_des, is_memory=True)

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]
            list_seen_des = []
            for i in range(len(seen_proto)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            with torch.no_grad():
                rep_des = []
                for i in range(len(list_seen_des)):
                    mean_hidden = []
                    for j in range(len(list_seen_des[i]['ids'])):
                        sample = {
                            'ids' : torch.tensor([list_seen_des[i]['ids'][j]]).to(self.config.device),
                            'mask' : torch.tensor([list_seen_des[i]['mask'][j]]).to(self.config.device)
                        }
                        hidden = encoder(sample, is_des=True)
                        hidden = hidden.detach().cpu().data
                        mean_hidden.append(hidden)
                        # calculate mean_hidden list of 3 elements to 1 elements
                    mean_hidden = torch.stack(mean_hidden, dim=0)  # Shape: (num_samples, hidden_dim)
                    mean_hidden = torch.mean(mean_hidden, dim=0) 
                    rep_des.append(mean_hidden)
                rep_des = torch.cat(rep_des, dim=0)
            

            # Eval current task and history task wo na
            test_data_initialize_cur_wo_na, test_data_initialize_seen_wo_na = [], []
            for rel in current_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_cur_wo_na += test_data[rel]
                
            for rel in seen_relations:
                if rel != self.id2rel[self.config.na_id]:
                    test_data_initialize_seen_wo_na += historic_test_data[rel]
            
            ac1_wo_na, ac1_des_wo_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_wo_na,rep_des)
            ac2_wo_na, ac2_des_wo_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_wo_na,rep_des)

            # Eval current task and history task w na
            test_data_initialize_cur_w_na, test_data_initialize_seen_w_na = [], []
            for rel in current_relations:
                test_data_initialize_cur_w_na += test_data[rel]
                
            for rel in seen_relations:
                test_data_initialize_seen_w_na += historic_test_data[rel]
            
            ac1_w_na, ac1_des_w_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur_w_na,rep_des)
            ac2_w_na, ac2_des_w_na = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen_w_na,rep_des)

            # Eval current task and history task with filtered na
            test_data_initialize_cur_w_na, test_data_initialize_seen_w_na = [], []
            for rel in current_relations:
                test_data_initialize_cur_w_na += test_data[rel]
                
            for rel in seen_relations:
                test_data_initialize_seen_w_na += historic_test_data[rel]
            
            ac1_w_filtered_na, ac1_des_w_filtered_na = 0, 0
            ac2_w_filtered_na, ac2_des_w_filtered_na = 0, 0
            
            # wo na
            cur_acc_num_wo_na.append(ac1_wo_na)
            total_acc_num_wo_na.append(ac2_wo_na)
            cur_acc_wo_na.append('{:.4f}'.format(ac1_wo_na))
            total_acc_wo_na.append('{:.4f}'.format(ac2_wo_na))
            print('cur_acc_wo_na: ', cur_acc_wo_na)
            print('his_acc_wo_na: ', total_acc_wo_na)

            cur_acc_num1_wo_na.append(ac1_des_wo_na)
            total_acc_num1_wo_na.append(ac2_des_wo_na)
            cur_acc1_wo_na.append('{:.4f}'.format(ac1_des_wo_na))
            total_acc1_wo_na.append('{:.4f}'.format(ac2_des_wo_na))
            print('cur_acc des_wo_na: ', cur_acc1_wo_na)
            print('his_acc des_wo_na: ', total_acc1_wo_na)

            # w na
            cur_acc_num_w_na.append(ac1_w_na)
            total_acc_num_w_na.append(ac2_w_na)
            cur_acc_w_na.append('{:.4f}'.format(ac1_w_na))
            total_acc_w_na.append('{:.4f}'.format(ac2_w_na))
            print('cur_acc_w_na: ', cur_acc_w_na)
            print('his_acc_w_na: ', total_acc_w_na)

            cur_acc_num1_w_na.append(ac1_des_w_na)
            total_acc_num1_w_na.append(ac2_des_w_na)
            cur_acc1_w_na.append('{:.4f}'.format(ac1_des_w_na))
            total_acc1_w_na.append('{:.4f}'.format(ac2_des_w_na))
            print('cur_acc des_w_na: ', cur_acc1_w_na)
            print('his_acc des_w_na: ', total_acc1_w_na)

            # w filtered na
            cur_acc_num_w_filtered_na.append(ac1_w_filtered_na)
            total_acc_num_w_filtered_na.append(ac2_w_filtered_na)
            cur_acc_w_filtered_na.append('{:.4f}'.format(ac1_w_filtered_na))
            total_acc_w_filtered_na.append('{:.4f}'.format(ac2_w_filtered_na))
            print('cur_acc_w_filtered_na: ', cur_acc_w_filtered_na)
            print('his_acc_w_filtered_na: ', total_acc_w_filtered_na)

            cur_acc_num1_w_filtered_na.append(ac1_des_w_filtered_na)
            total_acc_num1_w_filtered_na.append(ac2_des_w_filtered_na)
            cur_acc1_w_filtered_na.append('{:.4f}'.format(ac1_des_w_filtered_na))
            total_acc1_w_filtered_na.append('{:.4f}'.format(ac2_des_w_filtered_na))
            print('cur_acc des_w_filtered_na: ', cur_acc1_w_filtered_na)
            print('his_acc des_w_filtered_na: ', total_acc1_w_filtered_na)


        torch.cuda.empty_cache()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        return (total_acc_num_wo_na, total_acc_num1_wo_na), (total_acc_num_w_na, total_acc_num1_w_na), (total_acc_num_w_filtered_na, total_acc_num1_w_filtered_na)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    parser.add_argument("--only_unknown", action='store_true')

    # num_gen_augment
    parser.add_argument("--num_gen_augment", default=1, type=int)

    # batch_size
    parser.add_argument("--batch_size", default=16, type=int)

    # loss weight 1
    parser.add_argument("--w1", default=2.0, type=float)

    # loss weight 2
    parser.add_argument("--w2", default=2.0, type=float)

    # loss weight 3
    parser.add_argument("--w3", default=2.0, type=float)

    # loss weight 4
    parser.add_argument("--w4", default=1.0, type=float)

    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen
    config.only_unknown = args.only_unknown
    config.num_gen_augment = args.num_gen_augment
    config.batch_size = args.batch_size
    config.w1 = args.w1
    config.w2 = args.w2
    config.w3 = args.w3
    config.w4 = args.w4

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print(f'batch_size={config.batch_size}, lr={config.lr}, epoch={config.epoch}, epoch_mem={config.epoch_mem}, num_gen_augment={config.num_gen_augment}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.na_id = 80
        config.na_id_2 = 81
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = f'./data/CFRLFewRel/relation_description_detail_{config.num_gen_augment}.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = f'./data/CFRLFewRel/CFRLdata_10_100_10_5/train_0_{config.num_gen_augment}.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.na_id = 41
        config.na_id_2 = 42
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = f'./data/CFRLTacred/relation_description_detail_{config.num_gen_augment}.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = f'./data/CFRLTacred/CFRLdata_6_100_5_5/train_0_{config.num_gen_augment}.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    # wo na
    acc_list_wo_na = []
    acc_list1_wo_na = []

    # w na
    acc_list_w_na = []
    acc_list1_w_na = []

    # w filtered na
    acc_list_w_filtered_na = []
    acc_list1_w_filtered_na = []

    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)

        if config.only_unknown:
            (acc_wo_na, acc1_wo_na), (acc_w_na, acc1_w_na), (acc_w_filtered_na, acc1_w_filtered_na) = manager.train_unknown_only()
        else:
            (acc_wo_na, acc1_wo_na), (acc_w_na, acc1_w_na), (acc_w_filtered_na, acc1_w_filtered_na) = manager.train()
        
        # wo na
        acc_list_wo_na.append(acc_wo_na)
        acc_list1_wo_na.append(acc1_wo_na)

        # w na
        acc_list_w_na.append(acc_w_na)
        acc_list1_w_na.append(acc1_w_na)

        # w filtered na
        acc_list_w_filtered_na.append(acc_w_filtered_na)
        acc_list1_w_filtered_na.append(acc1_w_filtered_na)

        torch.cuda.empty_cache()
    
    # wo na
    accs_wo_na = np.array(acc_list_wo_na)
    ave_wo_na = np.mean(accs_wo_na, axis=0)
    std_wo_na = np.std(accs_wo_na, axis=0)
    print('----------END')
    print('his_acc mean_wo_na: ', np.around(ave_wo_na, 4))
    print('his_acc std_wo_na: ', np.around(std_wo_na, 4))
    accs1_wo_na = np.array(acc_list1_wo_na)
    ave1_wo_na = np.mean(accs1_wo_na, axis=0)
    std1_wo_na = np.std(accs1_wo_na, axis=0) 
    print('his_acc des mean_wo_na: ', np.around(ave1_wo_na, 4))
    print('his_acc des std_wo_na: ', np.around(std1_wo_na, 4))

    # w na
    accs_w_na = np.array(acc_list_w_na)
    ave_w_na = np.mean(accs_w_na, axis=0)
    std_w_na = np.std(accs_w_na, axis=0)
    print('his_acc mean_w_na: ', np.around(ave_w_na, 4))
    print('his_acc std_w_na: ', np.around(std_w_na, 4))
    accs1_w_na = np.array(acc_list1_w_na)
    ave1_w_na = np.mean(accs1_w_na, axis=0)
    std1_w_na = np.std(accs1_w_na, axis=0)
    print('his_acc des mean_w_na: ', np.around(ave1_w_na, 4))
    print('his_acc des std_w_na: ', np.around(std1_w_na, 4))

    # w filtered na
    accs_w_filtered_na = np.array(acc_list_w_filtered_na)
    ave_w_filtered_na = np.mean(accs_w_filtered_na, axis=0)
    std_w_filtered_na = np.std(accs_w_filtered_na, axis=0)
    print('his_acc mean_w_filtered_na: ', np.around(ave_w_filtered_na, 4))
    print('his_acc std_w_filtered_na: ', np.around(std_w_filtered_na, 4))
    accs1_w_filtered_na = np.array(acc_list1_w_filtered_na)
    ave1_w_filtered_na = np.mean(accs1_w_filtered_na, axis=0)
    std1_w_filtered_na = np.std(accs1_w_filtered_na, axis=0)
    print('his_acc des mean_w_filtered_na: ', np.around(ave1_w_filtered_na, 4))
    print('his_acc des std_w_filtered_na: ', np.around(std1_w_filtered_na, 4))