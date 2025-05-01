import argparse
import random
from sampler import data_sampler
from config import Config
import torch
from model.bert_encoder import Bert_Encoder
from model.dropout_layer import Dropout_Layer
from model.classifier import Softmax_Layer, Proto_Softmax_Layer
from data_loader import get_data_loader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
import collections
from copy import deepcopy
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
import time
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train_simple_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    for epoch_i in range(epochs):
        losses = []

        step = 0
        for batch_data in tqdm(data_loader):

            # step += 1
            # if step == 5:
            #     break

            optimizer.zero_grad()
            labels, _, tokens = batch_data
            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)

            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)
            reps, _ = dropout_layer(reps)
            logits = classifier(reps)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


def compute_jsd_loss(m_input):
    # m_input: the result of m times dropout after the classifier.
    # size: m*B*C
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def contrastive_loss(hidden, labels):

    logsoftmax = nn.LogSoftmax(dim=-1)

    return -(logsoftmax(hidden) * labels).sum() / labels.sum()


def construct_hard_triplets(output, labels, relation_data):
    positive = []
    negative = []
    pdist = nn.PairwiseDistance(p=2)
    for rep, label in zip(output, labels):
        positive_relation_data = relation_data[label.item()]
        negative_relation_data = []
        for key in relation_data.keys():
            if key != label.item():
                negative_relation_data.extend(relation_data[key])

        positive_relation_data = torch.stack(positive_relation_data).to(rep.device)
        negative_relation_data = torch.stack(negative_relation_data).to(rep.device)
        
        # positive_distance = torch.stack([pdist(rep.cpu(), p) for p in positive_relation_data])
        positive_distance = pdist(rep, positive_relation_data)
        # negative_distance = torch.stack([pdist(rep.cpu(), n) for n in negative_relation_data])
        negative_distance = pdist(rep, negative_relation_data)

        positive_index = torch.argmax(positive_distance)
        negative_index = torch.argmin(negative_distance)
        positive.append(positive_relation_data[positive_index.item()])
        negative.append(negative_relation_data[negative_index.item()])


    return positive, negative


def train_first(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    for epoch_i in range(epochs):
        losses = []
        step = 0
        for labels, _, tokens in tqdm(data_loader):

            # step += 1
            # if step == 5:
            #     break

            optimizer.zero_grad()

            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = torch.tensor([map_relid2tempid[x.item()] for x in labels]).to(config.device)

            # Get representations
            reps = encoder(tokens)
            outputs, _ = dropout_layer(reps)
            positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            # Replace for loop with single batch computation
            # Create f_pass copies of reps using expand
            batch_reps = reps.unsqueeze(0).expand(config.f_pass, -1, -1)  # Shape: [f_pass, batch_size, hidden_dim]

            # Apply dropout to all copies at once
            batch_outputs, batch_embeddings = dropout_layer(batch_reps)  # Shape: [f_pass, batch_size, hidden_dim]

            # Get logits for all passes in one computation
            logits_all = classifier(batch_outputs)  # Shape: [f_pass, batch_size, num_classes]

            # Process triplets
            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs

            # Create labels matrix
            m_labels = labels.expand(config.f_pass, labels.shape[0])  # Shape: [f_pass, batch_size]

            # Compute label weights
            flat_labels = m_labels.reshape(-1)

            # Compute losses
            loss1 = criterion(logits_all.reshape(-1, logits_all.shape[-1]), flat_labels)
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1.mean() + loss2 + tri_loss

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")


def train_mem_model(config, encoder, dropout_layer, classifier, training_data, epochs, map_relid2tempid, new_relation_data,
                prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index):
    data_loader = get_data_loader(config, training_data, shuffle=True)

    encoder.train()
    dropout_layer.train()
    classifier.train()

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00001},
        {'params': dropout_layer.parameters(), 'lr': 0.00001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    distill_criterion = nn.CosineEmbeddingLoss()
    T = config.kl_temp
    for epoch_i in range(epochs):
        losses = []
        step = 0
        for labels, _, tokens in tqdm(data_loader):
            
            # step += 1
            # if step == 5:
            #     break
            start_time = time.time()

            optimizer.zero_grad()

            logits_all = []
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            origin_labels = labels[:]
            labels = [map_relid2tempid[x.item()] for x in labels]
            labels = torch.tensor(labels).to(config.device)
            reps = encoder(tokens)
            normalized_reps_emb = F.normalize(reps.view(-1, reps.size()[1]), p=2, dim=1)
            outputs,_ = dropout_layer(reps)
            if prev_dropout_layer is not None:
                prev_outputs, _ = prev_dropout_layer(reps)
                positives,negatives = construct_hard_triplets(prev_outputs, origin_labels, new_relation_data)
            else:
                positives, negatives = construct_hard_triplets(outputs, origin_labels, new_relation_data)

            # print(f"construct triplets time: {time.time()-start_time}")
            start_time = time.time()

            # for _ in range(config.f_pass):
            #     output, output_embedding = dropout_layer(reps)
            #     logits = classifier(output)
            #     logits_all.append(logits)
            batch_reps = reps.unsqueeze(0).expand(config.f_pass, -1, -1)  # Shape: [f_pass, batch_size, hidden_dim]

            # Apply dropout to all copies at once
            batch_outputs, batch_embeddings = dropout_layer(batch_reps)  # Shape: [f_pass, batch_size, hidden_dim]

            # Get logits for all passes in one computation
            logits_all = classifier(batch_outputs)  # Shape: [f_pass, batch_size, num_classes]

            positives = torch.cat(positives, 0).to(config.device)
            negatives = torch.cat(negatives, 0).to(config.device)
            anchors = outputs
            m_labels = labels.expand((config.f_pass, labels.shape[0]))  # m,B

            loss1: torch.Tensor = criterion(logits_all.reshape(-1, logits_all.shape[-1]), m_labels.reshape(-1))
            loss2 = compute_jsd_loss(logits_all)
            tri_loss = triplet_loss(anchors, positives, negatives)
            loss = loss1.mean() + loss2 + tri_loss

            # print(f"compute loss time: {time.time()-start_time}")
            start_time = time.time()

            if prev_encoder is not None:
                prev_reps = prev_encoder(tokens).detach()
                normalized_prev_reps_emb = F.normalize(prev_reps.view(-1, prev_reps.size()[1]), p=2, dim=1)

                feature_distill_loss = distill_criterion(normalized_reps_emb, normalized_prev_reps_emb,
                                                         torch.ones(tokens.size(0)).to(
                                                             config.device))
                loss += feature_distill_loss
            
            # print(f"compute feature distill loss time: {time.time()-start_time}")
            start_time = time.time()

            if prev_dropout_layer is not None and prev_classifier is not None:
                # Batch process all dropout operations
                # batch_reps = reps.unsqueeze(0).expand(config.f_pass, -1, -1)  # [f_pass, batch_size, hidden_dim]
                # outputs, _ = dropout_layer(batch_reps)  # [f_pass, batch_size, hidden_dim]
                prev_outputs, _ = prev_dropout_layer(batch_reps)  # [f_pass, batch_size, hidden_dim]

                # Compute all logits at once
                pre_logits = prev_classifier(batch_outputs)  # [f_pass, batch_size, num_classes]
                pre_logits = pre_logits.index_select(2, prev_relation_index)  # Select relevant indices
                pre_logits = F.softmax(pre_logits / T, dim=2).detach()

                # Compute log logits for all passes
                log_logits = F.log_softmax(logits_all.index_select(2, prev_relation_index) / T, dim=2)

                # Compute prediction distillation loss for all passes at once
                prediction_distill_loss = -torch.mean(torch.sum(pre_logits * log_logits, dim=2))

                # Add to total loss
                loss += prediction_distill_loss

                # Compute mean outputs
                mean_dropout_output = batch_outputs.mean(dim=0)  # [batch_size, hidden_dim]
                mean_prev_dropout_output = prev_outputs.mean(dim=0)  # [batch_size, hidden_dim]

                # Normalize outputs
                normalized_output = F.normalize(mean_dropout_output, p=2, dim=1)
                normalized_prev_output = F.normalize(mean_prev_dropout_output, p=2, dim=1)

                # Compute hidden distillation loss
                hidden_distill_loss = distill_criterion(
                    normalized_output, 
                    normalized_prev_output,
                    torch.ones(tokens.size(0)).to(config.device)
                )
                loss += hidden_distill_loss

            # print(f"compute hidden distill loss time: {time.time()-start_time}")

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(f"loss is {np.array(losses).mean()}")




def batch2device(batch_tuple, device):
    ans = []
    for var in batch_tuple:
        if isinstance(var, torch.Tensor):
            ans.append(var.to(device))
        elif isinstance(var, list):
            ans.append(batch2device(var))
        elif isinstance(var, tuple):
            ans.append(tuple(batch2device(var)))
        else:
            ans.append(var)
    return ans

def f1(preds, labels, na_tempid):
    unique_labels = set(preds + labels)
    if na_tempid in unique_labels:
        unique_labels.remove(na_tempid)
    
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


def evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data, seen_relations, map_relid2tempid):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    dropout_layer.eval()
    classifier.eval()
    n = len(test_data)

    # find tempid for na_id
    na_tempid = map_relid2tempid[config.na_id]
    print(f"na_tempid: {na_tempid}")
    
    total_labels = []
    total_preds = []

    for batch_data in tqdm(data_loader):
        labels, _, tokens = batch_data
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)
        total_labels.extend(labels.cpu().tolist())

        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        reps, _ = dropout_layer(reps)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()

        # get prediction
        preds = np.argmax(seen_sim, axis=1)
        total_preds.extend(preds)

    print("total_preds: ", total_preds)
    print("total_labels: ", total_labels)

    return f1(total_preds, total_labels, na_tempid)


def select_data(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for batch_data in tqdm(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = dropout_layer(encoder(tokens))[1].cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(relation_dataset))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    memory = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = relation_dataset[sel_index]
        memory.append(instance)
    return memory


def get_proto(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    dropout_layer.eval()
    for batch_data in tqdm(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = dropout_layer(encoder(tokens))[1]
        features.append(feature)
    features = torch.cat(features, dim=0)
    proto = torch.mean(features, dim=0, keepdim=True).cpu()
    standard = torch.sqrt(torch.var(features, dim=0)).cpu()
    return proto, standard


def generate_relation_data(protos, relation_standard):
    relation_data = {}
    relation_sample_nums = 10
    for id in protos.keys():
        relation_data[id] = []
        difference = np.random.normal(loc=0, scale=1, size=relation_sample_nums)
        for diff in difference:
            relation_data[id].append(protos[id] + diff * relation_standard[id])
    return relation_data


def generate_current_relation_data(config, encoder, dropout_layer, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    relation_data = []
    encoder.eval()
    dropout_layer.eval()
    for batch_data in tqdm(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = dropout_layer(encoder(tokens))[1].cpu()
        relation_data.append(feature)
    return relation_data

from transformers import  BertTokenizer
def data_augmentation(config, encoder, train_data, prev_train_data):
    expanded_train_data = train_data[:]
    expanded_prev_train_data = prev_train_data[:]
    encoder.eval()
    all_data = train_data + prev_train_data
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    entity_index = []
    entity_mention = []
    for sample in all_data:
        e11 = sample['tokens'].index(30522)
        e12 = sample['tokens'].index(30523)
        e21 = sample['tokens'].index(30524)
        e22 = sample['tokens'].index(30525)
        entity_index.append([e11,e12])
        entity_mention.append(sample['tokens'][e11+1:e12])
        entity_index.append([e21,e22])
        entity_mention.append(sample['tokens'][e21+1:e22])

    data_loader = get_data_loader(config, all_data, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    for step, batch_data in enumerate(data_loader):
        labels, _, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        feature1, feature2 = torch.split(feature, [config.encoder_output_size,config.encoder_output_size], dim=1)
        features.append(feature1)
        features.append(feature2)
    features = torch.cat(features, dim=0)
    # similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    similarity_matrix = []
    for i in range(len(features)):
        similarity_matrix.append([0]*len(features))

    for i in range(len(features)):
        for j in range(i,len(features)):
            similarity = F.cosine_similarity(features[i],features[j],dim=0)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    similarity_matrix = torch.tensor(similarity_matrix).to(config.device)
    zero = torch.zeros_like(similarity_matrix).to(config.device)
    diag = torch.diag_embed(torch.diag(similarity_matrix))
    similarity_matrix -= diag
    similarity_matrix = torch.where(similarity_matrix<0.95, zero, similarity_matrix)
    nonzero_index = torch.nonzero(similarity_matrix)
    expanded_train_count = 0

    for origin, replace in nonzero_index:
        sample_index = int(origin/2)
        sample = all_data[sample_index]
        if entity_mention[origin] == entity_mention[replace]:
            continue
        new_tokens = sample['tokens'][:entity_index[origin][0]+1] + entity_mention[replace] + sample['tokens'][entity_index[origin][1]:]
        if len(new_tokens) < config.max_length:
            new_tokens = new_tokens + [0]*(config.max_length-len(new_tokens))
        else:
            new_tokens = new_tokens[:config.max_length]

        new_sample = {
            'relation': sample['relation'],
            'neg_labels': sample['neg_labels'],
            'tokens': new_tokens
        }
        if sample_index < len(train_data) and expanded_train_count < 5 * len(train_data):
            expanded_train_data.append(new_sample)
            expanded_train_count += 1
        else:
            expanded_prev_train_data.append(new_sample)
    return expanded_train_data, expanded_prev_train_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="tacred", type=str)
    parser.add_argument("--shot", default=10, type=int)
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    config.task = args.task
    config.shot = args.shot
    config.step1_epochs = 5
    config.step2_epochs = 15
    config.step3_epochs = 20
    config.temperature = 0.08

    if config.task == "FewRel":
        config.relation_file = "data/fewrel/relation_name.txt"
        config.rel_index = "data/fewrel/rel_index.npy"
        config.rel_feature = "data/fewrel/rel_feature.npy"
        config.rel_des_file = "data/fewrel/relation_description.txt"
        config.na_id = 80
        if config.shot == 5:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_5/test_0.txt"
        elif config.shot == 10:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_10/test_0.txt"
        else:
            config.rel_cluster_label = "data/fewrel/CFRLdata_10_100_10_2/rel_cluster_label_0.npy"
            config.training_file = "data/fewrel/CFRLdata_10_100_10_2/train_0.txt"
            config.valid_file = "data/fewrel/CFRLdata_10_100_10_2/valid_0.txt"
            config.test_file = "data/fewrel/CFRLdata_10_100_10_2/test_0.txt"
    else:
        config.relation_file = "data/tacred/relation_name.txt"
        config.rel_index = "data/tacred/rel_index.npy"
        config.rel_feature = "data/tacred/rel_feature.npy"
        config.na_id = 41
        if config.shot == 5:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_5/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_5/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_5/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_5/test_0.txt"
        else:
            config.rel_cluster_label = "data/tacred/CFRLdata_10_100_10_10/rel_cluster_label_0.npy"
            config.training_file = "data/tacred/CFRLdata_10_100_10_10/train_0.txt"
            config.valid_file = "data/tacred/CFRLdata_10_100_10_10/valid_0.txt"
            config.test_file = "data/tacred/CFRLdata_10_100_10_10/test_0.txt"

    result_cur_test = []
    result_whole_test_wo_na = []
    result_whole_test_w_na = []

    for rou in range(config.total_round):
        test_cur = []
        test_total_wo_na = []
        test_total_w_na = []
        random.seed(config.seed+rou*100)
        sampler = data_sampler(config=config, seed=config.seed+rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id

        encoder = Bert_Encoder(config=config).to(config.device)
        dropout_layer = Dropout_Layer(config=config).to(config.device)
        num_class = len(sampler.id2rel)

        memorized_samples = {}
        memory = collections.defaultdict(list)
        history_relations = []
        history_data = []
        prev_relations = []
        classifier = None
        prev_classifier = None
        prev_encoder = None
        prev_dropout_layer = None
        relation_standard = {}
        forward_accs = []
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            print(seen_relations)

            prev_relations = history_relations[:]
            train_data_for_initial = []
            count = 0
            for relation in current_relations:
                if relation not in history_relations:
                    history_relations.append(relation)
                train_data_for_initial += training_data[relation]


            temp_rel2id = [rel2id[x] for x in seen_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            prev_relation_index = []
            prev_samples = []
            for relation in prev_relations:
                prev_relation_index.append(map_relid2tempid[rel2id[relation]])
                prev_samples += memorized_samples[relation]
            prev_relation_index = torch.tensor(prev_relation_index).to(config.device)

            classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(history_relations)).to(
                config.device)

            temp_protos = {}
            for relation in current_relations:
                proto, _ = get_proto(config, encoder, dropout_layer, training_data[relation])
                temp_protos[rel2id[relation]] = proto

            for relation in prev_relations:
                proto, _ = get_proto(config, encoder, dropout_layer, memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto


            train_simple_model(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step1_epochs, map_relid2tempid)
            print(f"simple finished")


            temp_protos = {}

            for relation in current_relations:
                proto, standard = get_proto(config,encoder,dropout_layer,training_data[relation])
                temp_protos[rel2id[relation]] = proto
                relation_standard[rel2id[relation]] = standard


            for relation in prev_relations:
                proto, _ = get_proto(config,encoder,dropout_layer,memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto

            new_relation_data = generate_relation_data(temp_protos, relation_standard)

            for relation in current_relations:
                new_relation_data[rel2id[relation]].extend(generate_current_relation_data(config, encoder,dropout_layer,training_data[relation]))

            torch.cuda.empty_cache()
            print(len(train_data_for_initial))


            print(f"start training memory model")

            train_mem_model(config, encoder, dropout_layer, classifier, train_data_for_initial, config.step2_epochs, map_relid2tempid, new_relation_data,
                        prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index)
            print(f"first finished")

            for relation in current_relations:
                memorized_samples[relation] = select_data(config, encoder, dropout_layer, training_data[relation])
                memory[rel2id[relation]] = select_data(config, encoder, dropout_layer, training_data[relation])

            train_data_for_memory = []
            # train_data_for_memory += expanded_prev_samples
            train_data_for_memory += prev_samples
            for relation in current_relations:
                train_data_for_memory += memorized_samples[relation]
            print(len(seen_relations))
            print(len(train_data_for_memory))

            temp_protos = {}
            for relation in seen_relations:
                proto, _ = get_proto(config, encoder, dropout_layer, memorized_samples[relation])
                temp_protos[rel2id[relation]] = proto

            train_mem_model(config, encoder, dropout_layer, classifier, train_data_for_memory, config.step3_epochs, map_relid2tempid, new_relation_data,
                        prev_encoder, prev_dropout_layer, prev_classifier, prev_relation_index)
            print(f"memory finished")

            # wo na
            test_data_2_wo_na = []
            for relation in seen_relations:
                if relation != id2rel[config.na_id]:
                    test_data_2_wo_na += historic_test_data[relation]

            total_acc_wo_na = evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data_2_wo_na, seen_relations, map_relid2tempid)

            # w na
            test_data_2_w_na = []
            for relation in seen_relations:
                test_data_2_w_na += historic_test_data[relation]

            total_acc_w_na = evaluate_strict_model(config, encoder, dropout_layer, classifier, test_data_2_w_na, seen_relations, map_relid2tempid)

            print(f'Restart Num {rou + 1}')
            print(f'task--{steps + 1}:')
            print(f'history test acc wo na:{total_acc_wo_na}')
            print(f'history test acc w na:{total_acc_w_na}')

            test_total_wo_na.append(total_acc_wo_na)
            print("test_total_wo_na: ", test_total_wo_na)

            test_total_w_na.append(total_acc_w_na)
            print("test_total_w_na: ", test_total_w_na)

            prev_encoder = deepcopy(encoder)
            prev_dropout_layer = deepcopy(dropout_layer)
            prev_classifier = deepcopy(classifier)
            torch.cuda.empty_cache()
            
        # wo na
        result_whole_test_wo_na.append(np.array(test_total_wo_na)*100)
        print("result_whole_test wo na: ", result_whole_test_wo_na)
        avg_result_all_test_wo_na = np.average(result_whole_test_wo_na, 0)

        print("avg_result_all_test wo na: ", avg_result_all_test_wo_na)
        std_result_all_test_wo_na = np.std(result_whole_test_wo_na, 0)
        print("std_result_all_test wo na: ", std_result_all_test_wo_na)

        # w na
        result_whole_test_w_na.append(np.array(test_total_w_na)*100)
        print("result_whole_test w na: ", result_whole_test_w_na)
        avg_result_all_test_w_na = np.average(result_whole_test_w_na, 0)

        print("avg_result_all_test w na: ", avg_result_all_test_w_na)
        std_result_all_test_w_na = np.std(result_whole_test_w_na, 0)
        print("std_result_all_test w na: ", std_result_all_test_w_na)



