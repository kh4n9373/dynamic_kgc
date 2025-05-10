from torch.utils.data import Dataset
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os 
import numpy as np

class RelationDataset(Dataset):
    def __init__(self, sentences, head_entities, tail_entities, relations):
        """
        Initialize the dataset.
        
        Args:
            sentences (list): List of sentences
            head_entities (list): List of head entities
            tail_entities (list): List of tail entities
            relations (list): List of relation labels (integers)
        """
        assert len(sentences) == len(head_entities) == len(tail_entities) == len(relations)
        self.sentences = sentences
        self.head_entities = head_entities
        self.tail_entities = tail_entities
        self.relations = relations
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "head_entity": self.head_entities[idx],
            "tail_entity": self.tail_entities[idx],
            "relation": self.relations[idx]
        }


def load_data(csv_path, train_test_split_ratio=0.8, save_split_data=True):
    """
    Load and process WebNLG data from CSV file.
    Only include records where NOTA is 'false'.
    Split data into train and test sets using stratified sampling.
    Optionally save the split datasets to CSV files.
    
    Args:
        csv_path: Path to the WebNLG CSV file
        train_test_split_ratio: Ratio for train/test split (default: 0.8)
        save_split_data: Whether to save the split data to CSV files (default: True)
        
    Returns:
        tuple: (train_data, test_data, relation_to_id)
              train_data and test_data are dictionaries with keys:
              'sentences', 'head_entities', 'tail_entities', 'relations'
    """
    df = pd.read_csv(csv_path)
    
    filtered_df = df[df['NOTA'] == 'false']
    
    sentences = filtered_df['sentence'].tolist()
    head_entities = filtered_df['extracted_subject'].tolist()
    tail_entities = filtered_df['extracted_object'].tolist()
    relation_names = filtered_df['relation'].tolist()
    
    unique_relations = sorted(set(relation_names))
    relation_to_id = {rel: idx for idx, rel in enumerate(unique_relations)}
    
    id_to_relation = {idx: rel for rel, idx in relation_to_id.items()}
    
    relations = [relation_to_id[rel] for rel in relation_names]
    
    data_df = pd.DataFrame({
        'sentence': sentences,
        'head_entity': head_entities,
        'tail_entity': tail_entities,
        'relation_name': relation_names,
        'relation_id': relations
    })
    
    relation_counts = data_df['relation_name'].value_counts()
    print("Relation class distribution:")
    for rel, count in relation_counts.items():
        print(f"  {rel}: {count} examples")
  
    classes_with_one_sample = relation_counts[relation_counts == 1].index.tolist()
    if classes_with_one_sample:
        print(f"Warning: {len(classes_with_one_sample)} relations have only one sample and will be placed in training set")
        
        single_sample_df = data_df[data_df['relation_name'].isin(classes_with_one_sample)]
        multi_sample_df = data_df[~data_df['relation_name'].isin(classes_with_one_sample)]
        
        if len(multi_sample_df) > 0:
            try:
                multi_relation_counts = multi_sample_df['relation_name'].value_counts()
                if any(multi_relation_counts < 5):  # Need at least a few samples per class
                    print("Some classes have too few samples for stratified split. Using random split instead.")
                    train_multi, test_multi = train_test_split(
                        multi_sample_df, 
                        test_size=(1-train_test_split_ratio),
                        random_state=42
                    )
                else:
                    train_multi, test_multi = train_test_split(
                        multi_sample_df, 
                        test_size=(1-train_test_split_ratio),
                        random_state=42,
                        stratify=multi_sample_df['relation_name']
                    )
            except ValueError as e:
                print(f"Stratified split failed: {e}. Using random split instead.")
                train_multi, test_multi = train_test_split(
                    multi_sample_df, 
                    test_size=(1-train_test_split_ratio),
                    random_state=42
                )
            
            train_df = pd.concat([train_multi, single_sample_df])
            test_df = test_multi
        else:
            train_df = single_sample_df
            test_df = pd.DataFrame(columns=data_df.columns)
    else:
        try:
            train_df, test_df = train_test_split(
                data_df, 
                test_size=(1-train_test_split_ratio),
                random_state=42,
                stratify=data_df['relation_name']
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}. Using random split instead.")
            train_df, test_df = train_test_split(
                data_df, 
                test_size=(1-train_test_split_ratio),
                random_state=42
            )
    
    print(f"\nSplit statistics:")
    print(f"  Total examples: {len(data_df)}")
    print(f"  Training examples: {len(train_df)} ({len(train_df)/len(data_df)*100:.1f}%)")
    print(f"  Testing examples: {len(test_df)} ({len(test_df)/len(data_df)*100:.1f}%)")
    
    if save_split_data:
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = os.path.join("training_data", csv_basename)
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Lưu mapping từ relation ID sang relation name
        relation_mapping_path = os.path.join(output_dir, "relation_mapping.json")
        with open(relation_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(id_to_relation, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved split datasets to:")
        print(f"  Train: {train_path}")
        print(f"  Test: {test_path}")
        print(f"  Relation mapping: {relation_mapping_path}")
    
    train_data = {
        'sentences': train_df['sentence'].tolist(),
        'head_entities': train_df['head_entity'].tolist(),
        'tail_entities': train_df['tail_entity'].tolist(),
        'relations': train_df['relation_id'].tolist()
    }
    
    test_data = {
        'sentences': test_df['sentence'].tolist(),
        'head_entities': test_df['head_entity'].tolist(),
        'tail_entities': test_df['tail_entity'].tolist(),
        'relations': test_df['relation_id'].tolist()
    }
    
    if len(test_data['sentences']) == 0:
        print("Warning: Test set is empty! Using 20% of training data as test set.")
        train_size = int(0.8 * len(train_data['sentences']))
        
        indices = list(range(len(train_data['sentences'])))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        new_train_data = {
            'sentences': [train_data['sentences'][i] for i in train_indices],
            'head_entities': [train_data['head_entities'][i] for i in train_indices],
            'tail_entities': [train_data['tail_entities'][i] for i in train_indices],
            'relations': [train_data['relations'][i] for i in train_indices]
        }
        
        new_test_data = {
            'sentences': [train_data['sentences'][i] for i in test_indices],
            'head_entities': [train_data['head_entities'][i] for i in test_indices],
            'tail_entities': [train_data['tail_entities'][i] for i in test_indices],
            'relations': [train_data['relations'][i] for i in test_indices]
        }
        
        train_data = new_train_data
        test_data = new_test_data
        
        print(f"  New training examples: {len(train_data['sentences'])}")
        print(f"  New testing examples: {len(test_data['sentences'])}")
        
        if save_split_data:
            train_rows = []
            for i in range(len(train_data['sentences'])):
                row = {
                    'sentence': train_data['sentences'][i],
                    'head_entity': train_data['head_entities'][i],
                    'tail_entity': train_data['tail_entities'][i],
                    'relation_id': train_data['relations'][i]
                }
                relation_name = id_to_relation[train_data['relations'][i]]
                row['relation_name'] = relation_name
                train_rows.append(row)
            
            test_rows = []
            for i in range(len(test_data['sentences'])):
                row = {
                    'sentence': test_data['sentences'][i],
                    'head_entity': test_data['head_entities'][i],
                    'tail_entity': test_data['tail_entities'][i],
                    'relation_id': test_data['relations'][i]
                }
                relation_name = id_to_relation[test_data['relations'][i]]
                row['relation_name'] = relation_name
                test_rows.append(row)
            
            new_train_df = pd.DataFrame(train_rows)
            new_test_df = pd.DataFrame(test_rows)
            

            csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
            output_dir = os.path.join("training_data", csv_basename)
            
            train_path = os.path.join(output_dir, "train.csv")
            test_path = os.path.join(output_dir, "test.csv")
            
            new_train_df.to_csv(train_path, index=False)
            new_test_df.to_csv(test_path, index=False)

    return train_data, test_data, relation_to_id
