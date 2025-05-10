import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.manifold import TSNE
from encoder import RelationExtractor

def load_model(model_path, num_relations):
    model = RelationExtractor(num_relations=num_relations)
    
    checkpoint = torch.load(model_path, map_location=model.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    return model


def load_relation_mapping(mapping_path):
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            relation_mapping = json.load(f)
            return {int(k): v for k, v in relation_mapping.items()}
    except Exception as e:
        print(f"Error loading relation mapping: {e}")
        return {}


def load_test_examples(test_csv_path):

    try:
        test_df = pd.read_csv(test_csv_path)
        test_examples = []
        
        for _, row in test_df.iterrows():
            if pd.isna(row['sentence']) or pd.isna(row['head_entity']) or pd.isna(row['tail_entity']) or pd.isna(row['relation_name']):
                continue  
                
            example = {
                "sentence": row['sentence'],
                "head_entity": row['head_entity'],
                "tail_entity": row['tail_entity'],
                "expected_relation": row['relation_name'],
                "relation_id": int(row['relation_id']) if not pd.isna(row['relation_id']) else None
            }
            test_examples.append(example)
        
        return test_examples
    except Exception as e:
        print(f"Error loading test examples: {e}")
        return []


def visualize_prediction_confidence(relation_names, predictions, output_file='prediction_confidence.png'):
    plt.figure(figsize=(10, 6))
    relation_ids = list(predictions.keys())
    confidence_scores = list(predictions.values())
    
    relation_labels = [relation_names.get(rel_id, f"Unknown {rel_id}") for rel_id in relation_ids]
    
    sorted_indices = np.argsort(confidence_scores)[::-1]  # Sort in descending order
    sorted_labels = [relation_labels[i] for i in sorted_indices]
    sorted_scores = [confidence_scores[i] for i in sorted_indices]
    
    colors = ['#1f77b4' if score == max(confidence_scores) else '#d3d3d3' for score in sorted_scores]
    
    bars = plt.bar(range(len(sorted_scores)), sorted_scores, color=colors)
    plt.xticks(range(len(sorted_scores)), sorted_labels, rotation=45, ha='right')
    plt.title('Relation Prediction Confidence')
    plt.xlabel('Relation Type')
    plt.ylabel('Confidence Score')
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Confidence visualization saved as {output_file}")
    plt.close()


def visualize_embeddings(examples, model, relation_names, output_file='relation_embeddings.png'):
    embeddings = []
    relations = []
    
    for example in examples:
        sentence = example["sentence"]
        head_entity = example["head_entity"]
        tail_entity = example["tail_entity"]
        expected_relation = example["expected_relation"]
        
        embedding = model.get_latent_representation(sentence, head_entity, tail_entity)
        embeddings.append(embedding.cpu().detach().numpy())
        relations.append(expected_relation)
    
    if len(embeddings) > 1: 
        try:
            embeddings_array = np.array(embeddings)
            
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings_array)
            
            plt.figure(figsize=(10, 8))
            
            unique_relations = list(set(relations))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_relations)))
            relation_to_color = {rel: colors[i] for i, rel in enumerate(unique_relations)}
            
            for i, (x, y) in enumerate(embeddings_2d):
                relation = relations[i]
                plt.scatter(x, y, color=relation_to_color[relation], label=relation)
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='best')
            
            plt.title('Relation Embeddings Visualization (t-SNE)')
            plt.savefig(output_file)
            print(f"Embeddings visualization saved as {output_file}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create embeddings visualization: {e}")
    else:
        print("Not enough examples to create t-SNE visualization (need at least 2)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test relation extraction model')
    parser.add_argument('--dataset_name', type=str, default="webnlg_balanced",
                      help='Name of the dataset folder under training_data/')
    parser.add_argument('--model_path', type=str, 
                      help='Path to the trained model file. If not specified, uses models/{dataset_name}/model_epoch_1.pt')
    parser.add_argument('--mapping_path', type=str,
                      help='Path to relation mapping JSON file. If not specified, uses training_data/{dataset_name}/relation_mapping.json')
    parser.add_argument('--test_csv_path', type=str,
                      help='Path to test CSV file. If not specified, uses training_data/{dataset_name}/test.csv')
    parser.add_argument('--output_dir', type=str,
                      help='Directory to save test results. If not specified, uses training_process/{dataset_name}/test_results')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    mapping_path = args.mapping_path or f"training_data/{dataset_name}/relation_mapping.json"
    test_csv_path = args.test_csv_path or f"training_data/{dataset_name}/test.csv"
    model_path = args.model_path or f"models/{dataset_name}/model_epoch_5.pt"
    
    vis_output_dir = args.output_dir or os.path.join("training_process", dataset_name, "test_results")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    print(f"Visualizations will be saved to: {vis_output_dir}")
    
    relation_names = load_relation_mapping(mapping_path)
    num_relations = len(relation_names)
    print(f"Loaded {num_relations} relations from mapping file")
    
    test_examples = load_test_examples(test_csv_path)
    print(f"Loaded {len(test_examples)} test examples")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using an uninitialized model for demonstration.")
        model = RelationExtractor(num_relations=num_relations)
        model.eval()
    else:
        model = load_model(model_path, num_relations=num_relations)
    
    if not test_examples:
        print("No test examples found in CSV. Using built-in examples.")
        test_examples = [
            {
                "sentence": "Albert Einstein was born in Ulm, Germany.",
                "head_entity": "Albert Einstein",
                "tail_entity": "Ulm, Germany",
                "expected_relation": "birthPlace"
            },
            {
                "sentence": "The Great Wall is located in China.",
                "head_entity": "Great Wall",
                "tail_entity": "China",
                "expected_relation": "location"
            }
        ]
    
    print("\n=== Testing Relation Extraction ===")
    
    all_predictions = []
    all_expected = []
    all_confidences = {}
    test_results = []
    
    for i, example in enumerate(test_examples):
        sentence = example["sentence"]
        head_entity = example["head_entity"]
        tail_entity = example["tail_entity"]
        expected = example["expected_relation"]
        
        print(f"\nSentence: {sentence}")
        print(f"Head Entity: {head_entity}")
        print(f"Tail Entity: {tail_entity}")
        print(f"Expected Relation: {expected}")
        
        try:
            result = model.predict_relation(sentence, head_entity, tail_entity)
            
            pred_relation_id = result["relation_id"]
            pred_relation = relation_names.get(pred_relation_id, f"Unknown relation ({pred_relation_id})")
            confidence = result["probabilities"][pred_relation_id]
            
            all_predictions.append(pred_relation)
            all_expected.append(expected)
            
            test_results.append({
                "id": i,
                "sentence": sentence,
                "head_entity": head_entity,
                "tail_entity": tail_entity,
                "expected_relation": expected,
                "predicted_relation": pred_relation,
                "confidence": float(confidence),
                "correct": pred_relation == expected
            })
            
            print(f"Predicted Relation: {pred_relation} (ID: {pred_relation_id})")
            print(f"Confidence: {confidence:.4f}")
            
            print("All probabilities:")
            probabilities_dict = {}
            for rel_id, prob in enumerate(result["probabilities"]):
                if rel_id in relation_names:
                    rel_name = relation_names[rel_id]
                    print(f"  - {rel_name}: {prob:.4f}")
                    probabilities_dict[rel_id] = prob
            
            if len(probabilities_dict) > 0:
                conf_filename = os.path.join(vis_output_dir, f"confidence_example_{i+1}.png")
                visualize_prediction_confidence(relation_names, probabilities_dict, conf_filename)
                
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    if len(all_predictions) > 0:
        accuracy = sum(1 for pred, exp in zip(all_predictions, all_expected) if pred == exp) / len(all_predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        results_file = os.path.join(vis_output_dir, "test_results.json")
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"Detailed test results saved to {results_file}")
    
    if len(test_examples) > 0:
        embeddings_file = os.path.join(vis_output_dir, "embeddings_visualization.png")
        visualize_embeddings(test_examples, model, relation_names, embeddings_file)
    
    if len(all_predictions) > 1:
        from sklearn.metrics import confusion_matrix
        
        unique_relations = sorted(list(set(all_expected + all_predictions)))
        cm = confusion_matrix(
            all_expected, 
            all_predictions, 
            labels=unique_relations
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_relations,
                   yticklabels=unique_relations)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        confusion_file = os.path.join(vis_output_dir, "confusion_matrix.png")
        plt.savefig(confusion_file)
        print(f"Confusion matrix saved as {confusion_file}")
        plt.close()
    
    if len(test_examples) >= 2:
        print("\n=== Example of Comparing Two Sentences ===")
        
        if len(test_examples) > 1:
            num_examples = min(len(test_examples), 5)  # Limit to 5 to avoid too large matrices
            similarity_matrix = np.zeros((num_examples, num_examples))
            example_labels = []
            
            for i in range(num_examples):
                example_i = test_examples[i]
                sent_i = example_i["sentence"]
                head_i = example_i["head_entity"]
                tail_i = example_i["tail_entity"]
                relation_i = example_i["expected_relation"]
                
                z_i = model.get_latent_representation(sent_i, head_i, tail_i)
                example_labels.append(f"{relation_i}: {head_i}-{tail_i}")
                
                for j in range(num_examples):
                    example_j = test_examples[j]
                    sent_j = example_j["sentence"]
                    head_j = example_j["head_entity"]
                    tail_j = example_j["tail_entity"]
                    
                    z_j = model.get_latent_representation(sent_j, head_j, tail_j)
                    similarity = model.get_similarity(z_i, z_j)
                    similarity_matrix[i, j] = similarity
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(similarity_matrix, annot=True, fmt='.4f', cmap='viridis',
                       xticklabels=example_labels,
                       yticklabels=example_labels)
            plt.title('Sentence Pair Similarities')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            similarity_file = os.path.join(vis_output_dir, "sentence_similarities.png")
            plt.savefig(similarity_file)
            print(f"Sentence similarity matrix saved as {similarity_file}")
            plt.close()
        
        sent1 = test_examples[0]["sentence"]
        head1 = test_examples[0]["head_entity"]
        tail1 = test_examples[0]["tail_entity"]
        rel1 = test_examples[0]["expected_relation"]
        
        sent2 = test_examples[1]["sentence"] if len(test_examples) > 1 else sent1
        head2 = test_examples[1]["head_entity"] if len(test_examples) > 1 else head1
        tail2 = test_examples[1]["tail_entity"] if len(test_examples) > 1 else tail1
        rel2 = test_examples[1]["expected_relation"] if len(test_examples) > 1 else rel1
        
        print(f"Comparing:\n1. {sent1}\n2. {sent2}")
        
        z1 = model.get_latent_representation(sent1, head1, tail1)
        z2 = model.get_latent_representation(sent2, head2, tail2)
        
        similarity = model.get_similarity(z1, z2)
        print(f"Similarity score: {similarity:.4f}")
        
        # Visualize similarity as a heatmap
        examples = [
            f"{rel1}: {head1}-{tail1}",
            f"{rel2}: {head2}-{tail2}"
        ]
        similarity_matrix = np.array([[1.0, similarity], 
                                     [similarity, 1.0]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, annot=True, fmt='.4f', cmap='viridis',
                   xticklabels=examples,
                   yticklabels=examples)
        plt.title('Sentence Pair Similarity')
        plt.tight_layout()
        
        pair_similarity_file = os.path.join(vis_output_dir, "pair_similarity.png")
        plt.savefig(pair_similarity_file)
        print(f"Pair similarity visualization saved as {pair_similarity_file}")
        plt.close()
        
        if rel1 == rel2:
            print(f"Both sentences represent a '{rel1}' relation, so a high similarity is expected.")
        else:
            print(f"These sentences represent different relations ('{rel1}' vs '{rel2}'), so a lower similarity is expected.")
    else:
        print("\nNot enough test examples to perform sentence comparison.")
