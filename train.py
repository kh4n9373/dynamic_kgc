import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from trainer import train_relation_extractor, evaluate_model
from load_dataset import RelationDataset, load_data
from encoder import RelationExtractor
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE


def visualize_data_distribution(relation_to_id, train_data, test_data, output_file='data_distribution.png'):
    train_relations = train_data['relations']
    test_relations = test_data['relations']
    train_counts = {}
    test_counts = {}
    
    for rel_id in train_relations:
        if rel_id in train_counts:
            train_counts[rel_id] += 1
        else:
            train_counts[rel_id] = 1
    
    for rel_id in test_relations:
        if rel_id in test_counts:
            test_counts[rel_id] += 1
        else:
            test_counts[rel_id] = 1
    
    relation_names = []
    train_values = []
    test_values = []
    
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    for rel_id in sorted(id_to_relation.keys()):
        relation_names.append(id_to_relation[rel_id])
        train_values.append(train_counts.get(rel_id, 0))
        test_values.append(test_counts.get(rel_id, 0))
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(relation_names))
    width = 0.35
    
    plt.bar(x - width/2, train_values, width, label='Train')
    plt.bar(x + width/2, test_values, width, label='Test')
    
    plt.xlabel('Relation Types')
    plt.ylabel('Number of Examples')
    plt.title('Data Distribution across Relations')
    plt.xticks(x, relation_names, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Data distribution visualization saved as {output_file}")
    plt.close()


def visualize_training_metrics(metrics_dict, output_prefix='training'):
    epochs = list(range(1, len(metrics_dict['train_loss']) + 1))
    train_loss = metrics_dict['train_loss']
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics_dict['train_loss'], 'b-', marker='o', label='Total Loss')
    if 'contrastive_loss' in metrics_dict:
        plt.plot(epochs, metrics_dict['contrastive_loss'], 'g-', marker='s', label='Contrastive Loss')
    if 'classification_loss' in metrics_dict:
        plt.plot(epochs, metrics_dict['classification_loss'], 'r-', marker='^', label='Classification Loss')
    
    plt.title('Training Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_losses.png')
    print(f"Training losses plot saved as {output_prefix}_losses.png")
    plt.close()
    
    if 'val_f1' in metrics_dict and len(metrics_dict['val_f1']) > 0:
        plt.figure(figsize=(12, 6))
        
        metrics_to_plot = [
            ('val_accuracy', 'Accuracy', 'b-', 'o'),
            ('val_f1', 'F1 (weighted)', 'g-', 's'),
            ('val_f1_macro', 'F1 (macro)', 'r-', '^'),
            ('val_precision', 'Precision', 'c-', 'D'),
            ('val_recall', 'Recall', 'm-', 'x')
        ]
        
        for metric_key, label, line_style, marker in metrics_to_plot:
            if metric_key in metrics_dict:
                plt.plot(epochs, metrics_dict[metric_key], line_style, marker=marker, label=label)
        
        plt.title('Validation Metrics over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_validation_metrics.png')
        print(f"Validation metrics plot saved as {output_prefix}_validation_metrics.png")
        plt.close()
    
    if 'learning_rate' in metrics_dict:
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, metrics_dict['learning_rate'], 'b-', marker='o')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_learning_rate.png')
        print(f"Learning rate plot saved as {output_prefix}_learning_rate.png")
        plt.close()


def visualize_confusion_matrix(all_preds, all_labels, relation_to_id, output_file='confusion_matrix.png'):
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    unique_relations = sorted(list(set(all_labels)))
    relation_names = [id_to_relation.get(rel_id, f"Unknown {rel_id}") for rel_id in unique_relations]
    
    cm = confusion_matrix(all_labels, all_preds, labels=unique_relations)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=relation_names,
                yticklabels=relation_names)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Confusion matrix saved as {output_file}")
    plt.close()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=relation_names,
                yticklabels=relation_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file_parts = output_file.split('.')
    output_file_normalized = f"{output_file_parts[0]}_normalized.{output_file_parts[1]}"
    plt.savefig(output_file_normalized)
    print(f"Normalized confusion matrix saved as {output_file_normalized}")
    plt.close()


def visualize_embeddings(embeddings, labels, relation_to_id, output_file='embeddings_tsne.png'):
    if len(embeddings) == 0:
        return
        
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    label_names = [id_to_relation.get(label, f"Unknown {label}") for label in labels]
    unique_labels = sorted(list(set(labels)))
    
    embeddings_array = np.array(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    plt.figure(figsize=(12, 10))
    
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == label]
        plt.scatter(
            embeddings_2d[idx, 0], 
            embeddings_2d[idx, 1], 
            c=[cmap(i)], 
            label=id_to_relation.get(label, f"Unknown {label}"),
            alpha=0.7
        )
    
    plt.title('t-SNE Visualization of Relation Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Embeddings visualization saved as {output_file}")
    plt.close()


def training_callback(epoch, train_loss, val_metrics=None, callback_args=None):
    """Callback function to generate visualizations during training."""
    if callback_args is None or 'visualization_dir' not in callback_args or callback_args['visualization_dir'] is None:
        return
        
    vis_dir = callback_args['visualization_dir']
    epoch_num = epoch + 1
    
    if epoch_num % 3 == 0 or epoch_num == 1 or epoch_num == 10:  
        print(f"Creating visualizations for epoch {epoch_num}...")
        
        if val_metrics and 'embeddings' in val_metrics and len(val_metrics['embeddings']) > 0:
            if len(val_metrics['embeddings']) >= 2:
                try:
                    embeddings_output = os.path.join(vis_dir, f"embeddings_epoch_{epoch_num}.png")
                    visualize_embeddings(
                        val_metrics['embeddings'], 
                        val_metrics['ground_truth'], 
                        callback_args.get('relation_to_id', {}),
                        embeddings_output
                    )
                except Exception as e:
                    print(f"Warning: Could not create embeddings visualization: {e}")
            else:
                print(f"Warning: Not enough embeddings to create t-SNE visualization (need at least 2, got {len(val_metrics['embeddings'])})")
            
            if 'predictions' in val_metrics and 'ground_truth' in val_metrics:
                try:
                    confusion_output = os.path.join(vis_dir, f"confusion_matrix_epoch_{epoch_num}.png")
                    visualize_confusion_matrix(
                        val_metrics['predictions'],
                        val_metrics['ground_truth'],
                        callback_args.get('relation_to_id', {}),
                        confusion_output
                    )
                except Exception as e:
                    print(f"Warning: Could not create confusion matrix: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train relation extraction model')
    parser.add_argument('--csv_path', type=str, default="dataset_constructed/webnlg_balanced.csv",
                      help='Path to the input dataset CSV file')
    parser.add_argument('--model_name', type=str, default="bert-base-uncased",
                      help='Name of the pretrained BERT model')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate for optimizer')
    parser.add_argument('--train_split', type=float, default=0.8,
                      help='Train/test split ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Train/validation split ratio (from training data)')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                      help='Weight for contrastive loss')
    parser.add_argument('--classification_weight', type=float, default=1.0,
                      help='Weight for classification loss')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--save_best_only', action='store_true',
                      help='Only save the best model based on validation F1 score')
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    model_output_dir = os.path.join("models", dataset_name)
    vis_output_dir = os.path.join("training_process", dataset_name)
    
    os.makedirs(vis_output_dir, exist_ok=True)
    
    train_data, test_data, relation_to_id = load_data(csv_path, train_test_split_ratio=args.train_split, save_split_data=True)
    
    print(f"\nModel outputs will be saved to: {model_output_dir}")
    print(f"Visualizations will be saved to: {vis_output_dir}")
    
    data_dist_path = os.path.join(vis_output_dir, "data_distribution.png")
    visualize_data_distribution(relation_to_id, train_data, test_data, data_dist_path)
    
    print("\nData Examples (Training set):")
    for i in range(min(3, len(train_data['sentences']))):
        print(f"Example {i+1}:")
        print(f"  Sentence: {train_data['sentences'][i]}")
        print(f"  Head Entity: {train_data['head_entities'][i]}")
        print(f"  Tail Entity: {train_data['tail_entities'][i]}")
        print(f"  Relation ID: {train_data['relations'][i]}")
        relation_name = next(key for key, val in relation_to_id.items() if val == train_data['relations'][i])
        print(f"  Relation Name: {relation_name}")
        print()

    train_full_dataset = RelationDataset(
        train_data['sentences'], 
        train_data['head_entities'], 
        train_data['tail_entities'], 
        train_data['relations']
    )
    
    test_dataset = RelationDataset(
        test_data['sentences'], 
        test_data['head_entities'], 
        test_data['tail_entities'], 
        test_data['relations']
    )
    
    train_size = int((1.0 - args.val_split) * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size], 
        generator=generator
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    plt.figure(figsize=(8, 6))
    dataset_names = ['Train', 'Validation', 'Test']
    dataset_sizes = [len(train_dataset), len(val_dataset), len(test_dataset)]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    plt.bar(dataset_names, dataset_sizes, color=colors)
    plt.title('Dataset Split Sizes')
    plt.ylabel('Number of Examples')
    
    for i, v in enumerate(dataset_sizes):
        plt.text(i, v + 0.5, str(v), ha='center')
        
    plt.tight_layout()
    splits_path = os.path.join(vis_output_dir, "dataset_splits.png")
    plt.savefig(splits_path)
    print(f"Dataset split visualization saved as {splits_path}")
    plt.close()

    model = RelationExtractor(
        model_name=args.model_name,
        num_relations=len(relation_to_id), 
        freeze_bert=False 
    )

    callback_args = {
        'relation_to_id': relation_to_id,
        'visualization_dir': vis_output_dir
    }
    
    print(f"\nTraining model...")
    results = train_relation_extractor(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset, 
        batch_size=args.batch_size,  
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        classification_weight=args.classification_weight,
        output_dir=model_output_dir,
        visualization_dir=vis_output_dir,
        seed=args.seed,
        callback=lambda epoch, loss, metrics, args=None: training_callback(
            epoch, loss, metrics, {**callback_args, **(args or {})}
        )
    )

    print(f"Training complete. Best model saved at: {results['best_model_path']}")
    
    if 'training_history' in results:
        metrics_path = os.path.join(vis_output_dir, "final_metrics")
        visualize_training_metrics(results['training_history'], metrics_path)
    
    print("\nEvaluating on test set...")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate_model(model, test_dataloader)
    
    print(f"Test results - F1: {test_metrics['f1']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, "
          f"F1 (macro): {test_metrics['f1_macro']:.4f}, Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}")
    
    import json
    test_metrics_file = os.path.join(vis_output_dir, "test_metrics.json")
    with open(test_metrics_file, 'w') as f:
        metrics_to_save = {k: v for k, v in test_metrics.items() 
                          if k not in ['embeddings', 'probabilities', 'predictions', 'ground_truth']}
        json.dump(metrics_to_save, f, indent=2)
    print(f"Test metrics saved to {test_metrics_file}")
    
    confusion_path = os.path.join(vis_output_dir, "test_confusion_matrix.png")
    visualize_confusion_matrix(
        test_metrics['predictions'],
        test_metrics['ground_truth'],
        relation_to_id,
        confusion_path
    )
    
    embeddings_path = os.path.join(vis_output_dir, "test_embeddings.png")
    visualize_embeddings(
        test_metrics['embeddings'],
        test_metrics['ground_truth'],
        relation_to_id,
        embeddings_path
    )
    
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    relation_names = [id_to_relation.get(i, f"Unknown {i}") for i in range(len(relation_to_id))]
    
    report = test_metrics['classification_report']
    
    plt.figure(figsize=(14, 7))
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    width = 0.25
    x = np.arange(len(relation_names))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [report[rel][metric] if rel in report else 0 for rel in relation_names]
        plt.bar(x + (i-1)*width, values, width, label=metric.capitalize())
    
    plt.title('Performance Metrics per Relation Class')
    plt.xlabel('Relation Type')
    plt.ylabel('Score')
    plt.xticks(x, relation_names, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    metrics_path = os.path.join(vis_output_dir, "class_metrics.png")
    plt.savefig(metrics_path)
    print(f"Class metrics visualization saved as {metrics_path}")
    plt.close()
