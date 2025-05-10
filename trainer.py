import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import os
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from losses import SupervisedContrastiveLoss
from load_dataset import RelationDataset
from encoder import RelationExtractor

def evaluate_model(model, dataloader):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The RelationExtractor model
        dataloader: DataLoader containing the evaluation data
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    if model.classifier is None:
        return {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            sentences = batch["sentence"]
            head_entities = batch["head_entity"]
            tail_entities = batch["tail_entity"]
            relations = batch["relation"].to(model.device)
            
            outputs = model(sentences, head_entities, tail_entities)
            logits = outputs["relation_logits"]
            embeddings = outputs["latent_representations"]
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(relations.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            all_embeddings.extend(embeddings.cpu().detach().numpy())
    
    if len(all_preds) == 0:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "f1_macro": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "classification_report": {},
            "predictions": [],
            "ground_truth": [],
            "probabilities": [],
            "embeddings": []
        }
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1_weighted,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "classification_report": report,
        "predictions": all_preds,
        "ground_truth": all_labels,
        "probabilities": all_probs,
        "embeddings": all_embeddings
    }

def train_relation_extractor(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
    max_grad_norm=1.0,
    warmup_steps=0,
    weight_decay=0.01,
    contrastive_weight=1.0,
    classification_weight=1.0,
    output_dir="./saved_model",
    visualization_dir=None,
    seed=42,
    callback=None
):
    """
    Fine-tune the relation extractor with supervised contrastive loss.
    
    Args:
        model: The RelationExtractor model
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        max_grad_norm: Maximum gradient norm for gradient clipping
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        contrastive_weight: Weight for contrastive loss
        classification_weight: Weight for classification loss
        output_dir: Directory to save the model
        visualization_dir: Directory to save visualizations
        seed: Random seed for reproducibility
        callback: Optional callback function for tracking metrics during training
    
    Returns:
        dict: Training metrics and best model path
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if visualization_dir and not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    if model.classifier is not None:
        classification_loss_fn = nn.CrossEntropyLoss()
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                        if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    total_steps = len(train_loader) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    best_model_path = None
    
    training_history = {
        "epochs": [],
        "train_loss": [],
        "contrastive_loss": [],
        "classification_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_f1_macro": [],
        "val_precision": [],
        "val_recall": [],
        "learning_rate": []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        contrastive_losses = []
        classification_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            sentences = batch["sentence"]
            head_entities = batch["head_entity"]
            tail_entities = batch["tail_entity"]
            relations = batch["relation"].to(model.device)
            
            optimizer.zero_grad()
            
            outputs = model(sentences, head_entities, tail_entities)
            latent_reps = outputs["latent_representations"]
            
            c_loss = contrastive_loss_fn(latent_reps, relations)
            
            if model.classifier is not None and outputs["relation_logits"] is not None:
                cls_loss = classification_loss_fn(outputs["relation_logits"], relations)
                total_loss = (contrastive_weight * c_loss) + (classification_weight * cls_loss)
            else:
                cls_loss = torch.tensor(0.0, device=model.device)
                total_loss = c_loss
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            train_losses.append(total_loss.item())
            contrastive_losses.append(c_loss.item())
            classification_losses.append(cls_loss.item() if cls_loss.item() > 0 else 0)
            
            if model.classifier is not None:
                progress_bar.set_postfix({
                    "loss": total_loss.item(),
                    "c_loss": c_loss.item(),
                    "cls_loss": cls_loss.item() if cls_loss.item() > 0 else 0
                })
            else:
                progress_bar.set_postfix({"c_loss": c_loss.item()})
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
        avg_classification_loss = sum(classification_losses) / len(classification_losses)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{num_epochs} - Avg. Train Loss: {avg_train_loss:.4f}, "
              f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
              f"Classification Loss: {avg_classification_loss:.4f}")
        
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(avg_train_loss)
        training_history["contrastive_loss"].append(avg_contrastive_loss)
        training_history["classification_loss"].append(avg_classification_loss)
        training_history["learning_rate"].append(current_lr)
        
        if val_dataset:
            val_metrics = evaluate_model(model, val_loader)
            print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, "
                  f"F1 (macro): {val_metrics['f1_macro']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            
            # Update validation metrics history
            training_history["val_accuracy"].append(val_metrics["accuracy"])
            training_history["val_f1"].append(val_metrics["f1"])
            training_history["val_f1_macro"].append(val_metrics["f1_macro"])
            training_history["val_precision"].append(val_metrics["precision"])
            training_history["val_recall"].append(val_metrics["recall"])
            
            if callback:
                callback_args = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "contrastive_loss": avg_contrastive_loss,
                    "classification_loss": avg_classification_loss,
                    "val_metrics": val_metrics,
                    "visualization_dir": visualization_dir
                }
                callback(epoch, avg_train_loss, val_metrics, callback_args)
            
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
                
                model_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "contrastive_loss": avg_contrastive_loss,
                    "classification_loss": avg_classification_loss,
                    "f1_score": val_metrics["f1"],
                    "f1_macro": val_metrics["f1_macro"],
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"]
                }
                
                torch.save(model_state, best_model_path)
                print(f"Saved best model to {best_model_path}")
        elif callback:  # Call callback with only loss if no validation data
            callback_args = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "contrastive_loss": avg_contrastive_loss,
                "classification_loss": avg_classification_loss,
                "visualization_dir": visualization_dir
            }
            callback(epoch, avg_train_loss, None, callback_args)
    
    if best_model_path is None:
        last_model_path = os.path.join(output_dir, f"model_final_epoch.pt")
        
        model_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": num_epochs - 1,
            "train_loss": avg_train_loss,
            "contrastive_loss": avg_contrastive_loss,
            "classification_loss": avg_classification_loss
        }
        
        if val_dataset:
            model_state.update({
                "f1_score": val_metrics["f1"],
                "f1_macro": val_metrics["f1_macro"],
                "accuracy": val_metrics["accuracy"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"]
            })
        
        torch.save(model_state, last_model_path)
        print(f"No best model found. Saved final model to {last_model_path}")
        best_model_path = last_model_path
    
    if visualization_dir:
        history_path = os.path.join(visualization_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Saved training history to {history_path}")
    
    results = {
        "final_train_loss": avg_train_loss,
        "final_contrastive_loss": avg_contrastive_loss,
        "final_classification_loss": avg_classification_loss,
        "best_val_f1": best_val_f1 if val_dataset else None,
        "best_model_path": best_model_path,
        "training_history": training_history
    }
    
    return results
