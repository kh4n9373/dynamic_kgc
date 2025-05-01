import pandas as pd
import csv
from typing import Set, Tuple
from collections import defaultdict
import numpy as np
import os
import sys
import datetime
import argparse
from io import StringIO
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process merged output file to create evaluation with NOTA values and metrics')
    
    # File path arguments
    parser.add_argument('--input', type=str, default='merged_output.csv',
                        help='Path to input merged_output.csv file')
    parser.add_argument('--output', type=str, default='evaluation.csv',
                        help='Path to output evaluation.csv file')
    parser.add_argument('--metrics_folder', type=str, default='metrics_evaluation',
                        help='Folder to save metrics logs and NOTA.json')
    
    # Processing parameters
    parser.add_argument('--similarity_threshold', type=float, default=0.9,
                        help='Similarity threshold for determining NOTA values (default: 0.9)')
    
    # Predefined configurations
    parser.add_argument('--config', type=str, choices=['webnlg_simple', 'webnlg', 'wiki-nre', 'rebel'],
                        help='Use predefined dataset configuration')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle predefined configurations
    if args.config:
        if args.config == 'webnlg_simple':
            args.input = 'merged_output/webnlg_simple.csv'
            args.output = 'evaluation/webnlg_simple.csv'
            args.metrics_folder = 'metrics_evaluation/webnlg_simple'
        elif args.config == 'webnlg':
            args.input = 'merged_output/webnlg.csv'
            args.output = 'evaluation/webnlg.csv'
            args.metrics_folder = 'metrics_evaluation/webnlg'
        elif args.config == 'wiki-nre':
            args.input = 'merged_output/wiki-nre.csv'
            args.output = 'evaluation/wiki-nre.csv'
            args.metrics_folder = 'metrics_evaluation/wiki-nre'
        elif args.config == 'rebel':
            args.input = 'merged_output/rebel.csv'
            args.output = 'evaluation/rebel.csv'
            args.metrics_folder = 'metrics_evaluation/rebel'
    
    return args

def process_merged_output(input_file: str, output_file: str, metrics_folder: str = "metrics_evaluation", similarity_threshold: float = 0.9):
    """
    Process the merged_output.csv file to create evaluation.csv with NOTA values
    
    Args:
        input_file: Path to merged_output.csv
        output_file: Path to save evaluation.csv
        metrics_folder: Folder to save metrics logs
        similarity_threshold: Threshold for determining NOTA values (default: 0.9)
    """
    # Create metrics folder if it doesn't exist
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to capture console output
    log_capture = StringIO()
    log_file_path = os.path.join(metrics_folder, f"evaluation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Function to log to both console and string buffer
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_capture)
    
    log_print(f"Reading input file: {input_file}")
    log_print(f"Similarity threshold: {similarity_threshold}")
    df = pd.read_csv(input_file, dtype=str).fillna('')
    
    # Initialize set to track processed pairs
    NOTA_assigned: Set[Tuple[str, str]] = set()
    
    # Initialize output dataframe
    output_rows = []
    
    # Initialize dictionaries to track matching
    subj_obj_pairs = {}
    extracted_subj_obj_pairs = {}
    
    # First pass - build dictionaries of all pairs
    log_print("Building pair dictionaries...")
    for _, row in df.iterrows():
        subject = row.get('subject', '')
        obj = row.get('object', '')
        extracted_subject = row.get('extracted_subject', '')
        extracted_object = row.get('extracted_object', '')
        
        # Add to dictionaries
        subj_obj_pairs[(subject, obj)] = True
        extracted_subj_obj_pairs[(extracted_subject, extracted_object)] = True
    
    # Second pass - process each row
    log_print("Processing rows...")
    for _, row in df.iterrows():
        subject = row.get('subject', '')
        obj = row.get('object', '')
        extracted_subject = row.get('extracted_subject', '')
        extracted_object = row.get('extracted_object', '')
        definition_similarity = float(row.get('definition_similarity', 0))
        
        # Skip if already processed
        if (subject, obj) in NOTA_assigned or (extracted_subject, extracted_object) in NOTA_assigned:
            continue
        
        # Check if pairs match
        if subject.lower() == extracted_subject.lower() and obj.lower() == extracted_object.lower():
            # Pairs match - determine NOTA based on similarity
            nota = "false" if definition_similarity >= similarity_threshold else "true"
            
            # Create output row
            output_row = row.copy()
            output_row['NOTA'] = nota
            output_rows.append(output_row)
            
            # Add to NOTA_assigned
            NOTA_assigned.add((subject, obj))
            NOTA_assigned.add((extracted_subject, extracted_object))
        else:
            # Pairs don't match
            # Check if extracted pair has no matching subject/object pair
            if not any(subject.lower() == extracted_subject.lower() and obj.lower() == extracted_object.lower() 
                      for (subject, obj) in subj_obj_pairs.keys()):
                # Create output row with NOTA=true and null subject/object
                output_row = row.copy()
                output_row['subject'] = None
                output_row['object'] = None
                output_row['NOTA'] = "true"
                output_rows.append(output_row)
                
                # Add to NOTA_assigned
                NOTA_assigned.add((extracted_subject, extracted_object))
            
            # Check if subject/object pair has no matching extracted pair
            if not any(subject.lower() == es.lower() and obj.lower() == eo.lower() 
                      for (es, eo) in extracted_subj_obj_pairs.keys()):
                # Create output row with NOTA=invalid and null extracted_subject/extracted_object
                output_row = row.copy()
                output_row['extracted_subject'] = None
                output_row['extracted_object'] = None
                output_row['NOTA'] = "invalid"
                output_rows.append(output_row)
                
                # Add to NOTA_assigned
                NOTA_assigned.add((subject, obj))
    
    # Create output dataframe and write to CSV
    log_print(f"Writing output to {output_file}...")
    output_df = pd.DataFrame(output_rows)
    
    # Ensure column order includes NOTA
    all_columns = list(df.columns) + ['NOTA']
    output_df = output_df.reindex(columns=all_columns)
    
    # Calculate NOTA ratios
    total_rows = len(output_df)
    true_count = len(output_df[output_df['NOTA'] == 'true'])
    false_count = len(output_df[output_df['NOTA'] == 'false'])
    invalid_count = len(output_df[output_df['NOTA'] == 'invalid'])
    
    # Print NOTA distribution
    log_print("\nNOTA Distribution:")
    log_print(f"NOTA=true:    {true_count} rows ({true_count/total_rows:.2%})")
    log_print(f"NOTA=false:   {false_count} rows ({false_count/total_rows:.2%})")
    log_print(f"NOTA=invalid: {invalid_count} rows ({invalid_count/total_rows:.2%})")
    
    # Calculate metrics
    # For extraction evaluation:
    # - true positives (TP): NOTA=false (correct extractions)
    # - false positives (FP): NOTA=true (incorrect extractions)
    # - false negatives (FN): NOTA=invalid (missed extractions)
    
    # Calculate metrics by relation
    metrics_by_relation = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    
    # Count by relation
    for _, row in output_df.iterrows():
        relation = row.get('relation', 'unknown')
        extracted_relation = row.get('extracted_relation', 'unknown')
        nota = row.get('NOTA', '')
        
        # Set relation to extracted_relation if relation is empty
        relation = extracted_relation if not relation or pd.isna(relation) else relation
        
        if nota == 'false':
            metrics_by_relation[relation]['TP'] += 1
        elif nota == 'true':
            metrics_by_relation[relation]['FP'] += 1
        elif nota == 'invalid':
            metrics_by_relation[relation]['FN'] += 1
    
    # Calculate micro metrics (global counts)
    micro_tp = false_count  # NOTA=false are true positives
    micro_fp = true_count   # NOTA=true are false positives
    micro_fn = invalid_count  # NOTA=invalid are false negatives
    
    # Calculate micro precision, recall, F1
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Calculate macro metrics (average of per-relation metrics)
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    
    # Prepare metrics data for CSV
    relation_metrics_data = []
    
    log_print("\nMetrics by Relation:")
    log_print(f"{'Relation':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    log_print("-" * 60)
    
    for relation, counts in metrics_by_relation.items():
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']
        support = tp + fn
        
        # Skip relations with no examples
        if support == 0:
            continue
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)
        
        # Print metrics for this relation
        log_print(f"{relation[:20]:<20} {precision:.4f}     {recall:.4f}     {f1:.4f}     {support}")
        
        # Add to metrics data
        relation_metrics_data.append({
            'relation': relation,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    # Calculate macro averages
    macro_precision = np.mean(macro_precisions) if macro_precisions else 0
    macro_recall = np.mean(macro_recalls) if macro_recalls else 0
    macro_f1 = np.mean(macro_f1s) if macro_f1s else 0
    
    # Print summary metrics
    log_print("\nSummary Metrics:")
    log_print(f"Micro Precision: {micro_precision:.4f}")
    log_print(f"Micro Recall:    {micro_recall:.4f}")
    log_print(f"Micro F1:        {micro_f1:.4f}")
    log_print(f"Macro Precision: {macro_precision:.4f}")
    log_print(f"Macro Recall:    {macro_recall:.4f}")
    log_print(f"Macro F1:        {macro_f1:.4f}")
    
    # Write to CSV
    output_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    log_print(f"\nProcessing complete. Output saved to {output_file}")
    log_print(f"Total rows processed: {len(df)}")
    log_print(f"Total rows in output: {len(output_df)}")
    
    # Save metrics to files
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save relation metrics to CSV
    relation_metrics_file = os.path.join(metrics_folder, f"relation_metrics_{timestamp}.csv")
    pd.DataFrame(relation_metrics_data).to_csv(relation_metrics_file, index=False)
    log_print(f"Relation metrics saved to: {relation_metrics_file}")
    
    # Save summary metrics to CSV
    summary_metrics_file = os.path.join(metrics_folder, f"summary_metrics_{timestamp}.csv")
    summary_data = [{
        'metric': 'Micro Precision', 'value': micro_precision
    }, {
        'metric': 'Micro Recall', 'value': micro_recall
    }, {
        'metric': 'Micro F1', 'value': micro_f1
    }, {
        'metric': 'Macro Precision', 'value': macro_precision
    }, {
        'metric': 'Macro Recall', 'value': macro_recall
    }, {
        'metric': 'Macro F1', 'value': macro_f1
    }, {
        'metric': 'NOTA True Count', 'value': true_count
    }, {
        'metric': 'NOTA False Count', 'value': false_count
    }, {
        'metric': 'NOTA Invalid Count', 'value': invalid_count
    }, {
        'metric': 'Total Rows', 'value': total_rows
    }]
    pd.DataFrame(summary_data).to_csv(summary_metrics_file, index=False)
    log_print(f"Summary metrics saved to: {summary_metrics_file}")
    
    # Create NOTA.json with entries where NOTA = false
    nota_false_df = output_df[output_df['NOTA'] == 'true']
    nota_false_records = []
    
    for _, row in nota_false_df.iterrows():
        # Convert row to dict and handle NaN/None values
        row_dict = {}
        for col, val in row.items():
            if pd.isna(val):
                row_dict[col] = None
            else:
                row_dict[col] = val
        nota_false_records.append(row_dict)
    
    # Save to JSON file
    nota_json_file = os.path.join(metrics_folder, "NOTA.json")
    with open(nota_json_file, 'w', encoding='utf-8') as f:
        json.dump(nota_false_records, f, ensure_ascii=False, indent=2)
    
    log_print(f"NOTA=false records saved to: {nota_json_file}")
    
    # Save log to file
    with open(log_file_path, 'w') as log_file:
        log_file.write(log_capture.getvalue())
    log_print(f"Log saved to: {log_file_path}")
    
    return output_df

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Input file:          {args.input}")
    print(f"  Output file:         {args.output}")
    print(f"  Metrics folder:      {args.metrics_folder}")
    print(f"  Similarity threshold: {args.similarity_threshold}\n")
    
    # Process the merged output file
    process_merged_output(
        args.input, 
        args.output, 
        args.metrics_folder,
        args.similarity_threshold
    )

if __name__ == "__main__":
    main() 