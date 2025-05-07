from text_clustering.src.text_clustering import ClusterClassifier
# from datasets import load_dataset
import json
import argparse
import os
import glob

def main():
    parser = argparse.ArgumentParser(description='Run text clustering with configurable parameters')
    parser.add_argument('--dbscan_eps', type=float, default=0.7, 
                        help='DBSCAN eps parameter (neighborhood radius)')
    parser.add_argument('--dbscan_min_samples', type=int, default=2, 
                        help='DBSCAN min_samples parameter (min neighbors to form a core point)')
    parser.add_argument('--umap_n_neighbors', type=int, default=5, 
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--umap_min_dist', type=float, default=0.1, 
                        help='UMAP min_dist parameter')
    parser.add_argument('--input_file', type=str, default="",
                        help='Input JSON file with relation definitions')
    parser.add_argument('--dataset', type=str, default="wiki-nre_simple",
                        help='Dataset name (e.g., wiki-nre_simple)')
    parser.add_argument('--save_path', type=str, default="",
                        help='Path to save visualization')
    parser.add_argument('--model_save_path', type=str, default="./cc_100k",
                        help='Path to save cluster model')
    
    args = parser.parse_args()
    
    # If no input file is specified, try to find the NOTA.json file in the dataset directory
    if not args.input_file:
        # Look for NOTA.json in the evaluation directory with the dataset name
        dataset_eval_dir = f"evaluation/{args.dataset}"
        nota_path = os.path.join(dataset_eval_dir, "NOTA.json")
        
        if os.path.exists(nota_path):
            args.input_file = nota_path
        else:
            # Try to find the most recent evaluation directory if specific one not found
            eval_dirs = glob.glob("evaluation/*")
            if eval_dirs:
                latest_eval_dir = max(eval_dirs, key=os.path.getmtime)
                potential_nota_path = os.path.join(latest_eval_dir, "NOTA.json")
                if os.path.exists(potential_nota_path):
                    args.input_file = potential_nota_path
                    print(f"Found NOTA.json in {latest_eval_dir}")
                else:
                    print(f"Error: NOTA.json not found in {latest_eval_dir}. Please check if evaluation process completed successfully.")
                    return
            else:
                print("Error: No evaluation directories found. Run the evaluation step first.")
                return
    
    # If no save path is specified, create one based on the dataset name
    if not args.save_path:
        save_dir = f"cluster_visualization/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        args.save_path = os.path.join(save_dir, "cluster_viz.png")

    # Load data
    try:
        with open(args.input_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{args.input_file}' is not a valid JSON file.")
        return

    texts = [item["extracted_relation_definition"] for item in data]

    if not texts:
        print("No texts found for clustering. Check if NOTA.json contains data.")
        return

    print(f"Loaded {len(texts)} texts for clustering from {args.input_file}")
    print(f"Clustering parameters: dbscan_eps={args.dbscan_eps}, dbscan_min_samples={args.dbscan_min_samples}")
    print(f"UMAP parameters: umap_n_neighbors={args.umap_n_neighbors}, umap_min_dist={args.umap_min_dist}")

    cc = ClusterClassifier(
        embed_device="mps", 
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist
    )

    embs, labels, summaries = cc.fit(texts)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    cc.show(save_path=args.save_path)
    
    distributions_dir = os.path.dirname(args.save_path) + "/distributions"
    os.makedirs(distributions_dir, exist_ok=True)
    try:
        cc.plot_distributions(save_dir=distributions_dir)
    except Exception as e:
        print(f"Warning: Could not generate distribution plots: {e}")

    # Save model for reuse
    cc.save(args.model_save_path)
    
    # Summary statistics
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = sum(1 for l in labels if l == -1)
    
    print(f"Clustering complete! Found {num_clusters} clusters and {noise_count} noise points.")
    
    if summaries:
        print("\nCluster summaries:")
        for label, summary in summaries.items():
            if label != -1:
                count = sum(1 for l in labels if l == label)
                print(f"Cluster {label} ({count} items): {summary}")

if __name__ == "__main__":
    main()