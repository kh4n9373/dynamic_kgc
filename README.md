# Step 1: Non-dynamic Graph Construction

This repository contains a comprehensive pipeline for processing, evaluating, and visualizing knowledge graphs from various datasets. The pipeline extracts relation triplets, processes them through a sequence of steps, and generates evaluation metrics and visualizations.

## Pipeline Overview

The main workflow is managed by `run.sh`, which executes the following sequence:

```
Extract Dataset → EDC Processing → Preprocess EDC Result → Construct Dataset → Evaluation → NOTA Clustering 
```


### Models
- BAAI/bge-m3 (or other embedding model)
- Gemini-1.5-flash (or other LLM for relation extraction, you can configure to use different models and different base url, here I use gemini from google vertex ai, you could use gpt models from openai, claude from anthropic,... as long as you provide model_name and base_url in `.env`)

### Setup
1. Clone the repository
2. Activate your environment (venv / poetry / uv / conda,...)
3. Install dependencies: `pip install -r requirements.txt`
4. Configure your LLM provider base url in `.env` (google gemini base url for example): 
```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```
3. Set up API keys in environment (to deal with rate limits, we can configure to use different api keys): 
```
bash set_api_key.sh "api-key-1" "api-key-2" ...
```
4. Download the embedding url, here I download BGE/m3: 
```
ct2-transformers-converter --model BAAI/bge-m3 --output_dir embedding_model --force
```

## Configuration Options

The pipeline can be customized in multiple ways in first lines of `run.sh`:

### Dataset Selection
```bash
# In run.sh
DATASET=rebel_simple  # Options: webnlg_simple, webnlg, wiki-nre, rebel
```

### LLM Selection
```bash
# In run.sh
OIE_LLM=gemini-1.5-flash
SD_LLM=gemini-1.5-flash
SC_LLM=gemini-1.5-flash
EE_LLM=gemini-1.5-flash
```

### Embedder Selection
```bash
# In run.sh
SC_EMBEDDER=BAAI/bge-m3
```

### Hardware Optimization
Use the `--device` parameter with values:
- `auto`: Automatically detect best device
- `cpu`: CPU processing
- `cuda`: NVIDIA GPU acceleration
- `mps`: Apple Silicon acceleration

## Running the Pipeline

To run the entire pipeline:
```bash
bash ./run.sh
```

To run individual steps (rebel dataset for example):
```bash
# Extract dataset
python extract_original_dataset.py --config rebel

# Run EDC
python edc/run.py --input_text_file_path "./preprocessed_documents/rebel.txt" --target_schema_path "./edc/schemas/rebel_schema.csv"

# Construct dataset
python construct_final_dataset.py --original preprocessed_dataset/rebel.csv --edc preprocessed_edc_result/rebel.csv --device mps

# Evaluate
python process_evaluation.py --input raw_dataset_constructed/rebel.csv --similarity_threshold 0.85

# Cluster NOTA cases
python clustering.py --input_file evaluation/rebel/NOTA.json --dbscan_eps 0.8 --dbscan_min_samples 2
```
To reset the pipeline output and run again
```
bash reset.sh
```
## Output Structure

```
└── project_root/
    ├── preprocessed_dataset/           # Original extracted dataset
    ├── preprocessed_documents/         # Sentences for EDC processing
    ├── preprocessed_edc_result/        # Preprocessed EDC output
    ├── raw_dataset_constructed/        # Constructed dataset with similarities
    ├── dataset_constructed/            # Dataset with evaluation flags
    ├── evaluation/                     # Evaluation metrics and visualizations
    │   └── {dataset}/
    │       ├── confusion_matrix.png
    │       ├── similarity_distribution.png
    │       ├── relation_match_rates.png
    │       ├── metrics.json
    │       └── NOTA.json
    ├── cluster_visualization/          # NOTA cluster visualizations
    │   └── {dataset}/
    │       ├── cluster_viz.png
    │       └── distributions/
    └── cluster_model/                  # Saved cluster models
        └── {dataset}/
```

## Pipeline Steps in Detail

### 1. Extract Original Dataset
**Script**: `extract_original_dataset.py`

**Purpose**: từ tập dữ liệu và schema tương ứng cho trước (webnlg/rebel/wiki-nre...) chúng ta trích xuất các dữ liệu (text) trong tập mà các quan hệ (relation) trong dữ liệu đó được đĩnh nghĩa trong schema. Những bước tới ta sẽ chạy EDC trên tập dữ liệu đầu ra này để đánh giá chất lượng schema alignment EDC.

**Inputs**:
- `--schema`: CSV file with relation definitions (`edc/schemas/{dataset}_schema.csv`)
- `--triplets`: Text file with labeled relation triplets (`edc/evaluate/references/{dataset}.txt`)
- `--sentences`: the datasets (raw text) (`edc/datasets/{dataset}.txt`)

**Outputs**:
- `--output_csv`: CSV with extracted triplets (`preprocessed_dataset/{dataset}.csv`)
- `--output_sentences`: Text file with unique sentences for EDC processing (`preprocessed_documents/{dataset}.txt`)

**Configuration**:
- Use `--config` with values like `webnlg_simple`, `webnlg`, `wiki-nre`, `rebel` to use predefined settings

**Output Format**: CSV with columns `line_number`, `subject`, `object`, `relation`, `relation_definition`, `sentence`.

### 2. EDC (Extract Define Canonicalize) Processing
**Script**: `edc/run.py`

**Purpose**: Chạy luồng EDC gốc trên tập dữ liệu tiền xử lý sau bước 1 

**Inputs**:
- Input text: Sentences from `preprocessed_documents/{dataset}.txt`
- Schema: Relation definitions from `edc/schemas/{dataset}_schema.csv`

**Outputs**:
- EDC output JSON: `edc/output/{dataset}_target_alignment/iter0/result_at_each_stage.json`

**Configuration**:
- LLM settings: `--oie_llm`, `--sd_llm`, `--sc_llm`, `--ee_llm`
- Embedder: `--sc_embedder`
- Logging verbosity: `--logging_verbose`

### 3. Preprocess EDC Result
**Script**: `preprocess_edc_result.py`

**Purpose**: Tiền xử lý các triplets trích xuất ra từ EDC để đánh giá

**Inputs**:
- `--input`: EDC output JSON (`edc/output/{dataset}_target_alignment/iter0/result_at_each_stage.json`)

**Outputs**:
- `--output`: CSV with extracted relations (`preprocessed_edc_result/{dataset}.csv`)

**Output Format**: CSV with columns `sentence`, `extracted_subject`, `extracted_object`, `extracted_relation`, `extracted_relation_definition`.

### 4. Construct Dataset
**Script**: `construct_final_dataset.py`

**Purpose**: Join bảng dự đoán của EDC với bảng ở bước (1) để so sánh kết quả của EDC với labels

**Inputs**:
- `--original`: Original dataset CSV (`preprocessed_dataset/{dataset}.csv`)
- `--edc`: Preprocessed EDC output (`preprocessed_edc_result/{dataset}.csv`)
- `--model_name`: Embedding model name for similarity calculation
- `--model_path`: Local path to embedding model

**Outputs**:
- `--output`: Merged dataset with similarity scores (`raw_dataset_constructed/{dataset}.csv`)

**Configuration**:
- Hardware acceleration: `--device` (auto/cpu/cuda/mps)
- Processing options: `--batch_size`, `--cache_size`
- Similarity threshold: `--similarity_threshold`

**Output Format**: CSV with original and extracted relations, plus similarity scores between relation definitions.

### 5. Evaluation
**Script**: `process_evaluation.py`

**Purpose**: Xây dựng NOTA (None of the above) cho các relation EDC extract ra không align với schema + đánh giá chất lượng alignment của EDC theo micro/macro f1, precision, recall

**Inputs**:
- `--input`: Constructed dataset (`raw_dataset_constructed/{dataset}.csv`)
- `--similarity_threshold`: Threshold for determining matches (default: 0.85)

**Outputs**:
- `--output`: Processed dataset with evaluation flags (`dataset_constructed/{dataset}.csv`)
- `--metrics_folder`: Directory for metrics and visualizations (`evaluation/{dataset}/`)
  - `confusion_matrix.png`: Confusion matrix visualization
  - `similarity_distribution.png`: Distribution of similarity scores
  - `relation_match_rates.png`: Match rates by relation type
  - `metrics.json`: Evaluation metrics (precision, recall, F1, etc.)
  - `NOTA.json`: "None of the Above" cases for clustering

**Key Metrics**:
- True/False Positives
- True/False Negatives
- Accuracy, Precision, Recall, F1 Score
- NOTA rate (percentage of unmatched relations)

### 6. NOTA Clustering
**Script**: `clustering.py`

**Purpose**: Phân cụm NOTA "None of the Above" cases to identify patterns in unmatched relations.

**Inputs**:
- `--input_file`: NOTA cases from evaluation (`evaluation/{dataset}/NOTA.json`)

**Outputs**:
- `--save_path`: Cluster visualization (`cluster_visualization/{dataset}/cluster_viz.png`)
- `--model_save_path`: Saved cluster model (`cluster_model/{dataset}/`)
- Distribution visualizations in `cluster_visualization/{dataset}/distributions/`

**Configuration**:
- DBSCAN parameters: `--dbscan_eps`, `--dbscan_min_samples`
- UMAP parameters: `--umap_n_neighbors`, `--umap_min_dist`

## Visualization Options

### Knowledge Graph Visualization
**Script**: `visualize.py`

**Purpose**: Creates interactive visualizations of knowledge graphs.

**Inputs**:
- `--triplets_file`: Txt file containing knowledge graph triplets

**Usage**:
```bash
python visualize.py --triplets_file edc/output/{dataset}_target_alignment/iter0/canon_kg.txt
```

Then navigate to `http://127.0.0.1:5000/` in your browser.



# Training & Test

## Training
Run example training at `example_training.sh`

For real train:
```
python train.py \
  --csv_path {PATH_TOYOUR_CONSTRUCTED_DATASET} \
  --epochs 50 \
  --learning_rate 2e-5 \
  --train_split 0.8 \
  --val_split 0.2
```
for more configuration, check `python train.py --help`

## Testing
Run example testing at `example_testing.sh`

For real test:
```
python test.py \
  --dataset_name {NAME_OF_THE_TRAINED_DATASET} \
  --model_path {PATH_TO_THE_MODEL_CHECKPOINT} 
```

Check the visualization of training process at `/training_process`


