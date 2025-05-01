#!/bin/bash
# -----SETUP ARGUMENTS HEREEEEE-----

# OIE_LLM=mistralai/Mistral-7B-Instruct-v0.2
# SD_LLM=mistralai/Mistral-7B-Instruct-v0.2
# SC_LLM=mistralai/Mistral-7B-Instruct-v0.2
# SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
# DATASET=example
OIE_LLM=gemini-1.5-flash
SD_LLM=gemini-1.5-flash
SC_LLM=gemini-1.5-flash
EE_LLM=gemini-1.5-flash
SC_EMBEDDER=BAAI/bge-m3
DATASET=webnlg_simple
DEVICE=mps

# Set up logging functions
log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
    exit 1
}

log_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

log_section() {
    echo -e "\n\033[1;36m===== $1 =====\033[0m"
}

# Create all necessary directories
log_section "CREATING DIRECTORIES"
mkdir -p preprocessed_dataset
mkdir -p preprocessed_documents
mkdir -p preprocessed_edc_result
mkdir -p raw_dataset_constructed
mkdir -p dataset_constructed
mkdir -p evaluation/${DATASET}
mkdir -p cluster_visualization/${DATASET}
mkdir -p cluster_model/${DATASET}

log_success "Created all necessary directories"

# -----Preprocess original dataset from webnlg/wiki-nre/rebel-----
log_section "EXTRACTING ORIGINAL DATASET"

# Use default configuration (webnlg_simple)
# python extract_original_dataset.py

# Use a different predefined configuration
# python extract_original_dataset.py --config webnlg_simple
# python extract_original_dataset.py --config webnlg
# python extract_original_dataset.py --config wiki-nre
# python extract_original_dataset.py --config rebel

# Customize specific parameters
log_info "Running extract_original_dataset.py with dataset: ${DATASET}"
python extract_original_dataset.py \
--schema edc/schemas/${DATASET}_schema.csv \
--triplets edc/evaluate/references/${DATASET}.txt \
--sentences edc/datasets/${DATASET}.txt \
--output_csv preprocessed_dataset/${DATASET}.csv \
--output_sentences preprocessed_documents/${DATASET}.txt

if [ $? -ne 0 ]; then
    log_error "Extract original dataset step failed"
fi

log_success "Successfully extracted original dataset"

# -----EDC FLOW-----
log_section "RUNNING EDC PIPELINE"

# An example to run EDC (without refinement) on the example dataset

# python edc/run.py \
#     --oie_llm $OIE_LLM \
#     --oie_few_shot_example_file_path "./edc/few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
#     --sd_llm $SD_LLM \
#     --sd_few_shot_example_file_path "./edc/few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
#     --sc_llm $SC_LLM \
#     --ee_llm $EE_LLM \
#     --sc_embedder $SC_EMBEDDER \
#     --input_text_file_path "./edc/datasets/${DATASET}.txt" \
#     --target_schema_path "./edc/schemas/${DATASET}_schema.csv" \
#     --output_dir "./edc/output/${DATASET}_target_alignment" \
#     --logging_verbose

# Chạy ít
log_info "Running EDC with model: ${OIE_LLM}, embedder: ${SC_EMBEDDER}"
python edc/run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./edc/few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./edc/few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --ee_llm $EE_LLM \
    --sc_embedder $SC_EMBEDDER \
    --input_text_file_path "./preprocessed_documents/${DATASET}.txt" \
    --target_schema_path "./edc/schemas/${DATASET}_schema.csv" \
    --output_dir "./edc/output/${DATASET}_target_alignment" \
    --logging_verbose

if [ $? -ne 0 ]; then
    log_error "EDC run step failed"
fi

log_success "Successfully completed EDC pipeline"

# -----PREPROCESS EDC OUTPUT-----
log_section "PREPROCESSING EDC OUTPUT"

# Check if the output file exists
if [ ! -f "edc/output/${DATASET}_target_alignment/iter0/result_at_each_stage.json" ]; then
    log_error "EDC output file does not exist. Check the previous step."
fi

# Specify input/output files
log_info "Processing EDC results"
python preprocess_edc_result.py \
--input edc/output/${DATASET}_target_alignment/iter0/result_at_each_stage.json \
--output preprocessed_edc_result/${DATASET}.csv

if [ $? -ne 0 ]; then
    log_error "Preprocess EDC result step failed"
fi

log_success "Successfully preprocessed EDC output"

# -----CONSTRUCT FINAL DATASET-----
log_section "CONSTRUCTING DATASET"

# Check if input files exist
if [ ! -f "preprocessed_dataset/${DATASET}.csv" ]; then
    log_error "Preprocessed dataset file not found. Check previous steps."
fi

if [ ! -f "preprocessed_edc_result/${DATASET}.csv" ]; then
    log_error "EDC preprocessed file not found. Check previous steps."
fi

log_info "Constructing final dataset"
python construct_final_dataset.py \
 --original preprocessed_dataset/${DATASET}.csv \
 --edc preprocessed_edc_result/${DATASET}.csv \
 --output raw_dataset_constructed/${DATASET}.csv \
 --device ${DEVICE}

if [ $? -ne 0 ]; then
    log_error "Construct final dataset step failed"
fi

log_success "Successfully constructed raw dataset"

# -----RUN EVALUATION-----
log_section "RUNNING EVALUATION"

# Check if input file exists
if [ ! -f "raw_dataset_constructed/${DATASET}.csv" ]; then
    log_error "Raw dataset file not found. Check previous steps."
fi

log_info "Running evaluation with similarity threshold: 0.9"
python process_evaluation.py \
 --input raw_dataset_constructed/${DATASET}.csv \
 --output dataset_constructed/${DATASET}.csv \
 --metrics_folder evaluation/${DATASET}/ \
 --similarity_threshold 0.9

if [ $? -ne 0 ]; then
    log_error "Process evaluation step failed"
fi

log_success "Successfully completed evaluation"

# -----CLUSTERING NOTA-----
log_section "CLUSTERING NOTA"

# Check if NOTA.json exists
if [ ! -f "evaluation/${DATASET}/NOTA.json" ]; then
    log_error "NOTA.json file not found. Check previous steps."
fi

# Create distributions directory
mkdir -p "cluster_visualization/${DATASET}/distributions"
log_info "Running clustering with eps=${dbscan_eps}, min_samples=${dbscan_min_samples}"

python clustering.py \
--dbscan_eps 0.8 \
--dbscan_min_samples 2 \
--umap_n_neighbors 3 \
--umap_min_dist 0.05 \
--input_file "evaluation/${DATASET}/NOTA.json" \
--save_path "cluster_visualization/${DATASET}/cluster_viz.png" \
--model_save_path "cluster_model/${DATASET}/"

if [ $? -ne 0 ]; then
    log_error "Clustering step failed"
fi

log_success "Clustering completed successfully"
log_section "PIPELINE COMPLETED"
log_success "Knowledge graph processing pipeline completed successfully!"