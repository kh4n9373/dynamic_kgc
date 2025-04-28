# An example to run EDC (without refinement) on the example dataset

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
DATASET=example

python edc/run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./edc/few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./edc/few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --ee_llm $EE_LLM \
    --sc_embedder $SC_EMBEDDER \
    --input_text_file_path "./edc/datasets/${DATASET}.txt" \
    --target_schema_path "./edc/schemas/${DATASET}_schema.csv" \
    --output_dir "./edc/output/${DATASET}_target_alignment" \
    --logging_verbose