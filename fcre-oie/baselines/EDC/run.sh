# An example to run EDC (without refinement) on the example dataset

# OIE_LLM=mistralai/Mistral-7B-Instruct-v0.2
OIE_LLM=gpt-4o-mini
# SD_LLM=mistralai/Mistral-7B-Instruct-v0.2
SD_LLM=gpt-4o-mini
# SC_LLM=mistralai/Mistral-7B-Instruct-v0.2
SC_LLM=gpt-4o-mini
# SC_EMBEDDER=intfloat/e5-mistral-7b-instruct
SC_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
DATASET=example

python run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --input_text_file_path "./datasets/${DATASET}.txt" \
    --target_schema_path "./schemas/${DATASET}_schema.csv" \
    --output_dir "./output/${DATASET}_target_alignment" \
    --logging_verbose \
    --task_id "-1" \
    --rel_id "-1" \
    --phase_1_2 true

