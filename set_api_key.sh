#!/bin/bash

# Script to set up multiple Gemini API keys in the .env file
# Usage: ./set_api_keys.sh "key1" "key2" "key3" ...

ENV_FILE=".env"

grep -v "LLM_API_KEY" $ENV_FILE > temp_env || touch temp_env
mv temp_env $ENV_FILE

if [ -n "$1" ]; then
    echo "LLM_API_KEY=$1" >> $ENV_FILE
    echo "Added main LLM_API_KEY to $ENV_FILE"
    shift
else
    echo "Error: Please provide at least one API key"
    exit 1
fi

count=1
for key in "$@"; do
    echo "LLM_API_KEY_${count}=$key" >> $ENV_FILE
    echo "Added LLM_API_KEY_${count} to $ENV_FILE"
    count=$((count+1))
done

echo "API keys have been successfully set in $ENV_FILE"
echo "Total keys: $count"

chmod 600 $ENV_FILE 