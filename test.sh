DATASET_NAME=webnlg # or wiki_nre or rebel
MODEL_PATH="models/${DATASET_NAME}/model_epoch_17.pt"

python test.py \
  --dataset_name ${DATASET_NAME} \            
  --model_path ${MODEL_PATH}