DATASET_NAME=webnlg # or wiki_nre or rebel
EPOCHS=50
LEARNING_RATE=2e-5
TRAIN_SPLIT=0.8
VAL_SPLIT=0.2

python train.py \
  --csv_path "dataset_constructed/${DATASET_NAME}.csv" \
  --epochs ${EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --train_split ${TRAIN_SPLIT} \
  --val_split ${VAL_SPLIT}
