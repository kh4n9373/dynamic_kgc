CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 1 >> tacred-5shot-1nga-bz16.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 3 >> tacred-5shot-3nga-bz16.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 5 >> tacred-5shot-5nga-bz16.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 7 >> tacred-5shot-7nga-bz16.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 10 >> tacred-5shot-10nga-bz16.log

CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 48 --num_gen_augment 1 >> tacred-5shot-1nga-bz48.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 48 --num_gen_augment 3 >> tacred-5shot-3nga-bz48.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 48 --num_gen_augment 5 >> tacred-5shot-5nga-bz48.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 48 --num_gen_augment 7 >> tacred-5shot-7nga-bz48.log
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 48 --num_gen_augment 10 >> tacred-5shot-10nga-bz48.log