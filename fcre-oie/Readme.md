# Installation
```bash
pip install -r requirements.txt
```

# TACRED 5-shot Experiments
This section describes configurations for running TACRED 5-shot experiments. Adjust `CUDA_VISIBLE_DEVICES` according to your GPU setup.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 5 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-5nga-bz32-202005.log
```

# FewRel 5-shot Experiments
This section describes configurations for running FewRel 5-shot experiments. Adjust `CUDA_VISIBLE_DEVICES` according to your GPU setup.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --task_name FewRel --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 5 --w1 2.0 --w2 2.0 --w3 0.5 >> fewrel-5shot-5nga-bz32-202005.log
```