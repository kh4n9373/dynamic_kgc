Download embedding model:
```
pip install -U ctranslate2
 ct2-transformers-converter --model BAAI/bge-m3 --output_dir bge_m3 --force
```

Run edc:
```
bash edc/run.sh
```


Visualize extracted graph
```
python visualize.py --triplets_file "path_to_your_canon_kg.txt"
```