
set models aae char_rnn vae organ 
set data data5000

for model in $models
    python scripts/eval.py  --test_path $data/test.csv\
                        --gen_path $data/$model/gen\
                        --device cuda\
                        --test_scaffolds_path $data/test_scaffolds.csv\
                        --ptest_scaffolds_path $data/test_scaffolds_stats.npz
