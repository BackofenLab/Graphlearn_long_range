
set models aae char_rnn vae organ
set models  classic  aae char_rnn vae organ
set models  priosim coarse

#set models coarse2

set data data5000

for model in $models
    python -W ignore scripts/eval.py  --test_path $data/test.csv\
                        --gen_path $data/$model/gen\
                        --device cpu\
                        --test_scaffolds_path $data/test_scaffolds.csv\
                        --ptest_scaffolds_path $data/test_scaffolds_stats.npz > $model.tr
end

paste aae.tr char_rnn.tr vae.tr organ.tr  classic.tr coarse.tr priosim.tr  -d',' | cut -d"," -f1,2,4,6,8,10,12,14,16,18
