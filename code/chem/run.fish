



set models aae char_rnn vae organ


set data data5000

for model in $models

    set -e trainadd 
    set -e sampleadd 
    if test $model = 'organ'; set trainadd '--pg_iters' 10; end  # 10 is still ok just 2 for my test, neee
    mkdir $data/$model
    python scripts/train.py $model --train_load $data/train.csv\
                                   --model_save  $data/$model/mod\
                                   --vocab_save $data/$model/voc\
                                   --n_jobs 40\
                                   --device cpu\
                                   --config_save $data/$model/conf $trainadd

    python scripts/sample.py $model --model_load $data/$model/mod\
                                    --vocab_load $data/$model/voc\
                                    --n_samples 5000\
                                    --gen_save $data/$model/gen\
                                    --device cpu\
                                    --config_load $data/$model/conf $sampleadd
end

