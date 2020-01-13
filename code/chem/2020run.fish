set models aae char_rnn vae organ
set datas mosestrain/{p,n}{200,400,600,800,1000,1200}_{0,1,2}
# see also moseslearncurve.py in learncurve for the evaluation
for model in $models
    for data in $datas
        set -e trainadd 
        set -e sampleadd 
        if test $model = 'organ'; set trainadd '--pg_iters' 10; end  # 10 is still ok just 2 for my test, neee
        mkdir -p $data/$model
        python scripts/train.py $model --train_load $data/train.csv\
                                       --model_save  $data/$model/mod\
                                       --vocab_save $data/$model/voc\
                                       --n_jobs 10\
                                       --device cuda\
                                       --config_save $data/$model/conf $trainadd

        python scripts/sample.py $model --model_load $data/$model/mod\
                                        --vocab_load $data/$model/voc\
                                        --n_samples 1000\
                                        --gen_save $data/$model/gen\
                                        --device cuda\
                                        --config_load $data/$model/conf $sampleadd
   end
end

