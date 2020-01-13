set data data1000

switch $argv[1]

case --help
    echo "look at the code"

case debug
    python main2.py --n_jobs 30\
                        --train_load $data/train.csv\
                        --n_samples 10\
                        --gen_save $data/DEBUG\
                        -s 5 -p 10 -n 20


end 


