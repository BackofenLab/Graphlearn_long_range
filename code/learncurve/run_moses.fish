set data data5000
set path $HOME/repos/moses

switch $argv[1]

case --help
    echo "look at the code"

case debug
    python main2.py     --n_jobs 10\
                        --no-sge\
                        --train_load $path/$data/train.csv\
                        --n_samples 14\
                        --gen_save $path/$data/DEBUG\
                        --grammar classic\
                        --burnin 10\
                        --n_steps 11\
                        --emit 4


case classic
    mkdir -p $path/$data/classic/
    python main2.py     --n_jobs 10\
                        --no-sge\
                        --train_load $path/$data/train.csv\
                        --n_samples 1000\
                        --gen_save $path/$data/classic/gen\
                        --grammar classic\
                        --burnin 10\
                        --n_steps 11\
                        --emit 4
end 


