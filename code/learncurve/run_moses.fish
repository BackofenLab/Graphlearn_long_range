set data data5000
set path $HOME/repos/moses
set grammars classic coarse priosim coarseloco 

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
case runall 
    for grammar in $grammars 
        mkdir -p $path/$data/$grammar
        python3 main2.py --n_jobs 8\
                         --no-sge\
                         --grammar $grammar\
                         --train_load $path/$data/train.csv\
                         --gen_save $path/$data/$grammar/gen\
                         --n_samples 1000\
                         --testsize 500\
                         --loglevel 99\
                         --burnin 20\
                         --emit 5\
                         --n_steps 21\
                         --num_sample 50\
                         --thickness 2

    end
end

