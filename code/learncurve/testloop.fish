


set sge T 


#--neg AID/1345082_inactive.txt.gz --pos AID/1345082_active.txt.gz\
#--neg AID/AID2631_inactive.smi --pos AID/AID2631_active.smi  # 2 few left
#--neg AID/AID624466_inactive.smi --pos AID/AID624466_active.smi 
#--neg AID/AID651610_inactive.smi --pos AID/AID651610_active.smi
#--neg AID/bursi_neg.smi --pos AID/bursi_pos.smi\

function gogo

    python3 main2.py\
        --neg AID/AID624466_inactive.smi --pos AID/AID624466_active.smi\
        --n_jobs 24\
        --sge\
        --testsize 400\
        --loglevel  50\
        --grammar classic\
        --alternative_lc 1\
        --burnin $argv[1]\
        --n_steps (math $argv[1]+1)\
        --thickness $argv[2]\
        --emit 5\
        --reg $argv[3]\
        --min_cip $argv[4]\
        --trainsize  200 400 700\
        --optimize 1\
        --repeatseeds 12\
        --save (string join "_" $argv).sav #(date  "+%j_%H_%M").sav


        rm /home/mautner/scratch/logfiles/ld_e/* 
        rm /home/mautner/scratch/logfiles/ld_o/* 
        rm *.pickle 
        rm res/*

end 

for steps in 20 #5 10 15 20
    for thickness in 1 2 
        for reg in .1 .5 .9
            for mincip in 1 2
                echo "############### PARAMS  $steps $thickness $reg $mincip"
                gogo $steps $thickness $reg $mincip
            end
        end
    end
end


#grep "PARAMS|generated" 10hoptires.txt | grep "PARAMS| 0\.73"

