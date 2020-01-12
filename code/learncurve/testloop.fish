rm *.pickle

function gogo
	python3 main2.py\
		--n_jobs 24\
		--sge\
		--neg AID/bursi_neg.smi\
		--pos AID/bursi_pos.smi\
		--testsize 500\
		--loglevel  1\
		--grammar classic\
		--burnin 8\
		--emit 5\
		--n_steps 9\
		--radii (seq 0 $argv[1])\
		--thickness $argv[2]\
		--min_cip $argv[3]\
		--trainsize 300\
		--repeatseeds 123
end 

set maxrad 1 2 3 
set thickness 1 2
set minc 1 2 

for m in $maxrad 
    for t in $thickness 
        for c in $minc
            echo "###############      PARAMS $m $t $c"
            gogo $m $t $c
        end
    end
end
