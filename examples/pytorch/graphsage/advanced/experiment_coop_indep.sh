for batch_size in ${1}; do

for dataset in ogbn-products ogbn-papers100M ogbn-mag240M; do
for cache_size in 0 ${2}; do
for sampler in labor neighbor; do
for kappa in 1 256; do
for algo in coop indep; do

if [[ ("${sampler}" == labor && ${cache_size} -gt 0) || ${kappa} -le 1 ]]; then

bash ./experiment_${algo}.sh ${batch_size} ${dataset} ${cache_size} ${sampler} ${kappa} ${3} # last argument is num epochs

fi

done
done
done
done
done

done