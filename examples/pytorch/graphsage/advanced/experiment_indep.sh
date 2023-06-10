for dataset in ogbn-products ogbn-papers100M ogbn-mag240M; do
for batch_size in ${1}; do

case "${dataset}" in
ogbn-papers100M) varg=1;;
*)  unset varg;;
esac

model=rgcn

for cache_size in 0 ${2}; do
for sampler in labor neighbor; do
for kappa in 1 256; do

if [[ ("${sampler}" == labor && ${cache_size} -gt 0) || ${kappa} -le 1 ]]; then

torchrun --nnodes=1:64 --nproc_per_node=1 --rdzv_id=123123123 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:25555 train_dist_indep.py --num-epochs=20 --dataset=${dataset} --batch-size=${batch_size} --num-hidden=1024 --uva-ndata=features --cache-size=${cache_size} --sampler=${sampler} --batch-dependency=${kappa} --model=${model} --logdir=/localscratch/tb_logs_indep_0_$(hostname -s) ${varg:+--undirected}

fi

done
done
done

done
done