for dataset in ogbn-mag240M ogbn-papers100M ogbn-products; do

case "${dataset}" in
ogbn-papers100M) varg=1;;
*)  unset varg;;
esac

for replication in 0; do

model=rgcn

for cache_size in 0 1000000; do
for sampler in labor neighbor; do
for kappa in 1 256; do

if [[ ("${sampler}" == labor && ${cache_size} -gt 0) || ${kappa} -le 1 ]]; then

case "${cache_size}" in
0) vvarg=1;;
*)  unset vvarg;;
esac

torchrun --nnodes=1:64 --nproc_per_node=1 --rdzv_id=123123123 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:25555 train_dist_coop.py --num-epochs=10 --dataset=${dataset} --batch-size=1024 --train --num-hidden=1024 --uva-ndata=features --cache-size=${cache_size} --sampler=${sampler} --batch-dependency=${kappa} --replication=${replication} --logdir=tb_logs_runtimes_coop_${replication} --model=${model} ${varg:+--undirected} ${vvarg:+--uva-data}

fi

done
done
done


done
done