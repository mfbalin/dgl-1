for dataset in ogbn-products ogbn-papers100M ogbn-mag240M; do
for batch_size in 512; do

case "${dataset}" in
ogbn-papers100M) varg=1;;
*)  unset varg;;
esac

case "${dataset}" in
ogbn-mag240M)   part=random;;
*)              part=random;;
esac

for replication in 0; do

model=rgcn
multiplier=512

for cache_size in 0 250000; do
for sampler in labor neighbor; do
for kappa in 1 256; do

if [[ ("${sampler}" == labor && ${cache_size} -gt 0) || ${kappa} -le 1 ]]; then

case "${cache_size}" in
0) vvarg=1;;
*)  unset vvarg;;
esac

torchrun --nnodes=1:64 --nproc_per_node=1 --rdzv_id=123123123 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:25555 train_dist_coop.py --num-epochs=20 --dataset=${dataset} --batch-size=${batch_size} --num-hidden=1024 --uva-ndata=features --cache-size=${cache_size} --sampler=${sampler} --batch-dependency=${kappa} --replication=${replication} --logdir=/localscratch/tb_logs_runtimes_coop_${replication} --model=${model} --partition=${part} --num-parts-multiplier=${multiplier} ${varg:+--undirected} ${vvarg:+--uva-data}

fi

done
done
done


done
done
done