batch_size=${1}
dataset=${2}

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

cache_size=${3}
sampler=${4}
kappa=${5}
epochs=${6:-20}

case "${cache_size}" in
0) vvarg=1;;
*)  unset vvarg;;
esac

torchrun --nnodes=1:64 --nproc_per_node=1 --rdzv_id=123123123 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:25555 train_dist_coop.py --num-epochs=${epochs} --dataset=${dataset} --batch-size=${batch_size} --num-hidden=1024 --uva-ndata=features --cache-size=${cache_size} --sampler=${sampler} --batch-dependency=${kappa} --replication=${replication} --logdir=/localscratch/tb_logs_coop_${replication}_$(hostname -s) --model=${model} --partition=${part} --num-parts-multiplier=${multiplier} ${varg:+--undirected} ${vvarg:+--uva-data}


done
