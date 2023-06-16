batch_size=${1}
dataset=${2}

case "${dataset}" in
ogbn-papers100M) varg=1;;
*)  unset varg;;
esac

model=rgcn

cache_size=${3}
sampler=${4}
kappa=${5}
epochs=${6:-20}

torchrun --nnodes=1:64 --nproc_per_node=1 --rdzv_id=123123123 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:25555 train_dist_indep.py --num-epochs=${epochs} --dataset=${dataset} --batch-size=${batch_size} --num-hidden=1024 --uva-ndata=features --cache-size=${cache_size} --sampler=${sampler} --batch-dependency=${kappa} --model=${model} --logdir=/localscratch/tb_logs_indep_0_$(hostname -s) ${varg:+--undirected}
