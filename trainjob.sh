port=$(shuf -i 15000-16000 -n 1)
TDIR="/scratch/gpfs/awettig/processed_the_pile/bs6144"
TRAINDATASETS="${TDIR}/Wikipedia-train-0.5B/ ${TDIR}/ArXiv-train-0.5B/ ${TDIR}/Books3-train-0.5B/ ${TDIR}/Gutenberg-train-0.5B/ ${TDIR}/Github-train-0.5B/ ${TDIR}/YoutubeSubtitles-train-0.5B/ ${TDIR}/FreeLaw-train-0.5B/ ${TDIR}/HackerNews-train-0.5B/"

rmin=0.3
rmax=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
mo=bert-base-uncased
mp=none

name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG

module purge
module load anaconda3/2023.3
conda activate nlp
wandb offline

python train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 1000 --lr 0.01 \
        --name $name \
        --scheduler linear \
        --per_gpu_batch_size 64 \
        --output_dir /home/saumyam/contriever/logtrain/$name \
        --main_port $port \