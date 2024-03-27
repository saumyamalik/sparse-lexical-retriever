port=$(shuf -i 15000-16000 -n 1)
TDIR="/scratch/gpfs/awettig/processed_the_pile/bs6144"
TRAINDATASETS="${TDIR}/Wikipedia-train-0.5B/ ${TDIR}/ArXiv-train-0.5B/ ${TDIR}/Books3-train-0.5B/ ${TDIR}/Gutenberg-train-0.5B/ ${TDIR}/Github-train-0.5B/ ${TDIR}/YoutubeSubtitles-train-0.5B/ ${TDIR}/FreeLaw-train-0.5B/ ${TDIR}/HackerNews-train-0.5B/"

rmin=1.0
rmax=1.0
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0.
mo=bert-base-uncased
mp=none
bsz=10000

name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG

module purge
module load anaconda3/2023.3
conda activate nlp
wandb offline

python bm25.py \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --main_port $port \
        --total_steps 340 \
        --batch_size $bsz \