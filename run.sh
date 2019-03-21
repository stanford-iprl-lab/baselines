LR=.0003
CR=.3
NMB=32
NOPT=10
SEED=1
echo "starting training, LR=$LR SEED=${SEED}"
python baselines/run_ppo.py --lr $LR --seed $SEED --exp-name hopper-lranneal-$LR-trcrdecay-$CR-nmb-$NMB-nopt-$NOPT-seed-$SEED --env HopperBulletEnv-v0 \
      --network mlp --num_env 16 --cliprange $CR --cliprange_anneal --cliprange_flip --lr_anneal --num_timesteps 2e7 &> ../outs/HOPPER-LRANNEAL-$LR-TRCRDECAY-$CR-NMB-$NMB-NOPT-$NOPT-SEED-${SEED}.out &
