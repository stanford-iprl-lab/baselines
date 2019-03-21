LR=.00025
CR=.2
NMB=128
NOPT=10
SEED=1
echo "starting training, LR=$LR SEED=${SEED}"
python baselines/run_ppo.py --lr $LR --seed $SEED --exp-name hopper-lr-$LR-cr-$CR-crflip-nmb-$NMB-nopt-$NOPT-seed-$SEED --env HopperBulletEnv-v0 \
  --network mlp --num_env 12 --nminibatches $NMB --noptepochs $NOPT --cliprange_anneal --cliprange_flip --cliprange $CR --num_timesteps 2e7 &> ../outs/HOPPER-LR-$LR-CR-$CR-CRFLIP-NMB-$NMB-NOPT-$NOPT-SEED-${SEED}.out &
