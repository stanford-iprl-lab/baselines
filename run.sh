for SEED in 1 100 200
do 
    for LR in .001
    do
        echo "starting training, LR=${LR} SEED=${SEED}" 
        python baselines/run_ppo.py --lr ${LR} --seed $SEED --exp-name lr-$LR-seed-$SEED --env ReacherPyBulletEnv-v0 --network mlp --num_env 12  --num_timesteps 2e7 &> ../outs/LR-$LR-SEED-$SEED.out &
        sleep 10
    done
done
