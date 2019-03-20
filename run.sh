for SEED in 1
do 
	echo "starting training, SEED=${SEED}" 
	python baselines/run_ppo.py --lr 1e-3 --seed $SEED --exp-name hopper-lr-.001-nmb-32-nep-10-trcr-.2-seed-$SEED --env HopperBulletEnv-v0 --network mlp --num_env 12 --nminibatches 32 --noptepochs 10 --cliprange .2 --num_timesteps 2e7 --save_interval 100 &> ../outs/HOPPER-LR-.001-NMB-32-NEP-10-TRCR-.2-SEED-${SEED}.out &
	sleep 20
done
