for SEED in 1 100 200
do 
	echo "starting training, SEED=${SEED}" 
	python baselines/run_ppo.py --lr 1e-3 --seed $SEED --exp-name lr-.001-crflip-seed-$SEED --env ReacherPyBulletEnv-v0 --network mlp --num_env 12 --cliprange .3 --num_timesteps 2e7 --save_interval 100000 --cliprange-flip &> ../outs/LR-.001-CRFLIP-SEED-${SEED}.out &
	sleep 20
done
