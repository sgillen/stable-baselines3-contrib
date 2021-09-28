for ENV in Pendulum-v0 HumanoidBulletEnv-v0
do
    python test_dummy_env.py $ENV
    
    python test_subproc_env.py $ENV

    mpiexec -n 11 python test_mpi_env.py $ENV
    
done
	   
