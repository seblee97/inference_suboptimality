# Experiment Planar
###############################################################################################
#Convention:
###############################################################################################
# MNIST

# Standard

#Basic model for -MNIST trained on Planar flow
python main.py --experiment_name exp_MNIST_planar --ap planar_flow

# Locally optimised for -MNIST Planar trained on Planar
python ../../main.py --experiment_name exp_MNIST_Locally_Optimised_planar_planar --ol True --lap planar_flow --ap planar_flow

