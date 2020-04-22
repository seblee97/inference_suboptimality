# Experiment E
###############################################################################################
# Convention: last posterior in the name is the trained one.
# This is the frozen decoder one. You need to have the file results/exp2_MNIST_basic_FFG/saved_vae_weights.pt to load it here.
###############################################################################################
# MNIST
#Basic FFG MNIST (frozen decoder from expA) trained on FFG
python main.py --experiment_name expE_MNIST_FD_FFG

#Basic AF MNIST (frozen decoder from expA) trained on AF
python main.py --experiment_name expE_MNIST_FD_AF --ap rnvp_aux_flow

#Locally optimised FFG MNIST (frozen decoder from expA) trained on FFG
python main.py --experiment_name expE_MNIST_FD_Locally_Optimised_FFG_FFG --ol True

#Locally optimised AF MNIST (frozen decoder from expA) trained on AF
python main.py --experiment_name expE_MNIST_FD_Locally_Optimised_AF_AF --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow

#Likelihood estimator for MNIST (frozen decoder from expA) trained on FFG
python main.py --experiment_name expE_MNIST_FD_likelihood_FFG --ie True

#Likelihood estimator for MNIST (frozen decoder from expA) trained on AF
# The same as above

###############################################################################################
