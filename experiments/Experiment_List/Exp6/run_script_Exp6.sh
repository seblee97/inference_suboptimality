# Experiment 6
###############################################################################################
#Convention: last posterior in the name is the trained one.
###############################################################################################
# fashion-MNIST

# Standard

#Likelihood estimator for fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_likelihood_FFG --dataset fashion_mnist --ie True

#Basic FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_FFG --dataset fashion_mnist

#Locally optimised FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_Locally_Optimised_FFG_FFG --dataset fashion_mnist --ol True

###############################################################################################

# Flow

#Likelihood estimator for fashion-MNIST trained on flow
python main.py --experiment_name exp6_fashion_MNIST_likelihood_Flow --dataset fashion_mnist --ie True --ap rnvp_norm_flow

#Basic flow fashion-MNIST trained on flow
python main.py --experiment_name exp6_fashion_MNIST_Flow --dataset fashion_mnist --ap rnvp_norm_flow

#Locally optimised Flow fashion-MNIST trained on flow
python main.py --experiment_name exp6_fashion_MNIST_Locally_Optimised_Flow_flow --dataset fashion_mnist --ol True --ap rnvp_norm_flow --lap rnvp_norm_flow

###############################################################################################

# Large Encoder

#Likelihood estimator for fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_LE_likelihood_FFG --dataset fashion_mnist --ie True --ehd "500 500 500"

#Basic FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_LE_FFG --dataset fashion_mnist  --ehd "500 500 500"

#Locally optimised FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_LE_Locally_Optimised_FFG_FFG --dataset fashion_mnist --ol True --ehd "500 500 500"

###############################################################################################

# Large Decoder

#Likelihood estimator for fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_LD_likelihood_FFG --dataset fashion_mnist --ie True  --dhd "500 500 500"

#Basic FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_LD_FFG --dataset fashion_mnist --dhd "500 500 500"

#Locally optimised FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp6_fashion_MNIST_LD_Locally_Optimised_FFG_FFG --dataset fashion_mnist --ol True --dhd "500 500 500"


###############################################################################################
# Experiment 6B: in here

