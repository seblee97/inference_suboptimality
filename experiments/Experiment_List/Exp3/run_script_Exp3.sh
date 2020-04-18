# Experiment 3
###############################################################################################
#Convention: last posterior in the name is the trained one.
#Here it runs exactly as in 2 except the config file has a large encoder.
###############################################################################################
# MNIST
#Likelihood estimator for MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_likelihood_FFG --ie True

#Likelihood estimator for MNIST trained on AF
python main.py --experiment_name exp3_MNIST_likelihood_AF --ie True --ap rnvp_aux_flow

#Basic FFG MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_FFG

#Basic AF MNIST trained on AF
python main.py --experiment_name exp3_MNIST_AF --ap rnvp_aux_flow

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_Locally_Optimised_FFG_FFG --ol True

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow

#Locally optimised FFG MNIST trained on AF
python main.py --experiment_name exp3_MNIST_Locally_Optimised_FFG_AF --ol True --ap rnvp_aux_flow

#Locally optimised AF MNIST trained on AF
python main.py --experiment_name exp3_MNIST_Locally_Optimised_AF_AF --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow

###############################################################################################
# MNIST with Large decoder
python main.py --experiment_name exp3_MNIST_LD_FFG --ie True --ehd "200 200" --dhd "500 500"

#Basic FFG MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_LD_FFG --ehd "200 200" --dhd "500 500"

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_LD_Locally_Optimised_FFG_FFG --ol True --ehd "200 200" --dhd "500 500"

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_LD_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow --ehd "200 200" --dhd "500 500"

###############################################################################################
# MNIST Small decoder
python main.py --experiment_name exp3_MNIST_SD_FFG --ie True --ehd "200 200" --dhd " "

#Basic FFG MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_SD_FFG --ehd "200 200" --dhd " "

#Basic AF MNIST trained on AF
python main.py --experiment_name exp3_MNIST_SD_AF --ehd "200 200" --dhd " " --ap rnvp_aux_flow

#Locally optimised FFG MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_SD_Locally_Optimised_FFG_FFG --ol True --ehd "200 200" --dhd " "

#Locally optimised AF MNIST trained on FFG
python main.py --experiment_name exp3_MNIST_SD_Locally_Optimised_AF_FFG --ol True --lap rnvp_aux_flow --ehd "200 200" --dhd " "

#Locally optimised FFG MNIST trained on AF
python main.py --experiment_name exp3_MNIST_SD_Locally_Optimised_FFG_AF --ol True --lap gaussian --ehd "200 200" --dhd " "

#Locally optimised AF MNIST trained on AF
python main.py --experiment_name exp3_MNIST_SD_Locally_Optimised_AF_AF --ol True --lap rnvp_aux_flow --ehd "200 200" --dhd " "

###############################################################################################

# fashion-MNIST
#Likelihood estimator for fashion-MNIST trained on FFG
python main.py --experiment_name exp3_fashion_MNIST_likelihood_FFG --dataset fashion_mnist --ie True

#Likelihood estimator for fashion-MNIST trained on AF
python main.py --experiment_name exp3_fashion_MNIST_likelihood_AF --dataset fashion_mnist --ie True --ap rnvp_aux_flow

#Basic FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp3_fashion_MNIST_FFG --dataset fashion_mnist

#Basic AF fashion-MNIST trained on AF
python main.py --experiment_name exp3_fashion_MNIST_AF --dataset fashion_mnist --ap rnvp_aux_flow

#Locally optimised FFG fashion-MNIST trained on FFG
python main.py --experiment_name exp3_fashion_MNIST_Locally_Optimised_FFG_FFG --dataset fashion_mnist --ol True

#Locally optimised AF fashion-MNIST trained on FFG
python main.py --experiment_name exp3_fashion_MNIST_Locally_Optimised_AF_FFG --dataset fashion_mnist --ol True --lap rnvp_aux_flow

#Locally optimised FFG fashion-MNIST trained on AF
python main.py --experiment_name exp3_fashion_MNIST_Locally_Optimised_FFG_AF --dataset fashion_mnist --ol True --ap rnvp_aux_flow

#Locally optimised AF fashion-MNIST trained on AF
python main.py --experiment_name exp3_fashion_MNIST_Locally_Optimised_FFG_AF --dataset fashion_mnist --ol True --ap rnvp_aux_flow --lap rnvp_aux_flow

