import random 
import pandas as pd 
import numpy as np 

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('-path_to_csv', type=str)
parser.add_argument('-sample_type', type=str, choices=["balanced", "random"])

args = parser.parse_args()

lo_elbos = pd.read_csv(args.path_to_csv)['test_loss'].dropna().tolist()

print("Overall Sample Mean Local Optimisation ELBO: {}".format(np.mean(lo_elbos)))
print("Overall Sample STD Local Optimisation ELBO: {}".format(np.std(lo_elbos)))

balanced_digits = [8, 7, 2, 8, 7, 4, 8, 7, 0, 2, 8, 3, 2, 3, 9, 4, 9, 2, 7, 7, 
1, 0, 5, 7, 9, 9, 8, 3, 5, 0, 0, 3, 1, 2, 9, 6, 0, 8, 4, 9, 7, 
4, 8, 8, 5, 1, 2, 3, 5, 6, 5, 3, 2, 6, 6, 1, 4, 6, 9, 4, 9, 5, 
9, 2, 6, 4, 9, 1, 6, 8, 7, 3, 6, 5, 0, 5, 1, 0, 4, 6, 3, 7, 5, 
7, 4, 1, 2, 2, 0, 8, 3, 5, 6, 4, 1, 1, 3, 0, 0, 1]  

new_balanced_digits = [8, 9, 9, 5, 1, 5, 7, 8, 2, 0, 
6, 2, 7, 5, 8, 0, 0, 8, 1, 2, 
2, 4, 2, 9, 3, 5, 3, 7, 4, 9, 
9, 9, 3, 7, 3, 4, 8, 1, 0, 1, 
0, 0, 8, 3, 7, 1, 5, 4, 6, 7, 
7, 7, 2, 4, 2, 6, 6, 0, 3, 2, 
6, 3, 5, 4, 7, 0, 1, 6, 1, 6, 
5, 8, 4, 0, 5, 1, 1, 3, 4, 2, 
9, 3, 6, 4, 1, 9, 4, 8, 5, 5, 
9, 0, 6, 7, 8, 8, 9, 2, 6, 3
]

if args.sample_type == 'balanced':
    # nested list where list i corresponds to elbos of digit i
    by_digit = np.array([[elbo for e, elbo in enumerate(lo_elbos) if new_balanced_digits[e] == i] for i in range(10)])
    per_digit_mean = [np.mean(digit_elbo) for digit_elbo in by_digit]
    per_digit_variance = [np.std(digit_elbo) for digit_elbo in by_digit]
    print("Per digit mean: ", per_digit_mean)
    print("Per digit deviations: ", per_digit_variance)
    # nested list where each inner list containes exactly one elbo per digit
    elbo_subsets = np.rollaxis(by_digit, axis=1)
elif args.sample_type == 'random':
    elbo_subsets = [lo_elbos[10*i:10*(i+1)] for i in range(10)]

subset_means = [np.mean(subset) for subset in elbo_subsets]
std_subset_means = np.std(subset_means)

print("Standard Deviation of Estimator:{}".format(std_subset_means))