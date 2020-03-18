import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.scale as scale
#import pandas as pd


"""
TO USE, JUST HAVE THE MODEL CODE PRINT AN OUTPUT IN THE CORRECT FORMAT
Then use:
python experiments/main.py >> parse.txt
NB: if you have over 8 variables, the colors will repeat

examples:

=== in the rnvp_flow.py code
zer0 = torch.sum(z0).item()
print("noisez0: ", zer0)

=== in the rnvp_loss.py
vae_r = torch.sum(vae_reconstruction).item()
print("vae_reconstruction: ", vae_r)
vae_l = torch.sum(vae_latent).item()
print("vae_latent: ", vae_l)
m = torch.sum(mean).item()
print("mean: ", m)
log_v = torch.sum(log_var).item()
print("log_var: ", log_v)
logdt = torch.sum(log_det_jacobian).item()
print("log_det_jacobian: ", logdt)
x_scal = torch.sum(x).item()
print("x: ", x_scal)
logpxz = torch.sum(log_p_xz).item()
print("log_p_xz: ", logpxz)
logpz = torch.sum(log_p_z).item()
print("log_p_z: ", logpz)
logqzx = torch.sum(log_q_zx).item()
print("log_q_zx: ", logqzx)
logpx = torch.sum(log_p_x).item()
print("log_p_x: ", logpx)

"""


var_list = {}



with open("parse4.txt", "r") as file:
    lines = file.readlines()[2:-1]
    for line in lines:
        data = line.split(":")
        var = data[0].strip()
        if var not in var_list.keys():
            var_list[var] = []

        var_list[var].append(data[1].strip())

#print(var_list)

fig = plt.figure()
plot = fig.add_subplot()

for label, variables in var_list.items():
    num = len(var_list[label])
    x = list(range(0, num))
    strl = str(label)
    try:
        plot.plot(x, variables, label=strl)
    except RuntimeWarning:
        print(strl)


#start, end = plot.get_ylim()
#print(start, end)
plot.set_yscale(value='log')
#loc = ticker.MultipleLocator(base=100)
#plot.yaxis.set_major_locator(loc)

plot.set_xlabel("Steps")
#plot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%'))

plot.legend(loc="right")
#plot.set_yticks([]) # uncomment to clear all y-axis labels if they are messed up
plt.show()