import pandas as pd
import numpy as np
import itertools
import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse
import os
import json

from typing import Dict, List

parser = argparse.ArgumentParser()

parser.add_argument("-save_path", type=str, help="path to save figures and in which to find csv data")
parser.add_argument("-csv", type=str, help="csv name", default="data_logger.csv")
parser.add_argument("-plot_keys", type=str, help="path to file containing list of attributes to plot for summary", default=None)
parser.add_argument("-exp_name", type=str, help="name of experiment")
parser.add_argument("-rolling_mean", type=int, help="number of data points to calculate mean over")
parser.add_argument("-zoom", type=int, help="number of epochs before zooming in")
parser.add_argument("-compression", type=int, help="percentage to compress to", default=10)
parser.add_argument("-lo", type=str, help="Local optimisation?", default=None)


args = parser.parse_args()

# Hard-coded subplot layouts for different numbers of graphs
LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 3), 10: (2, 5), 11: (3, 4), 12: (3, 4)}

class SummaryPlot:

    def __init__(self, data_csv, plot_config_path: str, save_path:str, experiment_name:str):

        self.data = pd.read_csv(data_csv)

        with open(plot_config_path) as json_file:
            self.plot_config = json.load(json_file)

        self.save_path = save_path
        self.experiment_name = experiment_name

        if args.lo:
            self.plot_keys = self.plot_config["local_opt_keys"]
        else:
            self.plot_keys = self.plot_config["base_keys"]

        self.number_of_graphs = len(self.plot_keys)

        self.scale_axes = self.plot_config["num_epochs"]

        self.rows = LAYOUTS[self.number_of_graphs][0]
        self.columns = LAYOUTS[self.number_of_graphs][1]

        width = self.plot_config['width']
        height = self.plot_config['height']

        heights = [height for _ in range(self.rows)]
        widths = [width for _ in range(self.columns)]

        self.fig = plt.figure(constrained_layout=False, figsize=(self.columns * width, self.rows * height))

        self.spec = gridspec.GridSpec(nrows=self.rows, ncols=self.columns, width_ratios=widths, height_ratios=heights)

    def add_subplot(self, plot_data, row_index: int, column_index: int, title: str): #, labels: List, ylimits: List

        fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])

        # scale axes
        scaling = self.scale_axes / len(plot_data)
        x_data = [i * scaling for i in range(len(plot_data))]

        if args.zoom:
            data_to_compress = int((args.zoom * len(plot_data)) / self.scale_axes)
            upper_bound = plot_data.iloc[data_to_compress]
            lower_bound = plot_data.iloc[0]

            # shrink scale before zoom to 10%
            lower_bound_dif = abs(upper_bound - lower_bound)
            target_lower_bound_dif = (lower_bound_dif * args.compression) / 100
            target_lower_bound = upper_bound - target_lower_bound_dif

            for value in range(0,data_to_compress):
                x = plot_data.iloc[value]
                if x <= upper_bound:
                    plot_data.iloc[value] = -1 * ((abs(upper_bound - x) * target_lower_bound_dif) / lower_bound_dif) + upper_bound
                if x > upper_bound:
                    plot_data.iloc[value] = -1 * ((abs(upper_bound - x) * target_lower_bound_dif) / lower_bound_dif) + upper_bound

            #target_lower_bound_dif = ((abs(lower_bound - upper_bound) * 10) / 100)
            #target_lower_bound = upper_bound - target_lower_bound_dif
            #increase value
            #y_data = plot

        if args.rolling_mean:
            fig_sub.plot(x_data, plot_data, label="{}pt MEAN".format(args.rolling_mean), color='orange')
        else:
            fig_sub.plot(x_data, plot_data)



        # labelling
        fig_sub.set_xlabel("Epoch")
        fig_sub.set_ylabel(title)

        # grids
        fig_sub.minorticks_on()
        fig_sub.legend(loc='lower right')
        fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.7)
        fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

    def generate_plot(self):

        for row in range(self.rows):
            for col in range(self.columns):

                graph_index = (row) * self.columns + col

                if graph_index < self.number_of_graphs:

                    #print("index: ", graph_index)

                    attribute_title = self.plot_keys[graph_index]
                    attribute_data = self.data[attribute_title].dropna()

                    #print(attribute_data)
                    if args.rolling_mean:

                        attribute_data = attribute_data.rolling(window=args.rolling_mean).mean()

                        self.add_subplot(
                            plot_data=attribute_data, row_index=row, column_index=col, title=attribute_title
                            )
                    #print(rolling_mean)
                    else:
                        self.add_subplot(
                            plot_data=attribute_data, row_index=row, column_index=col, title=attribute_title
                            )


        self.fig.suptitle("Summary Plot: {}".format(self.experiment_name))

        self.fig.savefig("{}/{}_summary_plot.pdf".format(self.save_path, args.exp_name), dpi=500)
        plt.close()


if __name__ == "__main__":

    os.makedirs("{}/figures/".format(args.save_path), exist_ok=True)
    figure_save_path = "{}/figures".format(args.save_path)

    path_to_csv = os.path.join(args.save_path, args.csv)

    sP = SummaryPlot(data_csv=path_to_csv, plot_config_path=args.plot_keys, save_path=figure_save_path, experiment_name=args.exp_name)
    sP.generate_plot()
