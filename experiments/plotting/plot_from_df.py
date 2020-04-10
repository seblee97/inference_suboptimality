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
parser.add_argument("-plot_keys", type=str, help="path to file containing list of attributes to plot for summary", default=None)
parser.add_argument("-exp_name", type=str, help="name of experiment")

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

        fig_sub.plot(x_data, plot_data)
        
        # labelling
        fig_sub.set_xlabel("Epoch")
        fig_sub.set_ylabel(title)
    
        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.5)
        fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

    def generate_plot(self):

        for row in range(self.rows):
            for col in range(self.columns):
        
                graph_index = (row) * self.columns + col

                if graph_index < self.number_of_graphs:

                    print(graph_index)

                    attribute_title = self.plot_keys[graph_index]
                    attribute_data = self.data[attribute_title].dropna()

                    self.add_subplot(
                            plot_data=attribute_data, row_index=row, column_index=col, title=attribute_title
                            )

        self.fig.suptitle("Summary Plot: {}".format(self.experiment_name))

        self.fig.savefig("{}/summary_plot.pdf".format(self.save_path), dpi=500)
        plt.close()


if __name__ == "__main__":

    os.makedirs("{}/figures/".format(args.save_path), exist_ok=True)
    figure_save_path = "{}/figures".format(args.save_path)

    path_to_csv = os.path.join(args.save_path, "data_logger.csv")

    sP = SummaryPlot(data_csv=path_to_csv, plot_config_path=args.plot_keys, save_path=figure_save_path, experiment_name=args.exp_name)
    sP.generate_plot()
