import yaml
import matplotlib.pyplot as plt
import numpy as np
import argparse
from main import (Syns, Population)

class MyMonitor:
    """instance for checking and plotting values of neurons and synapses"""

    def __init__(self, neurons:dict, synapses:dict, yaml_file:dict) -> None:
        """saving instances of neurons and synapses for data extraction
           neurons - dict with instances of Population class from main.py
           synapses - dict with instances of Syns class from main.py
           yaml_file - dictionary containing network parameters"""

        self.neurons = neurons
        self.synapses = synapses
        self.yaml_file = yaml_file

    def _extract_connectivity(self) -> None:
        """extracts connectivity from synapses"""

        self.connectivity = {}
        """Each key in the dictionary is a synapse name, and the value is a numpy array of connectivity data;
           First "column" in array is presynaptic neuron index, second is postsynaptic neuron index, third is distance, 
           fourth is synaptic conductance, fifth is synaptic delay"""
        
        for name, syn in self.synapses.items():
            self.connectivity[name] = np.array(syn.connectivity)

    def connectivity_results(self):
        """returns results of the connectivity"""

        self._extract_connectivity() # run to create the connectivity dictionary
        
        # creating plots grid based on the number of connectivity types
        r = len(self.connectivity)

        fig, ax = plt.subplots(r,2,figsize=(12, r*6))
        ax_flat = ax.flatten()
        i = 0
        for name, data in self.connectivity.items():
            geom_type = self.yaml_file['synapses'][name]['geometry']['type']
            text = (f'Amount of neurons: {len(np.unique(self.connectivity[name][:,0]))}\n'
                    f'Amount of synapses: {len(data)}\n'
                    f'Distance function: {geom_type}')

            ax_flat[i].set_title(f'Distance over conductance for {name}')
            ax_flat[i].set_xlabel('Distance')
            ax_flat[i].set_ylabel('Conductance')
            ax_flat[i].scatter(data[:,2], data[:,3])
            ax_flat[i].text(0.35, 0.95, text, transform=ax_flat[i].transAxes, fontsize=10-r, verticalalignment='top')

            i += 1

            ax_flat[i].set_title(f'Histogram of conductance for {name}')
            ax_flat[i].set_xlabel('Conductance')
            ax_flat[i].set_ylabel('LogScale')
            ax_flat[i].hist(data[:,3], bins=150, log=True)

            i += 1

        
        if args.save:
            plt.savefig(args.save, dpi=300, bbox_inches='tight')
        if args.show_plots:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give script a yaml file and analyse it")
    parser.add_argument("filename", type=str, help="The path of the YAML file to load")
    parser.add_argument("-s", "--save", default=False, help="Save the plots with selected name")
    parser.add_argument("-p", "--show_plots", action="store_true", help="Show the plots")
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
            f = yaml.safe_load(file)

    pops = {}
    syns = {}
    geometry = f['geometry']

    for pop in f['populations']:
            pops[pop] = Population(f['populations'][pop], population=pop, geometry=geometry)
            #test = Population(f['populations'][pop], population=pop, geometry=geometry)

    for syn in f['synapses']:
            syns[syn] = Syns(f['synapses'][syn], pops, connection_name=syn)
            #test = Syns(f['synapses'][syn], neurons=pops, connection_name=syn)

    m = MyMonitor(neurons=pops, synapses=syns, yaml_file=f)
    m.connectivity_results()