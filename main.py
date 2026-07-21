import yaml
from numpy import *
from matplotlib.pyplot import *
from brian2 import *
from numpy import random as rnd

class Population:
    """main class to store for all yaml processing"""
    
    def __init__(self, file, geometry, population):
        """file - part of .yaml to work with population
           population - choose population from predifined subgroups in .yaml
           geometry - geometry of investigated area"""

        self.neuron_type = population
        self.file = file
        self.namespace = globals().copy()             # get access to the units in the global space

        # adding geometry variables to the instance
        for key, value in geometry.items():
            setattr(self, key, value)
            self.namespace[key] = value

        # we have to calculate amount of neurons if the formula is provided
        if isinstance(self.file['num_neurons'], str) and self.file['num_neurons']:
            self.dencity = self.file['dencity']
            if isinstance(self.dencity, str):
                self.dencity = eval(self.dencity, self.namespace)           # adding dencity first, but may be more -> this part is changable
            self.namespace['dencity'] = self.dencity                        # adding it to namespace as well

            res = int(eval(self.file['num_neurons'], self.namespace))     # amount of neurons should be int number
            self.num_neurons = res                
            self.namespace['num_neurons'] = res   # adding it to namespace as well
        else:
            self.num_neurons = self.file['num_neurons']      # for cases where initial amount is provided
            self.namespace['num_neurons'] = self.file['num_neurons']   # adding it to namespace as well

        def uniform_points_grid(L: int, H: int, N: int) -> np.array:
            """distribute given num_neurons (N) in the given 2D space L*H"""
            
            nx = int(round(np.sqrt(N * L / H)))
            ny = int(np.ceil(N / nx))
            while nx * ny < N:
                if L > H:
                    ny += 1
                else:
                    nx += 1
            base_nx_per_row = N // ny
            residuals = N % ny
            dx_total = L / nx
            dy_total = H / ny
            points = []
            for j in range(ny):
                points_in_this_row = base_nx_per_row + (1 if j < residuals else 0)
                if points_in_this_row == 0:
                    continue
                dx_row = L / points_in_this_row
                y = (j + 0.5) * dy_total
                for i in range(points_in_this_row):
                    x = (i + 0.5) * dx_row
                    points.append([x, y])

            return np.array(points)
            
        self.coord_grid = uniform_points_grid(L=self.L, H=self.H, N=self.num_neurons)
        
        self.load_constants()       # store constants from .yaml into instance
        self.load_equations()       # store equations from .yaml into instance
        self.set_neuron_group()     # creates neuron group
        
    def load_constants(self):
        """saves constans from the const section of .yaml into instance object"""

        const = self.file['const']         # choose only const part
        
        # looping through the found constants
        for name, expr in const.items():
            if isinstance(expr, str):               # only strings can be evaled
                value = eval(expr, self.namespace)  # looks for variables in the instance namespace
            else:
                value = expr
    
            setattr(self, name, value)      # adding variable "name" with value "value" to the instance namespace
            self.namespace[name] = value    # updating instance namespace with new variable

    def load_equations(self):
        """saves equations from the const section of .yaml into instance object"""
        
        eqs = self.file['equations']         # choose only equation part
        
        for name, expr in eqs.items():
            # some values here may be actual formulas
            try:                 
                expr = eval(expr, self.namespace)
            except SyntaxError:
                pass

            setattr(self, name, expr)      # adding variable "name" with value "expr" to the instance namespace
            self.namespace[name] = expr    # updating instance namespace with new variable

    def set_neuron_group(self):
        """creates a neuron group"""

        nrns = self.file['initials']       # choose only initials for neurons
        specs = self.file['specs']
        
        # create neuron group with previously saved values, all other specified parameters insert directly
        self.neurons = NeuronGroup(
                                   N=self.num_neurons,     # amount of neurons
                                   model=self.general_equ, # final equation
                                   namespace={k: v for k, v in self.namespace.items() if not k.startswith('_')},  # it doesn't like __methods
                                   **specs                 # all other specific keywords
                                  )
        
        for name, value in nrns.items():
            if isinstance(value, str):
                value = eval(value, self.namespace)     # those values are actual formulas
                setattr(self.neurons, name, value)      # adding variable "name" with value "expr" to the instance namespace
                self.namespace[name] = value            # updating instance namespace with new variable
            else:
                setattr(self.neurons, name, value)      # adding variable "name" with value "expr" to the instance namespace
                self.namespace[name] = value            # updating instance namespace with new variable


class Syns:
    """instance for working with synapses"""

    def __init__(self, file, neurons, connection_name):
        """file - part of .yaml to work with synapses
           neurons - dict with all instances of populations from .yaml
           connection_name - specific name of syn_to_syn connections from .yaml"""
        
        self.file = file
        self.name = connection_name

        self.namespace = globals().copy()             # get access to the units in the global space

        source = file['source'] # just a name of population
        target = file['target'] # just a name of population
        
        self.source = neurons[source]  # which population of neurons to take from .yaml
        self.target = neurons[target]  # which population of neurons to take from .yaml

        self.set_synapses()
        self.set_geometry()
        self.connect()

        
    def set_synapses(self):
        """specify synapses"""

        sns = self.file['general']
        self.synapses = Synapses(self.source.neurons, self.target.neurons, **sns)

    def set_geometry(self):
        """creates the geometry of synapses"""
        
        pre = self.source.num_neurons
        post = self.target.num_neurons

        self.connectivity = array([
            [source, target]
            for source in range(pre)
            for target in range(post)
            if source != target and rnd.rand() < 0.25], dtype=int)

    def connect(self):
        """connects synapses"""

        self.synapses.connect(i=self.connectivity[:,0].flatten().tolist(), j=self.connectivity[:,1].flatten().tolist())
        
        if self.file['connect']:
            for key, value in self.file['connect'].items():
                if isinstance(value, str):
                    value = eval(value, self.namespace)
                    setattr(self.synapses, key, value)
                    self.namespace[key] = value
                else:
                    setattr(self.synapses, key, value)
                    self.namespace[key] = value


with open('test.yaml', 'r') as file:
    f = yaml.safe_load(file)   
    
pops = {}
syns = {}
geometry = f['geometry']

for pop in f['populations']:
    pops[pop] = Population(f['populations'][pop], population=pop, geometry=geometry)
    #test = Population(f['populations'][pop], population=pop, geometry=geometry)

for syn in f['synapses']:
    syns[syn] = Syns(f['synapses'][syn], neurons=pops, connection_name=syn)
    #test = Syns(f['synapses'][syn], neurons=pops, connection_name=syn)