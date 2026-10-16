import yaml
import datetime
import json
from numpy import *
from matplotlib.pyplot import *
from brian2 import *
from numpy import random as rnd
from connectivity import connectivity
import argparse
import textwrap

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

            
        self.coord_grid = uniform_points_grid(L=self.L, H=self.H, N=self.num_neurons)
        self.namespace['positions'] = self.coord_grid

        self.load_constants()       # store constants from .yaml into instance
        self.load_equations()       # store equations from .yaml into instance
        self.set_neuron_group()     # creates neuron group
        
    def load_constants(self):
        """saves constans from the const section of .yaml into instance object"""

        const = self.file['const']         # choose only const part
        
        # looping through the found constants
        for name, expr in const.items():
            if isinstance(expr, str):               # only strings can be evaluated
                try:
                    value = eval(expr, self.namespace)  # looks for variables in the instance namespace
                except BaseException as e:
                    raise RuntimeError(f"Cannot evaluate expression for constants `{name}: {value}`: {e}")
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
        
        self.namespace['neurons'] = self.neurons
        for name, value in nrns.items():
            if isinstance(value, str):
                try:
                    value = eval(value, self.namespace) # those values are actual formulas
                except BaseException as e:
                    raise RuntimeError(f"Cannot evaluate expression for initial condition `{name}: {value}`: {e}")
                setattr(self.neurons, name, value)      # adding variable "name" with value "expr" to the instance namespace
                self.namespace[name] = value            # updating instance namespace with new variable
            else:
                setattr(self.neurons, name, value)      # adding variable "name" with value "expr" to the instance namespace
                self.namespace[name] = value            # updating instance namespace with new variable

class Syns:
    """instance for working with synapses"""

    def __init__(self, desc:dict, neurons, connection_name):
        """file - part of .yaml to work with synapses
           neurons - dict with all instances of populations from .yaml
           connection_name - specific name of syn_to_syn connections from .yaml"""
        
        self.file = desc
        self.name = connection_name

        self.namespace = globals().copy()             # get access to the units in the global space

        source = self.file['source'] # just a name of population
        target = self.file['target'] # just a name of population
        
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

        self.connectivity = connectivity(
            self.name,
            self.file,
            self.source.coord_grid,
            self.target.coord_grid,
            self.source.L,
            self.source.H)
        
        #DB>>
        con_stats = []
        connections_post = np.array(self.connectivity,dtype=int  )[:,1]
        connections_gd   = np.array(self.connectivity,dtype=float)[:,3:]
        for post in range(self.target.num_neurons):
            ids, = np.where(connections_post == post)
            con_stats.append(ids.shape[0])
        con_stats = np.array(con_stats)
        
        print(f'Connection {self.name}')
        print(f' > number of postsinaptic neurons: {len(con_stats)}')
        print(f' > number of connections:')
        print(f'   > min   : {np.amin(  con_stats)}')
        print(f'   > mean  : {np.mean(  con_stats)}')
        print(f'   > median: {np.median(con_stats)}')
        print(f'   > max   : {np.amax(  con_stats)}')
        print(f' > synaptic conductance :')
        print(f'   > min   : {np.amin(  connections_gd[:,0])}')
        print(f'   > mean  : {np.mean(  connections_gd[:,0])}')
        print(f'   > median: {np.median(connections_gd[:,0])}')
        print(f'   > max   : {np.amax(  connections_gd[:,0])}')
        print(f' > synaptic delays:')
        print(f'   > min   : {np.amin(  connections_gd[:,1])}')
        print(f'   > mean  : {np.mean(  connections_gd[:,1])}')
        print(f'   > median: {np.median(connections_gd[:,1])}')
        print(f'   > max   : {np.amax(  connections_gd[:,1])}')
        #<<DB
    def connect(self):
        """connects synapses"""
        
        presynapses = [
            pre
            for pre, post, d, gsyn, delay in self.connectivity
                      ]
        
        postsynapses = [
            post
            for pre,post,d,gsyn,delay in self.connectivity
                       ]
        self.synapses.connect(i=presynapses, j=postsynapses)
        
        self.synapses.gsyn = [
            gsyn
            for pre,post,d,gsyn,delay in self.connectivity
                             ]
        
        self.synapses.delay = [
            delay*ms
            for pre,post,d,gsyn,delay in self.connectivity
                              ]
        
        # if self.file['connect']:
        #     for key, value in self.file['connect'].items():
        #         if isinstance(value, str):
        #             value = eval(value, self.namespace)
        #             setattr(self.synapses, key, value)
        #             self.namespace[key] = value
        #         else:
        #             setattr(self.synapses, key, value)
        #             self.namespace[key] = value

class BrianSimulation:
    """Runs brian2 simulation"""

    def __init__(self, config_file: dict):
        """config_file - general dict recreated from YAML"""

        self.config_file = config_file
        self.sim_params = config_file['simulation']

    def _get_pops(self) -> dict:
        """Creates dict with population: instance"""

        pops = {}
        geometry = self.config_file['geometry']
        for pop in self.config_file['populations']:
            pops[pop] = Population(self.config_file['populations'][pop], population=pop, geometry=geometry)

        return pops

    def _get_syns(self) -> dict:
        """Creates dict with synapses: instance"""

        syns = {}
        for syn in self.config_file['synapses']:
            syns[syn] = Syns(self.config_file['synapses'][syn], self.pops, connection_name=syn)

        return syns

    def get_monitors(self) -> list:
        """Creates list of monitors for brian network simulation"""

        recordings = {}

        for pop in self.sim_params['pops_to_include']:
            if self.config_file['recording'][pop]['state']:
                recordings[pop] = BrianSimulationOutput(config_file=self.config_file, 
                                                        pop=pop, 
                                                        pops=self.pops, 
                                                        syns=self.syns, 
                                                        sim_params=self.sim_params)
        
        monitors = []
        for rec in recordings.keys():
            monitors.append(recordings[rec].stateM)
            monitors.append(recordings[rec].spikeM)

        return (monitors, recordings)

    def run(self):
        """Runs brian2 simulation based on YAML inputs"""

        n_runs = self.sim_params["n_runs"]
        print(f'Total number of simulation runs: {n_runs}')
        for i in range(n_runs):
            print(f'Creating model for simulation {i+1} of {n_runs}...')

            b2.defaultclock.dt = self.sim_params['time_step']*b2.ms
            ## clears any ghost objects left over from previous runs
            b2.device.reinit()
            b2.start_scope()

            self.pops = self._get_pops()
            self.syns = self._get_syns()
            active_neurons  = self.sim_params['pops_to_include']
            active_neurons  = [pop_obj[1].neurons for pop_obj in self.pops.items() if pop_obj[0] in active_neurons]
            active_synapses = self.sim_params['syns_to_include']
            active_synapses = [syn_obj[1].synapses for syn_obj in self.syns.items() if syn_obj[0] in active_synapses]
            monitors, recordings = self.get_monitors()

            if self.sim_params['connectivity']['save']:
                connectivity = {}
                """Each key in the dictionary is a synapse name, and the value is a numpy array of connectivity data;
                   First "column" in array is presynaptic neuron index, second is postsynaptic neuron index, third is distance, 
                   fourth is synaptic conductance, fifth is synaptic delay"""

                for name, syn in self.syns.items():
                    connectivity[name] = {
                                        'presynaptic_id': np.array(syn.connectivity)[:, 0],
                                        'postsynaptic_id': np.array(syn.connectivity)[:, 1],
                                        'distance': np.array(syn.connectivity)[:, 2],
                                        'conductance': np.array(syn.connectivity)[:, 3],
                                        'delay': np.array(syn.connectivity)[:, 4]
                                         }
                
                for name, con in connectivity.items():
                    file_name = f'{i+1}_' + f'connectivity_{name}' + self.sim_params['connectivity']['file_format']
                    if self.sim_params['connectivity']['annotation_state']:
                        annotation = textwrap.dedent(f"""\
                                            Connectivity for synapses: {name}.
                                            Created: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
                                            Used YAML file: {args.filename}.""")
                        annotation = self.sim_params['connectivity']['annotation'] + '\n' + annotation
                        con['annotation'] = annotation

                        print(f'Saving connectivity for synapses: {name}')
                        np.savez(file_name, **con)
            
            if self.sim_params['run']:
                net = b2.Network(active_neurons, active_synapses, *monitors)
                print(f'The model from file `{args.filename}` is fully created... running')
                net.run(self.sim_params['duration']*b2.ms, report='text')
                for pop in recordings.keys():
                    if self.config_file['recording'][pop]['save_res']:
                        output_file = recordings[pop].extract_results(prefix=f'{i+1}_')
                        print(f'Results for `{pop}` are saved into `{output_file}`')

class BrianSimulationOutput:
    """Configure the simulation parameters and recording settings"""

    def __init__(self, config_file: dict, pop: str, pops: dict, syns: dict, sim_params: dict):
        """pop - population of neurons found in the recording section of YAML"""

        self.config_file = config_file
        self.rec = self.config_file['recording'][pop]
        self.pop = pop
        self.pops = pops
        self.syns = syns
        self.sim_params = sim_params

        population = self.pops[pop]
        self.namespace = globals().copy()             # get access to the variables in the global space
        self.namespace.update(vars(population))       # add variadbles from Population instance
        self.namespace.update(self.rec['constants'])  # add user defined constants from YAML
        eqs = {}
        for k,v in self.rec['equations'].items():
            if isinstance(v, str):
                eqs[k] = eval(v, self.namespace)
            else:
                eqs[k] = v
        self.namespace.update(eqs)                    # add results of user defined equations from YAML

        self.stateM = self._set_state_monitor()
        self.namespace.update({'stateM': self.stateM})
        self.spikeM = self._set_spike_monitor()
        self.namespace.update({'spikeM': self.spikeM})

    def _set_state_monitor(self):
        """Set state monitor"""

        sm = self.rec['stateMonitor']
        variables = sm['variables']
        monitor_kwargs = {}
        

        for k, v in sm.items():                       # recollecting all parameters except variables into dict 
            if k == 'variables':
                continue
            if isinstance(v, str):
                try:
                    v = eval(v, self.namespace)
                    monitor_kwargs[k] = v
                except Exception as e:
                    raise ValueError(f"Failed to evaluate '{k}': '{v}'. Error: {e}")
            else:
                monitor_kwargs[k] = v

        return b2.StateMonitor(self.namespace['neurons'], variables, **monitor_kwargs)

    def _set_spike_monitor(self):
        """Set spike monitor"""

        monitor_kwargs = {}
        sm = self.rec['spikeMonitor']

        if self.rec['spikeMonitor']:
            for k, v in sm.items():
                if isinstance(v, str):
                    try:
                        v = eval(v, self.namespace)
                    except Exception as e:
                        raise ValueError(f"Failed to evaluate '{k}': '{v}'. Error: {e}")
                else:
                    monitor_kwargs[k] = v
        
        return b2.SpikeMonitor(self.namespace['neurons'], **monitor_kwargs)

    def _build_results(self):
        
        results = {}
        for k,v in self.rec['output'].items():
            if isinstance(v, str):
                results[k] = eval(v, self.namespace)
            else:
                results[k] = v
        return results

    def extract_results(self, prefix):
        results = self._build_results()

        file_name = prefix + self.pop + '_' + self.rec['file_name'] + self.rec['file_format']

        ann = self.rec['annotation']
        if ann['state']:
            annotation = textwrap.dedent(f"""\
                            Date of recording: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
                            Used YAML file: {args.filename}.
                            Recorded from {self.namespace['n_record']} neurons of `{self.pop}` population.
                            The simulation duration is {self.sim_params['duration']} ms. The time step is {self.sim_params['time_step']} ms.
                                        """)
            if ann['text']:
                annotation = ann['text'] + '\n' + annotation
            results['annotation'] = annotation

        np.savez(file_name, **results)
        return file_name
             


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Give script a file and get output from it")
    parser.add_argument("filename", type=str, help="The name or path of the YAML file to load")
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        f = yaml.safe_load(file)  
    print(f'Loaded a model file `{args.filename}`')

    import brian2 as b2

    simulation = BrianSimulation(f)
    simulation.run()
