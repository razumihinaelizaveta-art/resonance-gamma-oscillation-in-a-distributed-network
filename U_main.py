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
    def get_brian_object(self):
        return [ self.neurons ] 

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
    def get_brian_object(self):
        return [ self.synapses ] 
        # if self.file['connect']:
        #     for key, value in self.file['connect'].items():
        #         if isinstance(value, str):
        #             value = eval(value, self.namespace)
        #             setattr(self.synapses, key, value)
        #             self.namespace[key] = value
        #         else:
        #             setattr(self.synapses, key, value)
        #             self.namespace[key] = value

class Recorder:
    def __init__(self, rectype:str, recname:str, config: dict, pops:dict, syns:dict):
        """
            rectype - type of recorder: spikes, static, or states
            recname - name of the record variable
            config  - recorder configuration
            pops    - dict of populations
            syns    - dict of synaptic connections between populations
        """
        
        self.config  = config
        self.rectype = rectype
        self.recname = recname
        if   rectype == 'spikes':
            self._set_spike_monitor(config, pops, syns)
        elif rectype == 'static':
            self._set_static_record(config, pops, syns)
        elif rectype == 'states':
            self._set_state_monitor(config, pops, syns)
        else:
            raise RuntimeError(f"Unknown type of recorder `{rectype}` for recorder `{self.recname}`" )

    def _set_spike_monitor(self, config: dict, pops:dict, syns:dict):
        if not 'population' in config:
            raise RuntimeError(f"There is no `population` entrance in spike recorder `{self.recname}`")
        if not config['population'] in pops:
            raise RuntimeError(f" Population {config['population']} needed for spike recorder `{self.recname}` is not in the network" )
        self.recorder = SpikeMonitor( pops[ config['population'] ].neurons )

    def _set_static_record(self, config: dict, pops:dict, syns:dict):
        objs = syns | pops
        if not 'object' in config:
            raise RuntimeError(f"There is no `object` entrance in static recorder `{self.recname}`")
        if not config['object'] in objs:
            raise RuntimeError(f" object {config['object']} needed for static recorder `{self.recname}` is not in the network" )
        if not 'variable' in config:
            raise RuntimeError(f"There is no `variable` entrance in static recorder `{self.recname}`")
        self.namespace = globals().copy() | objs[ config['object'] ].namespace
        self.recorder  = eval(config['variable'], self.namespace )

    def _set_state_monitor(self, config: dict, pops:dict, syns:dict):
        objs = syns | pops
        if not 'object' in config:
            raise RuntimeError(f"There is no `object` entrance in states recorder `{self.recname}`")
        if not config['object'] in objs:
            raise RuntimeError(f" object {config['object']} needed for states recorder `{self.recname}` is not in the network" )
        
        self.namespace = globals().copy() | objs[ config['object'] ].namespace.copy()

        if not 'variable' in config:
            raise RuntimeError(f"There is no `variable` entrance in states recorder `{self.recname}`")
        if not type(config['variable']) is str:
            raise RuntimeError(f"The `variable` entrance is not a string in states recorder `{self.recname}`")
        
        # Checking states
        if not 'states' in config:
            raise RuntimeError(f"There is no `states` entrance in states recorder `{self.recname}`")
        if not type(config['states']) is list:
            raise RuntimeError(f"The `states` entrance is not a list in states recorder `{self.recname}`")
        for ni,n in enumerate( config['states'] ):
            if not type(n) is str:
                raise RuntimeError(f"The entrance #{ni+1} in `states` list of the states recorder `{self.recname}` is not a string")
        self.states = config['states']
        
        # Check record
        if not 'record' in config:
            raise RuntimeError(f"There is no `record` entrance in states recorder `{self.recname}`")
        if type(config['record']) is list:
            for ni,n in enumerate( config['record'] ):
                if not type(n) is int:
                    raise RuntimeError(f"The entrance #{ni+1} in `record` list of the states recorder `{self.recname}` is not an integer")
            self.record = config['record']
        elif type(config['record']) is str:
            try:
                self.record = eval(config['record'],self.namespace)
                # self.record = [ int(r) for r in self.record ]
                # #DB>>
                # print(f'{self.recname}.record = {self.record}')
                # #<<DB
            except BaseException as e:
                raise RuntimeError(f"Cannot convert `record` of the states recorder `{self.recname}` into an object: {e}")
        elif type(config['record']) is bool:
            pass
        else:
            raise RuntimeError(f"Incorrect type of entrance `record` for the states recorder `{self.recname}`. It should be a list of integers, boolian, or string")
        self.recorded_object = eval(config['variable'],self.namespace)

        # Check scales:
        if not 'scales' in config:
            self.scales = [ 1.0 for n in self.states]
        elif not type(config['scales']) is list:
            raise RuntimeError(f"Incorrect type of entrance `scales` for the states recorder `{self.recname}`. It should be a list of floats")
        elif len(config['scales']) != len(self.states):
            raise RuntimeError(f"Size of list `scales` does not match size of list states for the states recorder `{self.recname}`")
        for scli,scl in enumerate(config['scales']):
            if type(scl) is str:
                config['scales'][scli] = eval(scl, self.namespace)
            scl = config['scales'][scli]
            # if not (type(scl) is float or type(scl) is int):
                # raise RuntimeError(f"Entrance #{scli+1} in the list `scales` is not a float for the states recorder `{self.recname}`")
        self.scales = config['scales']
        
        self.recorder = StateMonitor(
            self.recorded_object,
            self.states, 
            record=self.record)

    def get_brian_object(self):
        if   self.rectype == 'static': return []
        elif self.rectype == 'spikes': return [ self.recorder ]
        elif self.rectype == 'states': return [ self.recorder ]

    def serialize(self):
        if   self.rectype == 'static':
            return {self.recname:self.recorder}
        elif self.rectype == 'spikes':
            return {
                self.recname:
                np.column_stack( (self.recorder.t/ms, self.recorder.i) )
            }
        elif self.rectype == 'states': 
            return {
                self.recname+'_ids'   : self.record, 
                self.recname+'_states': self.states,
                self.recname+'_time'  : self.recorder.t/ms,
            } | { self.recname+'_'+n: getattr(self.recorder,n).T*s for n,s in zip(self.states,self.scales) }

class Record:
    def __init__(self, config: dict, pops:dict, syns:dict, model:str):
        """
            config  - entire configuration of all recordings
            pops    - dict of populations
            syns    - dict of synaptic connections between populations
            model   - name of model YAML file
        """
        if not 'filename' in config:
            self.filename = None
            from warnings import warn
            warn("There is no file name in the record section of YAML. All simulation results will be save in memory and lost if not plotted")
        elif not type( config['filename'] ) is str:
            raise BaseException('`filename` is not a string')
        else:
            self.filename = config['filename']
        self.annotation = config.get('annotation',None)
        self.model      = config.get('model', True)
        self.map        = config.get('map', True)
        self.modelfile  = model
        self.config     = config
        self.recoders   = [
                            Recorder(
                                'spikes',
                                recname, 
                                config['spikes'][recname],
                                pops , syns)
                            for recname in config['spikes']
                        ] if 'spikes' in config else []
        self.recoders  += [
                            Recorder(
                                'static',
                                recname, 
                                config['static'][recname],
                                pops , syns)
                            for recname in config['static']
                        ] if 'static' in config else []
        self.recoders  += [
                            Recorder(
                                'states',
                                recname, 
                                config['states'][recname],
                                pops , syns)
                            for recname in config['states']
                        ] if 'states' in config else []
    
    def get_brian_object(self):
        ret = []
        for rec in self.recoders:
            ret += rec.get_brian_object()
        return ret
    
    def serialize(self ):
        ret = {}
        
        for rec in self.recoders:
            ret = ret | rec.serialize()
            
        if self.annotation:
            ret['annotation'] = self.annotation
        if self.model:
            with open(self.modelfile) as fd:
                ret['model'] = fd.read()
        if self.map:
            ret['map'] = [ n for n in ret ] 
        if self.filename:
            np.savez(self.filename, **ret)
            print(f"Save results in to `{self.filename}`")
        return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Give script a file and get output from it")
    parser.add_argument("filename", type=str, help="The name or path of the YAML file to load")
    parser.add_argument("-o","--output", type=str, default=None, help="The name of the .npz file to store data in")
    parser.add_argument("--do-not-simulate", default=True, action='store_false', help="Set to abort simulation from CLI")
    parser.add_argument("--save-only"    , default=False, action='store_true', help="Save npz file and exit")
    parser.add_argument("--not-shown"    , default=False, action='store_true', help="do not show anything on the screen, but export")
    parser.add_argument('--export-only'  , default=False, action='store_true', help="Export figures and exit")
    parser.add_argument('--export-video' , default=False, action='store_true', help="Export Video and exit")

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        f = yaml.safe_load(file)  
    print(f'Loaded a model file `{args.filename}`')

    # extract simulation and recording parameters from the YAML file
    simulation_params = f['simulation']
    record_params     = f['record']
    if args.output:
        record_params['filename'] = args.output
        

    pops = {}
    syns = {}
    geometry = f['geometry']

    for pop in f['populations']:
        pops[pop] = Population(f['populations'][pop], population=pop, geometry=geometry)
        #test = Population(f['populations'][pop], population=pop, geometry=geometry)

    for syn in f['synapses']:
        syns[syn] = Syns(f['synapses'][syn], pops, connection_name=syn)
        #test = Syns(f['synapses'][syn], neurons=pops, connection_name=syn)

    recorder = Record(record_params,pops,syns,args.filename)


    import brian2 as b2
    if 'time_step' in simulation_params:
        time_step = simulation_params['time_step']
        b2.defaultclock.dt = time_step*b2.ms

    # clears any ghost objects left over from previous cell runs
    b2.device.reinit()
    b2.start_scope()
    brain_objects = []
    for pop in pops:
        brain_objects += pops[pop].get_brian_object()
    for syn in syns:
        brain_objects += syns[syn].get_brian_object()
    brain_objects += recorder.get_brian_object()
    net = b2.Network(brain_objects)
    print(f'The model from file `{args.filename}` is fully created... running')
    
    duration = simulation_params.get('duration',100)
    sim_run  = simulation_params.get('run',True) and args.do_not_simulate
    if not sim_run:
        print("Simulation is aborted, relax!")
        exit(0)
    net.run(duration*b2.ms, report='text')
    
    allrecordings = recorder.serialize()
    if args.save_only:
        exit(0)
    
    if 'visualize' in f:
        visual_config = f['visualize']
        if args.not_shown:
            args.export_only  = True
            args.export_video = True
            
        from graphics import Visualizer
        visualizer = Visualizer(visual_config, allrecordings, export_only=args.export_only, video=args.export_video)
    
