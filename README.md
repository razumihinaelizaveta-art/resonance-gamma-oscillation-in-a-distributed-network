# Resonance Gamma Oscillation in a Distributed Network

Simulation and analysis of gamma-range network activity in a spatially distributed neuronal network.

This repository provides tools for constructing neuronal networks from YAML configuration files, running simulations with [Brian2](https://brian2.readthedocs.io/), analyzing spike trains using mutual information, and visualizing spatial patterns of neuronal activity.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Scripts](#scripts)
    - [Run a simulation](#1-run-a-simulation)
    - [Compute mutual information](#2-compute-mutual-information)
    - [Visualize network activity](#3-visualize-network-activity)
- [YAML Model Structure](#yaml-model-structure)

## Overview

The project models a population of neurons distributed across a two-dimensional spatial domain. Network structure, neuronal populations, external drive, and synaptic connectivity can be configured through YAML model files.

The simulation pipeline produces `.npz` datasets containing neuronal positions and spike times, which can then be used for information-theoretic analysis and visualization.

The main workflow is:

```text
YAML configuration
       │
       ▼
   Network setup
       │
       ▼
  Brian2 simulation
       │
       ▼
    NPZ output
       │
       ├───────────────┐
       ▼               ▼
Mutual information   Visualization
    analysis         and animation
```

## Requirements

The project requires Python and the following main packages:

* [Brian2](https://brian2.readthedocs.io/)
* NumPy
* SciPy
* Matplotlib
* PyYAML
* ButtsMI

Additional dependencies may be required by individual visualization or analysis scripts.

## Installation

Clone the repository:

```bash
git clone https://github.com/razumihinaelizaveta-art/resonance-gamma-oscillation-in-a-distributed-network.git
cd resonance-gamma-oscillation-in-a-distributed-network
```

## Scripts

### 1. Run a simulation

`main.py` builds and runs a neuronal network defined by a YAML configuration file.

#### Command

```bash
python main.py <yaml_file> <output_file>
```

where:

* `<yaml_file>` — path to the YAML model configuration;
* `<output_file>` — name or path of the `.npz` file in which the simulation results will be saved.

#### Example

```bash
python main.py yamls/large-PFC.yaml out.npz
```

This command loads the network configuration from:

```text
yamls/large-PFC.yaml
```

and saves the simulation results to:

```text
out.npz
```

#### Simulation parameters

Parameters that are hardcoded into `main.py`:

```text
Simulation duration: 1500 ms
Brian2 timestep:      0.05 ms
Recorded neurons:     30
Raster plot range:    0–500 ms
```

#### Output

The generated `.npz` file contains:

* `positions` — 2D coordinates of all neurons;
* `spikes` — spike times and neuron indices;
* `voltages` — membrane-potential traces for the recorded neurons;
* `voltage_id` — IDs of the recorded neurons;
* `features` — the `I0` value associated with each neuron.

The generated `.npz` file can then be passed to the mutual-information analysis and visualization scripts described below.


### 2. Compute mutual information

`computeMI.py` calculates mutual information between neuronal features and spike-train activity using the `ButtsMI` package.

#### Command

```bash
python computeMI.py <input> [options]
```

The input file must be an `.npz` file containing at least:

* the feature array specified with `--feature-array`;
* `spikes`.

#### Arguments

| Argument               | Type    | Default        | Description                                                         |
| ---------------------- | ------- | -------------- | ------------------------------------------------------------------- |
| `input`                | `str`   | —              | **Required.** Path to the input `.npz` file                         |
| `--feature-array`      | `str`   | `positions`    | Name of the array in the `.npz` file to use as the neuronal feature |
| `--remove-first`       | `float` | `0.0`          | Remove the first N milliseconds from the analysis                   |
| `--remove-last`        | `float` | `0.0`          | Remove the last N milliseconds from the analysis                    |
| `--keep-neurons`       | `int`   | `None`         | Use only the first N neurons                                        |
| `--output`             | `str`   | `results.json` | Path of the output JSON file                                        |
| `--shuffle-iterations` | `int`   | `0`            | Number of shuffled-feature control iterations                       |

#### Feature array

By default, the script uses the `positions` array as the feature:

```bash
python computeMI.py out.npz
```

A different array can be selected with `--feature-array`.

For example, to use the `features` array:

```bash
python computeMI.py out.npz --feature-array features
```

The specified array must exist in the input `.npz` file.


#### Output file

By default, results are saved to:

```text
results.json
```

A different output path can be specified with:

```bash
--output <filename>
```

For example:

```bash
python computeMI.py out.npz --output mi_results.json
```

The output JSON contains information about the analysis, including:

```json
{
    "input_file": "out.npz",
    "feature_array": "positions",
    "remove_first_ms": 0.0,
    "remove_last_ms": 0.0,
    "neurons_requested": null,
    "neurons_used": 2500,
    "spikes_total": 14848,
    "spikes_used": 14848,
    "silent_neurons": 366,
    "single_spike_neurons": 120,
    "usable_neurons": 2014,
    "mutual_information": 0.123,
    "shuffle_iterations": 100,
    "shuffle_mean": 0.045,
    "shuffle_std": 0.008,
    "shuffled_mutual_information_values": [...],
    "status": "success"
}
```

The numerical values above are illustrative; the actual values depend on the input dataset.



### 3. Visualize network activity

#### Plotting spatial activation 
`plotting_functions.py` provides an interactive visualization of spatial neuronal activity and membrane-potential traces from a simulation `.npz` file.

##### Command

```bash
python plotting_functions.py <data_file> [options]
```

##### Arguments

| Argument       | Type    | Default               | Description                                                      |
| -------------- | ------- | --------------------- | ---------------------------------------------------------------- |
| `data_file`    | `str`   | —                     | **Required.** Path to the input `.npz` simulation file           |
| `--window`     | `float` | `0.5`                 | Time window represented by each animation frame, in milliseconds |
| `--save-video` | flag    | `False`               | Save the animation as an MP4 file                                |
| `--output`     | `str`   | `spike_animation.mp4` | Output filename for the saved MP4 animation                      |
| `--quiet`      | flag    | `False`               | Suppress informational messages printed by the script            |


The window can also be changed interactively while the visualization is running.

The allowed range is:

```text
Minimum: 0.1 ms
Maximum: 20.0 ms
Step:    0.1 ms
```

Use `--save-video` to save the animation as an MP4 file. By default, the animation is saved as:

```text
spike_animation.mp4
```

A custom output filename can be specified with `--output`:

```bash
python plotting_functions.py out.npz \
    --save-video \
    --output network_activity.mp4
```

The video is encoded using **FFmpeg**, which must therefore be installed and available in the system PATH.


##### Interactive controls

Once the visualization is open, the simulation can be controlled using the time slider, Play button, and keyboard.

| Control         | Action                            |
| --------------- | --------------------------------- |
| **Time slider** | Move directly to a specific frame |
| **Play button** | Start or pause the animation      |
| `→`             | Move forward by 1 frame           |
| `←`             | Move backward by 1 frame          |
| `↑`             | Move forward by 10 frames         |
| `↓`             | Move backward by 10 frames        |
| `Space`         | Play/Pause                        |
| `+` / `=`       | Increase the animation window     |
| `-` / `_`       | Decrease the animation window     |
| `>`             | Increase neuron marker size       |
| `<`             | Decrease neuron marker size       |

When the window size changes, the visualization recalculates the animation frames and attempts to preserve the current simulation time.

##### Visualization colors

The spatial neuron plot uses different colors to distinguish neuronal state:

| Color     | Meaning                                       |
| --------- | --------------------------------------------- |
| Gray      | Non-spiking neuron                            |
| Red       | Spiking neuron                                |
| Dark gray | Recorded neuron that is not currently spiking |
| Yellow    | Recorded neuron that is currently spiking     |

Recorded neurons are identified using the `voltage_id` array from the input dataset.

### Visualize neuronal features

`plotting_functions_features.py` provides an interactive spatial visualization of neuronal activity while simultaneously displaying the distribution of a neuronal feature across the network.

#### Command

```bash 
python plotting_functions_features.py <data_file> [options]
```

#### Arguments

| Argument       | Type    | Default               | Description                                                      |
| -------------- | ------- | --------------------- | ---------------------------------------------------------------- |
| `data_file`    | `str`   | —                     | **Required.** Path to the input `.npz` simulation file           |
| `--window`     | `float` | `1.0`                 | Time window represented by each animation frame, in milliseconds |
| `--save-video` | flag    | `False`               | Save the animation as an MP4 file                                |
| `--output`     | `str`   | `spike_animation.mp4` | Output filename for the saved MP4 animation                      |
| `--quiet`      | flag    | `False`               | Suppress informational messages                                  |


The default animation window is `1.0 ms`.

The `features` array is automatically loaded from the input `.npz` file:

The minimum and maximum feature values are used to normalize the feature distribution, and the `cool` Matplotlib colormap is used to assign colors to neurons.

The feature value is represented by the **edge color** of each neuron. A colorbar is displayed next to the spatial network.

#### Spike visualization

During each animation frame:

| Neuron state              | Appearance          |
| ------------------------- | ------------------- |
| Non-recorded, not spiking | Feature-based color |
| Non-recorded, spiking     | White               |
| Recorded, not spiking     | Dark gray           |
| Recorded, spiking         | Yellow              |

For non-recorded neurons, the feature-dependent edge color remains visible while spiking neurons temporarily flash white.

Recorded neurons are identified using the `voltage_id` array.

#### Interactive controls

The visualization can be controlled using the slider, Play button, and keyboard in the same way as `plotting_functions.py`.


### Visualize wave propagation

`wave_player.py` provides an interactive visualization of spiking activity across the neuronal sheet. Neurons are displayed at their spatial `(x, y)` positions, while a moving firing window and activity trail visualize the spatial organization of population activity. The player is designed to help distinguish synchronized population volleys from spatially propagating waves.

#### Command

```bash
python wave_player.py [data_file]
```

If no file is specified, the script uses `out.npz`.

The input `.npz` file must contain:

- `positions` — neuronal `(x, y)` coordinates;
- `spikes` — spike times and neuron IDs.

#### Parameters

The main playback parameters are configured directly in the script:

| Parameter | Default | Description |
|---|---:|---|
| `T_START` | `None` | Start time of the displayed interval |
| `T_END` | `None` | End time of the displayed interval |
| `BURST_DT` | `0.10 ms` | Temporal step during high-activity volleys |
| `QUIET_SPEEDUP` | `8` | Speed-up factor during low-activity periods |
| `WINDOW` | `0.60 ms` | Width of the firing window |
| `TRAIL` | `6` | Number of previous activity windows displayed |
| `FPS` | `20` | Playback frame rate |
| `SUBSAMPLE_N` | `None` | Number of neurons to display; `None` shows all |
| `MIN_NEURON_SPIKES` | `0` | Minimum number of spikes required for a neuron to be displayed |
| `BURST_THRESHOLD` | `'auto'` | Population activity threshold used to identify volleys |

These parameters are not exposed as command-line flags and are modified directly in `wave_player.py`.

#### Visualization

The main panel shows:

- inactive neurons as light background dots;
- currently firing neurons in red;
- previous activity as a fading trail.

A population activity strip below the network shows the firing activity across the complete simulation. The current simulation time is marked by a red cursor.

During high population activity, playback advances slowly to show the temporal structure of each volley. During low-activity periods, playback is automatically accelerated.

#### Interactive controls

| Control | Action |
|---|---|
| **Time slider** | Move to a specific frame |
| **Play** | Start or pause playback |
| **< Step** | Move backward by one frame |
| **Step >** | Move forward by one frame |

Playback loops back to the beginning after reaching the final frame.


## YAML Model Structure

The configuration defines the simulation geometry, neuronal populations, model constants, differential equations, neuron specifications, initial conditions, synaptic connections, and optional recorders.

The expected top-level structure is:

```yaml
geometry:
    ...

populations:
    ...

synapses:
    ...

recorder:
    ...
```

### Geometry

The `geometry` section defines the spatial dimensions of the simulated neuronal sheet.

| Parameter | Description |
|---|---|
| `L` | Length of the simulated area |
| `H` | Height of the simulated area |

### Neuronal populations

The `populations` section defines the neuronal populations in the model.

For example:

```yaml
populations:
    pvbc:
        dencity: 3600e-6*1.5
        num_neurons: L*H*dencity
```

`num_neurons` can either be calculated from the spatial dimensions and population density or specified directly.


### Model constants

Population-specific constants are defined under `const`.

```yaml
populations:
    pvbc:
        ...

        const:
            eps: 1.e-9*mV
            FactorScaleKV3: 1

            norm_syn: 1*msiemens
            tau_syn_a: 0.8
            tau_syn_b: 2

            ENa: 50*mV
            EK: -90*mV
            Esyn: -75*mV

            gL: 14.705184211981187*nS
            EL: -71.97508650437244*mV
            taum: 5.222681687884308310*ms
            Rin: 1.0e3/(gL/nS)*Mohm
            Cap: taum/Rin

            gNa: 16804.9678*nS
            gKv3: 631.700393*nS
            gKv1: 59.0431313*nS
```

Constants can contain numerical values, physical units, or formulas involving other constants.

### Equations

The `equations` section contains the differential equations and other mathematical expressions defining the neuronal model.

```yaml
equations:

    gat_vars: |
        alpham = 1.0/exprel(-(v-thm1-eps)/sigm1)/ms : Hz
        betam = km2*exp(v/sigm2)/ms : Hz
        dm/dt = alpham*(1-m) - betam*m : 1

        INa = -gNa*m**3*h*(v-ENa) : ampere
        IKv3 = -gKv3*n**4*(v-EK) : ampere
        IKv1 = -gKv1*a**4*(v-EK) : ampere
        IL   = -gL*(v-EL) : ampere

    ext_eqs: |
        I : ampere

    eqs_syn: |
        Isyn = norm_syn*(synb-syna)*(Esyn-v) : amp
        dsyna/dt = -syna/tau_syn_a/ms : 1
        dsynb/dt = -synb/tau_syn_b/ms : 1

    volt_eqs: |
        dv/dt = (INa + IKv1 + IKv3 + IL + I + Isyn)/Cap : volt

    general_equ: volt_eqs + gat_vars + ext_eqs + eqs_syn
```

The equation blocks are combined into a single equation set using:

```yaml
general_equ: volt_eqs + gat_vars + ext_eqs + eqs_syn
```

### Neuron specifications

The `specs` section defines the numerical integration method, spike threshold, and refractory condition.

| Parameter | Description |
|---|---|
| `method` | Numerical integration method |
| `threshold` | Condition used to detect a spike |
| `refractory` | Condition defining the refractory state |

### Initial conditions

The `initials` section defines the initial state of the neurons.

Initial values can contain constants, formulas, random values, and physical units.

### Synapses

The `synapses` section defines connections between neuronal populations.

```yaml
synapses:
    pv2pv:
        source: 'pvbc'
        target: 'pvbc'

        general:
            model: 'gsyn:1'

            on_pre: |
                syna_post += gsyn
                synb_post += gsyn

        geometry:
            type: 'lognormal'
            mu: 5.0063
            sigma: 0.5630

        conductance:
            type: 'exponential'
            max: 0.00005
            sigma: 70

        delay:
            type: 'linear'
            min: 0.1
            k: 50
```

The `source` and `target` fields must correspond to population names defined in the `populations` section.

### Recorders

Recording options can be enabled through the optional `recorder` section.

For example:

```yaml
recorder:
    spikes:
        populations: ['pvbc']

    states:
        voltages:
            population: 'pvbc'
            ids: rnd.randint(0,num_neurons,10)
```

The recorder can be used to save:

- spike events;
- membrane potentials;
- selected neuronal states;
- recordings from selected neurons.

