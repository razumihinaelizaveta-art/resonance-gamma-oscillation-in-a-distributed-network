# Network Visualization Specification v1.0

## Overview

The visualization framework is designed as an interactive research workspace for exploring network simulations stored in NPZ files.

The framework must support:

- spike propagation visualization in network space,
- multi-population raster plots,
- continuous-variable recordings,
- synchronized multi-figure visualization,
- frame-by-frame exploration,
- animation playback,
- figure export,
- video export,
- headless execution,
- interactive GUI controls,
- standalone execution from NPZ files.

All figures share a common global simulation time.

---

# Design Principles

1. Single public API:

```python
visualize(config, data)
```

2. All figures are synchronized through a single global time cursor.

3. A researcher should be able to distribute figures across multiple monitors.

4. The active time window used by spatial propagation must be visible on every plot type.

5. Playback controls must live in a dedicated control window and never appear in exported figures or videos.

6. Interactive and batch execution use exactly the same rendering code.

---

# Entry Point

The complete visualization system is accessed through:

```python
visualize(config: dict, data: dict)
```

Arguments:

```python
config
```

Visualization configuration dictionary.

```python
data
```

Dictionary containing all NPZ arrays.

Example:

```python
npz = np.load("record.npz", allow_pickle=True)

data = dict(npz)

visualize(config, data)
```

---

# Standalone Execution

The visualizer must also operate as a standalone program.

```bash
python visualize.py record.npz
```

Execution sequence:

1. Load NPZ.
2. Convert NPZ to dictionary.
3. Extract model YAML from NPZ.
4. Parse YAML.
5. Extract `visualize`.
6. Call:

```python
visualize(config, data)
```

The standalone entry point must not use a different code path from the API.

---

# Top-Level YAML Structure

```yaml
visualize:

    playback:
        ...

    export:
        ...

    figures:
        ...
```

---

# Playback

Global playback configuration.

```yaml
playback:

    t_start : 0
    t_stop  : null

    dt      : 1.0
    fps     : 30

    tail_time        : 20
    tail_exponential : false
```

---

## t_start

Starting simulation time.

```yaml
t_start : 0
```

Units:

```text
milliseconds
```

---

## t_stop

Playback stop time.

```yaml
t_stop : 5000
```

or

```yaml
t_stop : null
```

for full simulation duration.

---

## dt

Initial playback step size.

```yaml
dt : 1.0
```

Units:

```text
milliseconds
```

The user may subsequently increase or decrease the step size interactively.

---

## fps

Playback frame rate.

```yaml
fps : 30
```

Used for:

- animation playback,
- video generation.

---

## tail_time

Duration a spike remains visible.

```yaml
tail_time : 20
```

Units:

```text
milliseconds
```

A spike remains visible while:

```text
current_time - tail_time
            <=
spike_time
            <=
current_time
```

---

## tail_exponential

Controls spike-tail decay.

Linear fade:

```yaml
tail_exponential : false
```

```text
alpha = 1 - age/tail_time
```

---

Exponential fade:

```yaml
tail_exponential : true
```

```text
alpha = exp(-age/τ)
```

τ is implementation-defined.

---

# Global Time Model

The visualizer maintains:

```python
current_time
```

All figures update simultaneously whenever it changes.

---

# Active Time Window

The active interval is:

```text
[current_time-tail_time,current_time]
```

This interval drives:

- spatial propagation,
- raster highlighting,
- trace highlighting.

---

# Playback Control Window

The visualizer shall create a dedicated control figure when:

```yaml
show : true
```

This figure is NOT listed in YAML.

The control window is never:

- exported,
- saved as an image,
- included in videos.

---

## Control Layout

Conceptually:

```text
+--------------------------------------------------+
| << | < | Play/Pause | > | >>                    |
+--------------------------------------------------+

Current time: 1250 ms

Step size: 5 ms

[-] 5 ms [+]

+--------------------------------------------------+
|====================o============================ |
+--------------------------------------------------+
```

---

## Controls

### `<<`

Jump to beginning.

```python
current_time = t_start
```

---

### `<`

Step backward.

```python
current_time -= step_size
```

---

### `Play / Pause`

Toggle animation.

---

### `>`

Step forward.

```python
current_time += step_size
```

---

### `>>`

Jump to end.

```python
current_time = t_stop
```

---

### `-`

Decrease step size.

---

### `+`

Increase step size.

---

### Slider

Move directly to arbitrary simulation time.

---

# Keyboard Shortcuts

### Left Arrow

Step backward.

---

### Right Arrow

Step forward.

---

### Home

Jump to simulation start.

---

### End

Jump to simulation end.

---

### Space

Play / pause.

---

### +

Increase step size.

---

### -

Decrease step size.

---

# Export

```yaml
export:

    show : true

    path : results/

    figures : true

    formats:

        - png
        - svg

    snapshot_time : 1500

    video : true

    video_prefix : propagation
    video_format : mp4

    video_fps : 30

    t_start : null
    t_stop  : null
```

---

## show

Display interactive GUI.

```yaml
show : true
```

---

Batch mode:

```yaml
show : false
```

Runs without showing any window.

---

## path

Output directory.

```yaml
path : results/
```

---

## figures

Export static figure images.

```yaml
figures : true
```

---

## formats

Figure formats.

```yaml
formats:

    - png
    - svg
    - pdf
```

---

## snapshot_time

Simulation time used for exported images.

```yaml
snapshot_time : 1500
```

Before export:

```python
current_time = snapshot_time
```

---

## video

Enable video export.

```yaml
video : true
```

---

## video_prefix

Video filename prefix.

```yaml
video_prefix : propagation
```

---

## video_format

Video file format.

```yaml
video_format : mp4
```

---

## video_fps

Video frame rate.

```yaml
video_fps : 30
```

---

## Export Time Range

```yaml
t_start : 1000
t_stop  : 2000
```

Only this interval is exported.

---

# Video Export

Each figure produces its own video.

Example:

```yaml
video_prefix : propagation
```

and

```yaml
figures:

    activity:
    voltages:
    currents:
```

produces:

```text
results/

    propagation_activity.mp4

    propagation_voltages.mp4

    propagation_currents.mp4
```

The playback-control window is excluded.

---

# Figures

Each entry corresponds directly to:

```python
matplotlib.figure.Figure
```

Example:

```yaml
figures:

    activity:
        ...

    voltages:
        ...
```

Figures may be displayed on separate monitors.

---

# Figure Definition

```yaml
activity:

    size : [12,8]

    grid:

        rows : 4
        cols : 4

    plots:
        ...
```

---

## size

Maps directly to:

```python
plt.figure(figsize=(width,height))
```

Units:

```text
inches
```

---

## grid

Defines subplot grid.

```yaml
grid:

    rows : 4
    cols : 4
```

Used with:

```python
subplot2grid()
```

---

# Plot Placement

All plot types use the same placement model.

```yaml
position : [column,row]

width  : 3
height : 2
```

---

## position

Upper-left grid cell.

```yaml
position : [0,0]
```

Format:

```yaml
[column,row]
```

---

## width

Number of occupied columns.

---

## height

Number of occupied rows.

---

## Internal Mapping

```yaml
position : [c,r]

width  : w
height : h
```

maps to:

```python
subplot2grid(
    (rows, cols),
    (r, c),
    rowspan=height,
    colspan=width
)
```

---

# Spatial Plot

## Definition

```yaml
network:

    type : spatial

    position : [0,0]

    width  : 3
    height : 4

    spatial:
        ...
```

---

# Population Definition

```yaml
spatial:

    stl:

        positions : stl_positions
        spikes    : stl_spikes

        color     : stl_Iapp
        color_map : rainbow
```

---

## positions

Required.

Shape:

```text
N×2
```

or

```text
N×3
```

Columns:

```text
0 = x
1 = y
2 = z (optional)
```

Only x and y are displayed.

---

## spikes

Required.

Format:

```text
column 0 = spike time
column 1 = neuron id
```

---

# Spike Visibility

Visible if:

```text
current_time-tail_time
            <=
spike_time
            <=
current_time
```

---

# color

Constant color:

```yaml
color : red
```

or

```yaml
color : '#ff0000'
```

---

Variable-driven color:

```yaml
color : stl_Iapp
```

---

# color_map

Only used with variable-driven coloring.

Example:

```yaml
color     : stl_Iapp
color_map : rainbow
```

Any matplotlib colormap is valid.

---

# size

Marker size.

```yaml
size : 5
```

---

# alpha

Constant population transparency multiplier.

```yaml
alpha : 0.5
```

Final alpha:

```text
tail_alpha × population_alpha
```

---

# Raster Plot

## Definition

```yaml
raster:

    type : raster

    position : [3,0]

    width  : 1
    height : 4

    raster:

        pvbc:

            spikes : pvbc_spikes

            color  : red
            marker : '|'
            size   : 6

        stl:

            spikes : stl_spikes

            color  : black
            marker : '.'
            size   : 2
```

---

## Coordinate System

```text
x-axis = spike time

y-axis = automatically stacked neuron index
```

---

## spikes

Required.

```text
column 0 = spike time
column 1 = neuron id
```

---

## color

Marker color.

---

## marker

Examples:

```yaml
marker : '.'
marker : '|'
marker : 'o'
marker : '^'
marker : 's'
```

---

## size

Marker size.

---

# Automatic Population Stacking

Example:

```text
PVBC: 0-199

STL: 0-999
```

Displayed as:

```text
PVBC: 0-199

STL : 200-1199
```

Offsets are computed automatically.

---

# Raster Decorations

Automatically display:

- population labels,
- separators,
- current-time cursor,
- active time interval.

---

# Trace Plot

Used for:

- membrane voltages,
- gating variables,
- conductances,
- synaptic currents,
- arbitrary continuous recordings.

---

# Data Layout

For source:

```text
stl_states
```

and state:

```text
v
```

the visualizer expects:

```text
stl_states_v
stl_states_time
stl_states_id
```

---

## Variable Array

Shape:

```text
(num_time_samples,
 num_recorded_neurons)
```

---

## Time Array

Shape:

```text
(num_time_samples,)
```

---

## Neuron IDs

Shape:

```text
(num_recorded_neurons,)
```

Contains real neuron IDs.

---

# Trace Definition

```yaml
voltages:

    type : trace

    position : [0,0]

    width  : 1
    height : 2

    offset : auto

    traces:

        stl:

            source : stl_states
            state  : v

            color : black

            neurons : 'stl_states_id[:20]'

        pvbc:

            source : pvbc_voltages
            state  : v

            color : red

            neurons : 'pvbc_voltages_id[:10]'
```

---

# offset

Trace separation.

Automatic:

```yaml
offset : auto
```

Manual:

```yaml
offset : 10
```

Rendering:

```text
trace(i) + i*offset
```

---

# traces

Conta*ner holding all displayed trace so*rces.

Each trace source contribut*s to the same axes.

---

## sourc*

Recording source.

Example:

```*aml
source : stl_states
```

Loads*

```text
stl*states_<state>
stl_states_time
stl*states_id
```

---

## state

Reco*ded variable.

Examples:

```yaml
*tate : v
state : n
state : Iexc
st*te : Iinh
```

---

## color

Trac* color.

---

## neurons

Python e*pression evaluated against loaded *rrays.

Examples:

All recorded ne*rons:

```yaml
neurons : 'stl_stat*s_id'
```

First twenty:

```yaml
*eurons : 'stl_states_id[:20]'
```
*Every fifth:

```yaml
neurons : 's*l_states_id[::5]'
```

Conditional*selection:

```yaml
neurons : 'stl*states_id[stl_states_id < 100]'
``*

Expression must evaluate to a li*t or array of neuron IDs.

---

# *race Decorations

Automatically di*play:

- current-time cursor,
- ac*ive interval

```text
[current_tim*-tail_time,current_time]
```

matc*ing all other plot types.

---

# *atch Mode

```yaml
show : false
``*

Behavior:

```text
Load NPZ
Buil* figures
Export figures
Export vid*os
Exit
```

No windows opened.

-*-

# Interactive Mode

```yaml
sho* : true
```

Behavior:

```text
Op*n playback-control window
Open all*configured figures
Allow interacti*e exploration
Allow export operati*ns
```

---

# Internal Architectu*e Requirements

Recommended implem*ntation:

```python
visualize(conf*g, data)

    PlaybackController

*   FigureManager

        SpatialP*ot
        RasterPlot
        Trac*Plot

    ExportManager
```

---

* Mandatory Rule

All functionality* including:

- GUI visualization,
* batch export,
- figure generation*
- video generation,

must use the same rendering code path based on:

```python
visualize(config, data)
```

No duplicated rendering logic is allowed.
