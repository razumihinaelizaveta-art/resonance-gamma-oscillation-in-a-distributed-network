"""Interactive wave-propagation player  
Watch spiking activity move across the NEURON SHEET at each cell's real (x, y)
position. A "firing now" window sweeps through time so you can literally see
whether each gamma volley blinks everywhere at once (synchrony) or sweeps
across the sheet as a front (propagation).

This reads the out.npz produced by main.py  (needs 'positions' and 'spikes').

HOW TO RUN
----------
From the project folder, after you've made an out.npz:

    python wave_player.py out.npz

(If you leave the filename off, it defaults to "out.npz" in the current folder.)

You need an interactive matplotlib window (the normal Windows TkAgg backend
works out of the box). If you run it inside a Jupyter cell and see a static
image, switch to a pop-up backend first:  %matplotlib qt   (or tk)

CONTROLS
--------
  Play / Pause ............. start / stop playback
  < Step  /  Step > ........ move one frame back / forward (auto-pauses)
  Slider ................... drag to scrub through time (frame-based)
  Clock label .............. shows the true simulation time in ms
  Activity strip (bottom) .. whole-run population rate; red line = you are here
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ===========================================================================
#  PARAMETERS  -  edit these
# ===========================================================================
# File to play. Taken from the command line if given, else "out.npz".
FILE_PATH = sys.argv[1] if len(sys.argv) > 1 else "out.npz"

# --- which slice of the recording to play (milliseconds) ---
T_START = None        # None = start of recording
T_END   = None        # None = end of recording

# --- playback timing (all in ms) ---
# Tuned for continuous ~35-45 Hz gamma (cycle ~25 ms) rather than sparse bursts.
BURST_DT      = 0.10  # time advanced per frame WHILE IN A VOLLEY (small = slow-mo)
QUIET_SPEEDUP = 8     # between-volley troughs advance this many times faster
WINDOW        = 0.60  # width of the "firing now" window shown each frame (ms)
TRAIL         = 6     # fading trail steps behind the front (0 = off) - shows direction
FPS           = 20    # frames per second of the on-screen playback

# --- which neurons to show ---
SUBSAMPLE_N       = None   # None = all neurons, or an int to randomly thin them
MIN_NEURON_SPIKES = 0      # only show neurons that fire at least this many times

# --- how a "volley" is defined for the fast-forward logic ---
#   'auto' = mean + 1*std of the population rate, or set a number (spikes per bin)
BURST_THRESHOLD = 'auto'

# --- cosmetics ---
DOT_SIZE_BG   = 14        # bigger background dots (this sheet has ~121 neurons)
DOT_SIZE_FIRE = 60
CMAP_TRAIL    = 'Oranges'
COLOR_FIRE    = '#D8271E'


# ===========================================================================
#  LOAD  (expects positions Nx2 and spikes Mx2 as [time, neuron_id])
# ===========================================================================
def load(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"File not found: {path}\n"
            f"Usage: python wave_player.py <simulation.npz>\n"
            f"First produce one with:  python main.py test_fin.yaml out.npz"
        )
    data = np.load(path, allow_pickle=True)
    if 'positions' not in data or 'spikes' not in data:
        raise KeyError(
            f"{path} is missing 'positions' or 'spikes'. "
            "Make sure it was produced by main.py."
        )
    positions = data['positions'].astype(float)
    spikes_raw = data['spikes']
    n = positions.shape[0]
    per_neuron = [[] for _ in range(n)]
    for row in spikes_raw:
        per_neuron[int(row[1])].append(row[0])
    for i in range(n):
        per_neuron[i] = np.array(sorted(per_neuron[i]))
    return positions, per_neuron


positions, per_neuron = load(FILE_PATH)
n_all = positions.shape[0]

# --- apply the neuron filters ---
keep = [i for i in range(n_all) if len(per_neuron[i]) >= MIN_NEURON_SPIKES]
if SUBSAMPLE_N is not None and SUBSAMPLE_N < len(keep):
    rng = np.random.default_rng(0)
    keep = list(rng.choice(keep, SUBSAMPLE_N, replace=False))

positions = positions[keep]
per_neuron = [per_neuron[i] for i in keep]
n_show = len(keep)

# --- flatten all shown spikes into sorted parallel arrays (time, x, y) ---
all_t, all_x, all_y = [], [], []
for i in range(n_show):
    for tt in per_neuron[i]:
        all_t.append(tt)
        all_x.append(positions[i][0])
        all_y.append(positions[i][1])

if len(all_t) == 0:
    raise SystemExit(
        "No spikes in this file - nothing to play. "
        "Try a stronger drive I or stronger inhibition, then re-run main.py."
    )

all_t = np.array(all_t)
order = np.argsort(all_t)
all_t = all_t[order]
all_x = np.array(all_x)[order]
all_y = np.array(all_y)[order]

t_start = float(all_t.min()) if T_START is None else float(T_START)
t_end   = float(all_t.max()) if T_END   is None else float(T_END)

print(f"file: {FILE_PATH}")
print(f"neurons shown: {n_show} / {n_all}   spikes shown: {len(all_t)}")
print(f"time range played: {t_start:.1f} - {t_end:.1f} ms")


# ===========================================================================
#  POPULATION RATE  (for the activity strip AND the fast-forward logic)
# ===========================================================================
RATE_BIN = 0.5  # ms
rate_edges = np.arange(t_start, t_end + RATE_BIN, RATE_BIN)
rate_vals, _ = np.histogram(all_t, bins=rate_edges)
rate_centers = rate_edges[:-1] + RATE_BIN / 2.0

if BURST_THRESHOLD == 'auto':
    thr = rate_vals.mean() + rate_vals.std()
else:
    thr = float(BURST_THRESHOLD)


def rate_at(tau):
    """population rate in the 0.5 ms bin containing tau"""
    k = int((tau - rate_edges[0]) / RATE_BIN)
    k = min(max(k, 0), len(rate_vals) - 1)
    return rate_vals[k]


# ===========================================================================
#  VARIABLE-SPEED FRAME SCHEDULE
#  small steps inside volleys, big steps through troughs -> frame-based slider
# ===========================================================================
def build_schedule():
    quiet_dt = BURST_DT * QUIET_SPEEDUP
    frames = []
    tau = t_start
    while tau < t_end:
        frames.append(tau)
        tau += BURST_DT if rate_at(tau) > thr else quiet_dt
    return np.array(frames)


frame_times = build_schedule()
n_frames = len(frame_times)
print(f"total frames: {n_frames}  (volleys slow, troughs fast-forwarded {QUIET_SPEEDUP}x)")


# ===========================================================================
#  FIGURE + ARTISTS
# ===========================================================================
fig = plt.figure(figsize=(7.2, 8.4))
ax       = fig.add_axes([0.08, 0.30, 0.84, 0.64])   # main sheet
ax_strip = fig.add_axes([0.08, 0.17, 0.84, 0.08])   # activity strip
ax_slider = fig.add_axes([0.08, 0.09, 0.84, 0.03])  # slider
ax_play  = fig.add_axes([0.08, 0.02, 0.14, 0.05])
ax_back  = fig.add_axes([0.24, 0.02, 0.14, 0.05])
ax_fwd   = fig.add_axes([0.40, 0.02, 0.14, 0.05])

# --- main sheet ---
ax.scatter(positions[:, 0], positions[:, 1], s=DOT_SIZE_BG, color='#ececec', zorder=1)
trail_alphas = np.linspace(0.5, 0.08, TRAIL) if TRAIL > 0 else []
trail_artists = []
cmap = plt.get_cmap(CMAP_TRAIL)
for r in range(TRAIL):
    sc = ax.scatter([], [], s=DOT_SIZE_FIRE * 0.7, color=cmap(0.6),
                    alpha=trail_alphas[r], zorder=2)
    trail_artists.append(sc)
fire_artist = ax.scatter([], [], s=DOT_SIZE_FIRE, color=COLOR_FIRE,
                         edgecolors='white', linewidths=0.3, zorder=5)
pad = max(positions[:, 0].max() - positions[:, 0].min(), 1.0) * 0.03
ax.set_xlim(positions[:, 0].min() - pad, positions[:, 0].max() + pad)
ax.set_ylim(positions[:, 1].min() - pad, positions[:, 1].max() + pad)
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
clock = ax.set_title("", fontsize=13)

# --- activity strip ---
ax_strip.fill_between(rate_centers, rate_vals, color='#9aa7c7', lw=0)
ax_strip.axhline(thr, color='#c0392b', lw=0.8, ls='--', alpha=0.7)
cursor = ax_strip.axvline(t_start, color='#c0392b', lw=1.8)
ax_strip.set_xlim(t_start, t_end)
ax_strip.set_ylim(0, rate_vals.max() * 1.05)
ax_strip.set_yticks([])
ax_strip.set_xlabel("whole run (ms) - red line = current position, dashed = volley threshold",
                    fontsize=8)

# --- widgets ---
slider = Slider(ax_slider, 'frame', 0, n_frames - 1, valinit=0, valstep=1)
btn_play = Button(ax_play, 'Play')
btn_back = Button(ax_back, '< Step')
btn_fwd  = Button(ax_fwd, 'Step >')


# ===========================================================================
#  DRAW / STATE
# ===========================================================================
state = {'frame': 0, 'playing': False}


def draw_frame(i):
    tau = frame_times[i]
    lo = np.searchsorted(all_t, tau - WINDOW)
    hi = np.searchsorted(all_t, tau)
    if hi > lo:
        fire_artist.set_offsets(np.c_[all_x[lo:hi], all_y[lo:hi]])
    else:
        fire_artist.set_offsets(np.empty((0, 2)))
    for r in range(TRAIL):
        a = tau - WINDOW * (r + 2)
        b = tau - WINDOW * (r + 1)
        lo = np.searchsorted(all_t, a)
        hi = np.searchsorted(all_t, b)
        if hi > lo:
            trail_artists[r].set_offsets(np.c_[all_x[lo:hi], all_y[lo:hi]])
        else:
            trail_artists[r].set_offsets(np.empty((0, 2)))
    n_now = np.searchsorted(all_t, tau) - np.searchsorted(all_t, tau - WINDOW)
    clock.set_text(f"t = {tau:8.2f} ms      ({n_now} firing)")
    cursor.set_xdata([tau, tau])
    fig.canvas.draw_idle()


def on_slider(val):
    state['frame'] = int(val)
    draw_frame(state['frame'])


def advance():
    if state['playing']:
        nxt = state['frame'] + 1
        if nxt >= n_frames:
            nxt = 0
        slider.set_val(nxt)


def toggle_play(event):
    state['playing'] = not state['playing']
    btn_play.label.set_text('Pause' if state['playing'] else 'Play')


def step_fwd(event):
    state['playing'] = False
    btn_play.label.set_text('Play')
    slider.set_val(min(state['frame'] + 1, n_frames - 1))


def step_back(event):
    state['playing'] = False
    btn_play.label.set_text('Play')
    slider.set_val(max(state['frame'] - 1, 0))


slider.on_changed(on_slider)
btn_play.on_clicked(toggle_play)
btn_fwd.on_clicked(step_fwd)
btn_back.on_clicked(step_back)


# ===========================================================================
#  RUN
# ===========================================================================
if os.environ.get("SMOKE_TEST"):
    draw_frame(0)
    draw_frame(n_frames // 2)
    draw_frame(n_frames - 1)
    print("SMOKE TEST OK")
else:
    timer = fig.canvas.new_timer(interval=int(1000 / FPS))
    timer.add_callback(advance)
    timer.start()
    draw_frame(0)
    plt.show()