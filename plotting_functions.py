#Downloading libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from matplotlib.widgets import Slider, Button

# Lоading data from .npz file
def load_data(data_path, verbose=True):

    with np.load(data_path) as data:

        if verbose:
            print("Available keys:", data.files)
            for key in data.files:
                arr = data[key]
                print(f"{key}: shape={arr.shape}")

        return (
            {key: data[key] for key in data.files}
        )


#Function for creation of interactive plot
def interactive_plot(
    data: dict, #input data
    window: float = 0.5, #time resolution of a single frame
    save_video: bool = False, #video saving mode
    output_path: str = "spike_animation.mp4", 
    save_every_nth_frame: int = 20, #a step between saved frames in a video
    verbose: bool = True, 
    window_step: float = 0.1, #how much the window size changes when you press the + or - keys (adjustment increment)
    window_min: float = 0.1, #min adjustment increment
    window_max: float = 20.0, #max adjustment increment
):
    
    #Input validation
    required = {
        "positions": {"ndim": 2, "ncols": 2},
        "spikes": {"ndim": 2, "ncols": 2},
        "voltages": {"ndim": 2},
        "voltage_id": {"ndim": 1},
    }
    
    for key, expected in required.items():

        if key not in data:
            raise KeyError(f"Missing required key '{key}'.")

        arr = data[key]

        if not isinstance(arr, np.ndarray):
            raise TypeError(f"'{key}' must be a NumPy array.")

        if arr.ndim != expected["ndim"]:
            raise ValueError(
                f"'{key}' must be {expected['ndim']}D, got {arr.ndim}D."
            )

        if "ncols" in expected and arr.shape[1] != expected["ncols"]:
            raise ValueError(
                f"'{key}' must have shape (N, {expected['ncols']}), got {arr.shape}."
            )
        
    #Input unpacking
    positions = data["positions"]
    spikes = data["spikes"]
    voltages = data["voltages"]
    voltage_id = data["voltage_id"]
          


    #Separating and sorting spike data
    spike_times = spikes[:, 0]
    spike_indices = spikes[:, 1].astype(int)

    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    spike_indices = spike_indices[order]

    N = positions.shape[0]
    marker_size = np.clip(1500 / np.sqrt(N), 2, 30) #marker size scales with network size
    
    if verbose:
        print(f"Number of neurons: {N}")
        print(f"Simulation time range: {spike_times.min():.2f} - {spike_times.max():.2f} ms")

    t_start = 0.0
    t_end = spike_times.max()

    # Function that (re)computes frames + active-neuron matrix for a given window size
    def compute_frames(window_val):
        frames = np.arange(t_start, t_end, window_val)
        n_frames = len(frames)

        active = np.zeros((n_frames, N), dtype=bool)
        spike_ptr = 0
        n_spikes = len(spike_times)

        for f_idx, t in enumerate(frames):
            t_next = t + window_val
            while spike_ptr < n_spikes and spike_times[spike_ptr] < t_next:
                if spike_times[spike_ptr] >= t:
                    active[f_idx, spike_indices[spike_ptr]] = True
                spike_ptr += 1

        return frames, n_frames, active

    # Mutable state holding the current window/frames/active — updated on key press
    state = {
    "window": window,
    "marker_size": marker_size,
    }

    state["frames"], state["n_frames"], state["active"] = compute_frames(window)
    

    if verbose:
        print(f"Number of frames: {state['n_frames']} (window = {state['window']} ms)")

    # COLOR SETUP: gray = resting, red = spiking
    GRAY = np.array([0.7, 0.7, 0.7, 1.0])         # resting, regular neuron
    RED = np.array([1.0, 0.0, 0.0, 1.0])          # spiking, regular neuron
    DARK_GRAY = np.array([0.25, 0.25, 0.25, 1.0]) # resting, recorded neuron
    YELLOW = np.array([1.0, 0.85, 0.0, 1.0])      # spiking, recorded neuron

    is_recorded = np.zeros(N, dtype=bool)
    is_recorded[voltage_id.astype(int)] = True

    def frame_colors(frame_idx):
        colors = np.tile(GRAY, (N, 1))
        colors[is_recorded] = DARK_GRAY
        spiking = state["active"][frame_idx]
        colors[spiking & ~is_recorded] = RED
        colors[spiking & is_recorded] = YELLOW
        return colors

    # Creating figure
    
    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(len(voltage_id), 2, width_ratios=[1.1, 1],
                        left=0.06, right=0.97, top=0.92, bottom=0.22,
                        wspace=0.25, hspace=0.4)

    # Left panel: spike change on neuron coords
    ax = fig.add_subplot(gs[:, 0])
    scat = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=frame_colors(0),
        s=state["marker_size"],
        edgecolors='k',
        linewidths=0.2
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal")
    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
    title = ax.set_title(f"t = {state['frames'][0]:.2f} ms   (window = {state['window']:.2f} ms)")

    # Right panel: voltage traces, one subplot per recorded neuron 
    time_v = voltages[:, 0]  # ms
    volt_axes = []
    vlines = []
    for i, neuron_id in enumerate(voltage_id):
        ax_v = fig.add_subplot(gs[i, 1])
        v = voltages[:, i + 1]
        ax_v.plot(time_v, v, color='k', linewidth=0.8)
        ax_v.set_ylabel("V (mV)")
        ax_v.set_title(f"Neuron #{neuron_id}", fontsize=10, loc='right')
        ax_v.grid(alpha=0.3)
        vline = ax_v.axvline(state["frames"][0], color=YELLOW, linewidth=1.2)
        volt_axes.append(ax_v)
        vlines.append(vline)
    volt_axes[-1].set_xlabel("Time (ms)")

    def draw_frame(idx):
        idx = int(idx)
        t = state["frames"][idx]
        scat.set_color(frame_colors(idx))
        title.set_text(f"t = {t:.2f} ms   (window = {state['window']:.2f} ms)")
        for vline in vlines:
            vline.set_xdata([t, t])
        fig.canvas.draw_idle()

    #Interaction creation
    ax_slider = plt.axes([0.15, 0.10, 0.55, 0.03])
    slider = Slider(ax_slider, "Time", 0, state["n_frames"] - 1,
                    valinit=0, valstep=1)

    ax_button = plt.axes([0.80, 0.09, 0.10, 0.05])
    button = Button(ax_button, "Play")
    is_playing = [False]

    def on_slider(val):
        draw_frame(slider.val)

    slider.on_changed(on_slider)

    def on_button(event):
        is_playing[0] = not is_playing[0]
        button.label.set_text("Pause" if is_playing[0] else "Play")

    button.on_clicked(on_button)

    timer = fig.canvas.new_timer(interval=50)

    def timer_callback():
        if is_playing[0]:
            next_idx = (slider.val + 1) % state["n_frames"]
            slider.set_val(next_idx)

    timer.add_callback(timer_callback)
    timer.start()

    def change_marker_size(factor):
        new_size = np.clip(state["marker_size"] * factor, 1, 100)
        state["marker_size"] = new_size
        scat.set_sizes(np.full(N, new_size))
        fig.canvas.draw_idle()

    # Changing the timestep 
    def change_window(new_window):
        new_window = min(max(new_window, window_min), window_max)
        if new_window == state["window"]:
            return  # already at the limit, nothing to do

        # remember current time (ms), not just frame index, to stay at roughly the same moment
        current_time = state["frames"][int(slider.val)]

        state["window"] = new_window
        state["frames"], state["n_frames"], state["active"] = compute_frames(new_window)

        # find the frame closest to the previous current_time
        new_idx = int(np.argmin(np.abs(state["frames"] - current_time)))

        # update slider range to match the new number of frames
        slider.valmax = state["n_frames"] - 1
        slider.ax.set_xlim(slider.valmin, slider.valmax)

        slider.set_val(new_idx)  # triggers on_slider -> draw_frame
        if verbose:
            print(f"Window changed to {new_window:.2f} ms ({state['n_frames']} frames)")

    #Key function creation
    def on_key(event):
        current_idx = int(slider.val)

        if event.key == "right":
            slider.set_val(min(current_idx + 1, state["n_frames"] - 1))
        elif event.key == "left":
            slider.set_val(max(current_idx - 1, 0))
        elif event.key == "up":
            slider.set_val(min(current_idx + 10, state["n_frames"] - 1))
        elif event.key == "down":
            slider.set_val(max(current_idx - 10, 0))
        elif event.key == " ":
            on_button(event)
        elif event.key in ("shift++", "shift+=", "+", "="):
            change_window(state["window"] + window_step)
        elif event.key in ("shift+-", "shift+_", "-", "_"):
            change_window(state["window"] - window_step)
        elif event.key == ">":
            change_marker_size(1.2)
        elif event.key == "<":
            change_marker_size(1/1.2)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_frame(0)

    #Saving animation
    if save_video:

        export_frame_indices = np.arange(0, state["n_frames"], save_every_nth_frame)

        def update(idx):
            draw_frame(idx)
            return scat, title, *vlines

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=export_frame_indices,
            interval=50,
            blit=False
        )

        if verbose:
            print(f"Saving video to {output_path} ...")
        ani.save(output_path, writer="ffmpeg", fps=20, dpi=150)
        if verbose:
            print("Done.")
    plt.show()
    return fig


#Main function
def main():
    
    parser = argparse.ArgumentParser(
        description="Visualize spiking neural network simulations."
    )

    parser.add_argument(
        "data_file",
        help="Path to the .npz simulation file.",
    )

    parser.add_argument(
        "--window",
        type=float,
        default=0.5,
        help="Animation window size in milliseconds.",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the animation as an MP4 file.",
    )

    parser.add_argument(
        "--output",
        default="spike_animation.mp4",
        help="Output filename for the saved animation.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational messages.",
    )

    args = parser.parse_args()

    data = load_data(
    args.data_file,
    verbose=not args.quiet,
    )

    interactive_plot(
        data,
        window=args.window,
        save_video=args.save_video,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()