#Downloading libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from matplotlib.widgets import Slider, Button

# Lоading data
def load_data(data_path, verbose=True):
    
    with np.load(data_path) as data:

        if verbose:
            print("Available keys:", data.files)
            for key in data.files:
                arr = data[key]
                print(f"{key}: shape={arr.shape}")

        return (
            data["positions"],
            data["spikes"],
            data["voltages"],
            data["voltage_id"],
        )


#Function for creation of interactive plot
def interactive_plot(
    positions: np.ndarray,
    spikes: np.ndarray,
    voltages: np.ndarray,
    voltage_id: np.ndarray,
    window: float = 0.5,
    save_video: bool = False,
    output_path: str = "spike_animation.mp4",
    save_every_nth_frame: int = 2,
    verbose: bool = True,
):
    #Input validation 
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")

    if spikes.ndim != 2 or spikes.shape[1] != 2:
        raise ValueError("spikes must have shape (M, 2)")
    
    if voltages.ndim != 2:
        raise ValueError("voltages must be a 2D array")

    if voltage_id.ndim != 1:
        raise ValueError("voltage_id must be a 1D array")

    #Separating and sorting spike data
    spike_times = spikes[:, 0]
    spike_indices = spikes[:, 1].astype(int)

    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    spike_indices = spike_indices[order]

    N = positions.shape[0]
    if verbose:
        print(f"Number of neurons: {N}")
        print(f"Simulation time range: {spike_times.min():.2f} - {spike_times.max():.2f} ms")


    # Animation parameters

    t_start = 0.0
    t_end = spike_times.max()


    frames = np.arange(t_start, t_end, window)
    n_frames = len(frames)
    if verbose:
        print(f"Number of frames: {n_frames} (window = {window} ms)")
    
    # Active neurons precomputition 

    active = np.zeros((n_frames, N), dtype=bool)

    spike_ptr = 0
    n_spikes = len(spike_times)

    for f_idx, t in enumerate(frames):
        t_next = t + window
        while spike_ptr < n_spikes and spike_times[spike_ptr] < t_next:
            if spike_times[spike_ptr] >= t:
                active[f_idx, spike_indices[spike_ptr]] = True
            spike_ptr += 1


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
        spiking = active[frame_idx]
        colors[spiking & ~is_recorded] = RED       
        colors[spiking & is_recorded] = YELLOW     
        return colors


    # Creating figure 

    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(len(voltage_id), 2, width_ratios=[1.1, 1],
                        left=0.06, right=0.97, top=0.92, bottom=0.22,
                        wspace=0.25, hspace=0.4)

    # --- Left panel: spike raster (spans all rows) ---
    ax = fig.add_subplot(gs[:, 0])
    scat = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=frame_colors(0),
        s=30,
        edgecolors='k',
        linewidths=0.2
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal")
    ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
    ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
    title = ax.set_title(f"t = {frames[0]:.2f} ms")

    # --- Right panel: voltage traces, one subplot per recorded neuron ---
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
        vline = ax_v.axvline(frames[0], color=YELLOW, linewidth=1.2)
        volt_axes.append(ax_v)
        vlines.append(vline)
    volt_axes[-1].set_xlabel("Time (ms)")
    #fig.suptitle("Left: network raster   |   Right: voltage traces") #Title at the top of the graph


    def draw_frame(idx):
        idx = int(idx)
        t = frames[idx]
        scat.set_color(frame_colors(idx))
        title.set_text(f"t = {t:.2f} ms")
        for vline in vlines:
            vline.set_xdata([t, t])
        fig.canvas.draw_idle()


    #Interaction creation

    ax_slider = plt.axes([0.15, 0.10, 0.55, 0.03])
    slider = Slider(ax_slider, "Time", 0, n_frames - 1,
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
            next_idx = (slider.val + 1) % n_frames
            slider.set_val(next_idx)

    timer.add_callback(timer_callback)
    timer.start()

    def on_key(event):
        current_idx = int(slider.val)

        if event.key == "right":
            slider.set_val(min(current_idx + 1, n_frames - 1))
        elif event.key == "left":
            slider.set_val(max(current_idx - 1, 0))
        elif event.key == "up":
            slider.set_val(min(current_idx + 10, n_frames - 1))
        elif event.key == "down":
            slider.set_val(max(current_idx - 10, 0))
        elif event.key == " ":
            on_button(event)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_frame(0)
    
    #Saving animation
    if save_video:

        export_frame_indices = np.arange(0, n_frames, save_every_nth_frame)

        def update(idx):
            scat.set_color(frame_colors(idx))
            title.set_text(f"t = {frames[idx]:.2f} ms")
            return scat, title

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
    """Run the interactive visualization from the command line."""

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

    positions, spikes, voltages, voltage_id = load_data(
        args.data_file,
        verbose=not args.quiet,
    )

    interactive_plot(
        positions,
        spikes,
        voltages,
        voltage_id,
        window=args.window,
        save_video=args.save_video,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()