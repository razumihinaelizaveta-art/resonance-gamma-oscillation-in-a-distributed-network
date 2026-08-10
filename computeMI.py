import numpy as np
from ButtsMI import ButtsMI
from ButtsMI.dist_comput import _ftnorms
import argparse
import json


def parse_arguments():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Compute mutual information from a simulation."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input NPZ file."
    )

    parser.add_argument(
        "--feature-array",
        type=str,
        default="positions",
        help="An array used for features (default is positions)"
    )

    parser.add_argument(
        "--remove-first",
        type=float,
        default=0.0,
        help="Remove the first N milliseconds from the analysis."
    )

    parser.add_argument(
        "--remove-last",
        type=float,
        default=0.0,
        help="Remove the last N milliseconds from the analysis."
    )

    parser.add_argument(
        "--keep-neurons",
        type=int,
        default=None,
        help="Keep only the first N neurons for the analysis."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to the output JSON file."
    )

    parser.add_argument(
        "--shuffle-iterations",
        type=int,
        default=0,
        help="Number of shuffled control iterations."
    )

    return parser.parse_args()


# load the data
def load_data(filename,feature_array):
    """Load neuron positions and spike events from an NPZ file."""

    data = np.load(filename, allow_pickle=True)

    features   = data[feature_array]#.astype(float)
    spikes_raw = data["spikes"]

    
    return features, spikes_raw

# mixes the feature between the neurons. The feature stay the same, only which neuron sits where changes.
def shuffle_features(old_feature):
    """Randomly shuffle neuron feature while preserving their distribution."""

    new_feature = old_feature.copy()
    np.random.shuffle(new_feature)
    return new_feature

def preprocess_data(feature, spikes_raw, args, spike_part,):


    n_spikes_total = len(spikes_raw)
    
    recording_end = float(spikes_raw[:, 0].max())

    time_lo = args.remove_first
    time_hi = recording_end - args.remove_last

    if time_hi <= time_lo:
        raise ValueError(
        f"Invalid analysis window: start={time_lo} ms, end={time_hi} ms"
        )
    
    # cut 1: time window
    spikes_in_window = []
    for row in spikes_raw:
        time_of_spike = row[0]
        if time_of_spike < time_lo:
            continue
        if time_of_spike >= time_hi:
            continue
        spikes_in_window.append(row)
    spikes_raw = spikes_in_window

    print("spikes inside the window:", len(spikes_raw),
        "(", round(100.0 * len(spikes_raw) / n_spikes_total, 1), "% of the file )")

    # cut 2: spikes
    keep_spikes = len(spikes_raw) // spike_part
    spikes_raw = spikes_raw[:keep_spikes]

    n_neurons = feature.shape[0]
    print("number of neurons:", n_neurons)
    print("spikes kept:", len(spikes_raw))

    # sort the spikes - one list of firing times per neuron
    spikes = [[] for _ in range(n_neurons)]

    for row in spikes_raw:
        time_of_spike = row[0]
        which_neuron = int(row[1])
        spikes[which_neuron].append(time_of_spike)

    for i in range(n_neurons):
        spikes[i] = np.array(sorted(spikes[i]))

    # cut 3: neurons
    if args.keep_neurons is None:
        keep_neurons = n_neurons
    else:
        keep_neurons = min(args.keep_neurons, n_neurons)
    feature = feature[:keep_neurons]
    spikes = spikes[:keep_neurons]
    print(f"neurons used: {keep_neurons}")

    # how much usable data is left - a neuron needs at least 2 spikes
    # to make an interval, so the other ones add nothing
    silent = one_spike = usable = 0

    for neuron_spikes in spikes:
        if len(neuron_spikes) == 0:
            silent += 1
        elif len(neuron_spikes) == 1:
            one_spike += 1
        else:
            usable += 1

    spikes_used = len(spikes_raw)

    return (feature, spikes, keep_neurons, spikes_used, silent, one_spike, usable, n_spikes_total)

def compute_mutual_information(feature, spikes):
    """Compute mutual information from neuron feature and spike trains."""

    print("Computing mutual information...")

    bmi = ButtsMI(reduceBOS=False, isimax=100)

    # Distance between every pair of neurons
    dist = _ftnorms(feature, bmi.ftnormord)
    print("pairs:", dist.shape[0])

    # Build the distance histogram
    dh, db, nbins = bmi._nvar(
        dist,
        dist.min(),
        dist.max(),
        bmi.d_inbin_min,
        bmi.d_inbin_max,
        "distances",
        nbininit=20,
    )

    if dh is None:
        print("Could not build the distance histogram.")
        return None

    # Normalize histogram
    dh = dh / dh.sum()
    print("number of distance bins:", nbins)

    # Compute mutual information
    ret = bmi._computeMI(feature, dh, db, spikes)

    if ret is None:
        print("Mutual Information could not be computed.")
        return None

    isi, timing, bins, mi = ret

    print(f"Mutual Information = {mi} bits")

    return {
        "bmi": bmi,
        "distance_histogram": dh,
        "distance_bins": db,
        "num_bins": nbins,
        "mutual_information": float(mi),
    }


def run_shuffle_controls(bmi, feature, spikes, dh, db, iterations):
    shuffled_mi = []

    print(f"Running {iterations} shuffled controls...")

    for i in range(iterations):
        shuffled_feature = shuffle_features(feature)

        ret = bmi._computeMI(
            shuffled_feature,
            dh,
            db,
            spikes,
        )

        if ret is not None:
            _, _, _, mi = ret
            shuffled_mi.append(float(mi))

        print(f"  shuffle {i + 1}/{iterations}", end="\r")

    print()

    return shuffled_mi


def main():

    args = parse_arguments()

    feature, spikes_raw = load_data(args.input, args.feature_array)

    (
        feature,
        spikes,
        keep_neurons,
        spikes_used,
        silent,
        one_spike,
        usable,
        n_spikes_total,
    ) = preprocess_data(
        feature,
        spikes_raw,
        args,
        spike_part=1,
    )

    mi_results = compute_mutual_information(
        feature,
        spikes,
    )

    shuffled_mi = []

    if mi_results is not None and args.shuffle_iterations > 0:
        shuffled_mi = run_shuffle_controls(
            mi_results["bmi"],
            feature,
            spikes,
            mi_results["distance_histogram"],
            mi_results["distance_bins"],
            args.shuffle_iterations,
        )

    results = {
    "input_file": args.input,
    "feature_array": args.feature_array,
    "remove_first_ms": args.remove_first,
    "remove_last_ms": args.remove_last,
    "neurons_requested": args.keep_neurons,
    "neurons_used": keep_neurons,
    "spikes_total": n_spikes_total,
    "spikes_used": spikes_used,
    "silent_neurons": silent,
    "single_spike_neurons": one_spike,
    "usable_neurons": usable,
    "mutual_information": mi_results["mutual_information"] if mi_results is not None else None,
    "shuffled_mutual_information": shuffled_mi if shuffled_mi else None,
    "shuffle_iterations": args.shuffle_iterations,
    "shuffle_mean": float(np.mean(shuffled_mi)) if shuffled_mi else None,
    "shuffle_std": float(np.std(shuffled_mi)) if shuffled_mi else None,
    "shuffle_values": shuffled_mi,
    "status": "success" if mi_results is not None else "failed",
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

