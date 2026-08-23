import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict


def plot_mutual_information(results_files, save:(str,None)=None, logscale:bool=False):
    records = []
    statistics = []
    granttotal = []
    for label, json_files in results_files.items():

        records.append(
            {
                "label": label,
                "mutual_information": [],
                "shuffle_mutual_information": [],
            }
        )
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
            if not data["mutual_information"]: continue
            records[-1][        "mutual_information"] += [ data["mutual_information"] ]
            records[-1]["shuffle_mutual_information"] += data["shuffled_mutual_information_values"]
            granttotal                                += data["shuffled_mutual_information_values"]
        statistics.append(
            {
               "label": str(label),
                "mi_mean"     : np.mean(records[-1][        "mutual_information"]),
                "mi_std"      : np.std( records[-1][        "mutual_information"]),
                "shuffle_mean": np.mean(records[-1]["shuffle_mutual_information"]),
                "shuffle_std" : np.std( records[-1]["shuffle_mutual_information"]),
            }
        )
    statistics.append(
        {
            "label": 'Shuffle Grant Total',
            "mi_mean"     : None,
            "mi_std"      : None,
            "shuffle_mean": np.mean(granttotal),
            "shuffle_std" : np.std( granttotal),
        }
    )
    df = pd.DataFrame(statistics)

    x = np.arange(len(df))

    
    fig, ax = plt.subplots(figsize=(10, 5))


    # Mutual information markers
    # l, = ax.plot(x,df["mi_mean"],'o', ms=10, color='#D86023', label="Mean Mutual Information")
    ax.errorbar(df["mi_mean"],x,fmt='o',xerr=df["mi_std"], ms=8, color='#D86023',elinewidth=3, label="Mutual Information (Mean ± STD)")
    # ax.fill_between(
        # x,
        # df["mi_mean"] - df["mi_std"],
        # df["mi_mean"] + df["mi_std"],
        # alpha=0.25,
        # label="Mutual Information Mean ± Std",
        # facecolors = l.get_color()
    # )


    # l = ax.errorbar(x,df["mi_mean"],fmt='-o', yerr=df["mi_std"], ms=10, color='#D86023', label="Mean Mutual Information")

# Shuffle mean line
    # l, = ax.plot(x,df["shuffle_mean"],"o",label="Shuffle Mean")
    ax.errorbar(df["shuffle_mean"],x,fmt="o", ms=8,xerr=df["shuffle_std"],elinewidth=3 ,label="Shuffled drive vector (Mean ± STD)")
    #Confidence band (mean ± std)
    # ax.fill_between(
        # x,
        # df["shuffle_mean"] - df["shuffle_std"],
        # df["shuffle_mean"] + df["shuffle_std"],
        # alpha=0.25,
        # label="Shuffle Mean ± Std",
        # facecolors=l.get_color()
    # )    

    # l = ax.errorbar(x,df["shuffle_mean"],fmt="x",yerr=df["shuffle_std"],label="Shuffle Mean")

    if logscale:
        ax.set_xscale('log')
    ax.set_yticks(x)
    ax.set_yticklabels(df["label"], rotation=0, ha="right",fontsize=20)
    ax.set_ylabel("")
    ax.set_xlabel("Mutual Information ( bit / ISI )",fontsize=20)
    # ax.set_title("Mutual Information vs Shuffle Distribution")
    ax.legend(fontsize=24)
    ax.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()


def parse_pairs(items):
    result = defaultdict(list)

    for item in items:
        try:
            name, value = item.split("=", 1)
            result[name].append(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid argument format: '{item}'. Expected NAME=VALUE"
            )

    return dict(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input JSON file with dictionary of all files to process"
    )
    parser.add_argument(
        "--save-figure",
        type=str,
        default=None,
        help="Save figure into a file (default None)"
    )
    parser.add_argument(
        "--log-scale",
        action=argparse.BooleanOptionalAction,
        help="Set x log-scaled"
    )
    parser.add_argument(
        "pairs",
        nargs="*",
        help="Arguments in the form NAME=VALUE",
    )

    args = parser.parse_args()
    if args.input:
        with open(args.input) as fd:
            files = json.load(fd)
    else:
        files = parse_pairs(args.pairs)

    if len(files) == 0:
        raise RuntimeError("Nothing to plot!")
    plot_mutual_information(files, save=args.save_figure,logscale=args.log_scale)
