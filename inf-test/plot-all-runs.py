import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files={
    'Organized Drive, Local Connections' : [
        'SpatiallyOrganizedDrive-Lognormal91cpn-0.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-1.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-2.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-3.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-4.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-5.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-6.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-7.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-8.json',
        'SpatiallyOrganizedDrive-Lognormal91cpn-9.json'
    ],
    'Random Drive, Local Connections'    : [
        'SpatiallyRandomDrive-Lognormal91cpn-0.json',
        'SpatiallyRandomDrive-Lognormal91cpn-1.json',
        'SpatiallyRandomDrive-Lognormal91cpn-2.json',
        'SpatiallyRandomDrive-Lognormal91cpn-3.json',
        'SpatiallyRandomDrive-Lognormal91cpn-4.json',
        'SpatiallyRandomDrive-Lognormal91cpn-5.json',
        'SpatiallyRandomDrive-Lognormal91cpn-6.json',
        'SpatiallyRandomDrive-Lognormal91cpn-7.json',
        'SpatiallyRandomDrive-Lognormal91cpn-8.json',
        'SpatiallyRandomDrive-Lognormal91cpn-9.json'
    ],
    'Organized Drive, Random Connections': [
        'SpatiallyOrganizedDrive-Random91cpn-0.json',
        'SpatiallyOrganizedDrive-Random91cpn-1.json',
        'SpatiallyOrganizedDrive-Random91cpn-2.json',
        'SpatiallyOrganizedDrive-Random91cpn-3.json',
        'SpatiallyOrganizedDrive-Random91cpn-4.json',
        'SpatiallyOrganizedDrive-Random91cpn-5.json',
        'SpatiallyOrganizedDrive-Random91cpn-6.json',
        'SpatiallyOrganizedDrive-Random91cpn-7.json',
        'SpatiallyOrganizedDrive-Random91cpn-8.json',
        'SpatiallyOrganizedDrive-Random91cpn-9.json'
    ],
    'Random Drive, Random Connections'   : [
        'SpatiallyRandomDrive-Random91cpn-0.json',
        'SpatiallyRandomDrive-Random91cpn-1.json',
        'SpatiallyRandomDrive-Random91cpn-2.json',
        'SpatiallyRandomDrive-Random91cpn-3.json',
        'SpatiallyRandomDrive-Random91cpn-4.json',
        'SpatiallyRandomDrive-Random91cpn-5.json',
        'SpatiallyRandomDrive-Random91cpn-6.json',
        'SpatiallyRandomDrive-Random91cpn-7.json',
        'SpatiallyRandomDrive-Random91cpn-8.json',
        'SpatiallyRandomDrive-Random91cpn-9.json'
    ]
}



def plot_mutual_information(results_files):
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
            records[-1][        "mutual_information"] += [ data["mutual_information"] ]
            records[-1]["shuffle_mutual_information"] += data["shuffled_mutual_information_values"]
            granttotal                                += data["shuffled_mutual_information_values"]
        statistics.append(
            {
               "label": label,
                "mi_mean"        : np.mean(records[-1][        "mutual_information"]),
                "mi_std"         : np.std( records[-1][        "mutual_information"]),
                "shuffle_mean": np.mean(records[-1]["shuffle_mutual_information"]),
                "shuffle_std" : np.std( records[-1]["shuffle_mutual_information"]),
            }
        )
    statistics.append(
        {
            "label": 'Shuffle Grant Total',
            "mi_mean"        : None,
            "mi_std"         : None,
            "shuffle_mean": np.mean(granttotal),
            "shuffle_std" : np.std( granttotal),
        }
    )
    df = pd.DataFrame(statistics)

    x = np.arange(len(df))

    
    fig, ax = plt.subplots(figsize=(10, 5))


    # Mutual information markers
    l, = ax.plot(x,df["mi_mean"],'-o', ms=10, color='#D86023', label="Mean Mutual Information")

    ax.fill_between(
        x,
        df["mi_mean"] - df["mi_std"],
        df["mi_mean"] + df["mi_std"],
        alpha=0.25,
        label="Mutual Information Mean ± Std",
        facecolors = '#D86023'
    )


# Shuffle mean line
    l, = ax.plot(x,df["shuffle_mean"],"-o",label="Shuffle Mean")
    # Confidence band (mean ± std)
    ax.fill_between(
        x,
        df["shuffle_mean"] - df["shuffle_std"],
        df["shuffle_mean"] + df["shuffle_std"],
        alpha=0.25,
        label="Shuffle Mean ± Std",
        facecolors=l.get_color()
    )    

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=60, ha="right",fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Mutual Information ( bit / ISI )",fontsize=12)
    # ax.set_title("Mutual Information vs Shuffle Distribution")
    ax.legend(fontsize=16)
    ax.grid(True)

    plt.tight_layout()
    plt.show()

plot_mutual_information(files)
