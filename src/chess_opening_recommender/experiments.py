import json
from os.path import basename
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chess import BLACK, WHITE
from clustering import (cluster_data, dicts_to_openings, extract_openings,
                        find_fitting_openings, find_recommended_openings,
                        get_bic_score, get_cluster_dim_limits,
                        get_cluster_game_counts, get_openings_in_clusters,
                        jefferson_share_method, json_to_dicts,
                        lowest_dists_in_cluster, normalise_values, plot_points,
                        plot_points_in_clusters, plot_recommendations,
                        read_opening_data)
from opening import AXIS_NAMES_SHORT, Opening
from process_database import process_database


def test_bic_score():
    """Create a series of BIC scores for varying minimum proportions and 
    opening lengths"""
    max_sample = 1_600_000

    database = "game_databases/lichess_db_2017-01.pgn"
    files = create_varying_opening_db(database, max_sample)

    with open("results/experiments_unweighted_b.txt", "w", encoding='utf-8') as out:
        for file in files:
            openings: List[Opening] = dicts_to_openings(json_to_dicts(file))
            games_recorded: int = 0

            for opening in openings:
                games_recorded += opening.results_total[0]

            cluster_factor = int(games_recorded / 1000)
            if cluster_factor == 0:
                cluster_factor = 1

            data = extract_openings(openings, BLACK)

            if len(openings) < 45:
                out.write(f"{basename(file)}: \n")
                out.write("Side: Black\n")
                out.write(f"Openings recorded = {len(openings)} \n")
                out.write("Could not cluster")

            best_score = get_bic_score(data)
            for _ in range(5):
                temp_best_score = get_bic_score(data)
                if temp_best_score["BIC score"] < best_score["BIC score"]:
                    best_score = temp_best_score

            out.write(f"{basename(file)}: \n")
            out.write("Side: Black\n")
            out.write(f"Openings recorded = {len(openings)} \n")
            out.write(json.dumps(best_score.to_dict()))


def create_varying_opening_db(database: str, max_sample: int) -> List[str]:
    """Takes in a database path and creates a series of opening databases
    by varying opening length and sampling rate"""
    max_sample = 1_600_000

    files = []

    for opening_length in range(3, 11):

        process_database(database,
                         "experiments",
                         2 / 100000,
                         opening_length,
                         max_sample
                         )

        with (open(f"experiments/lichess_db_ol{int(opening_length)}_p2.json",
                   "r", encoding='utf-8')) as in_file:
            data = json.load(in_file)

        for min_prop in range(4, 21, 2):
            temp_opening = []
            for opening in data["Openings"]:
                if opening["Total Results"][0] > (max_sample * min_prop/100000):
                    temp_opening.append(opening)

            data["Openings"] = temp_opening
            data["Opening Length"] = opening_length
            data["Sample Rate"] = min_prop

            with (open(f"experiments/lichess_db_ol{opening_length}_p{min_prop}.json",
                       "w", encoding='utf-8')) as out:
                out.write(json.dumps(data, indent=4))

    for opening_length in range(3, 11):
        for min_prop in range(2, 21, 2):
            files.append(f"experiments/lichess_db_ol{opening_length}_p{min_prop}.json")

    return files


def test_clustering():
    c_model = cluster_data("experiments/lichess_db_ol6_p6.json", WHITE)

    if c_model is None:
        print("!!! Couldnt cluster from file !!!")
        return

    plot_points_in_clusters(c_model, WHITE, "experiments/lichess_db_ol6_p6.json", "exp_plots", True)

    openings = dicts_to_openings(json_to_dicts("experiments/lichess_db_ol6_p6.json"))
    # graph_points(openings, WHITE, "exp_plots", "test_", True)
    openings_in_clusters = get_openings_in_clusters(c_model, openings, WHITE)

    for i, cluster in enumerate(openings_in_clusters):
        plot_points(cluster, WHITE, "exp_plots", f"test_c{i}_", True)


def plot_dist_funcs():
    return


def rand_points(n: int = 20) -> Tuple[Any, Any]:
    """Generate 30 random x, y values between 0-1"""
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    return (x, y)


def plot_each_cluster():
    c_model = cluster_data("experiments/lichess_db_ol6_p6.json", WHITE)

    if c_model is None:
        print("!!! Couldnt cluster from file !!!")
        return

    openings = dicts_to_openings(json_to_dicts("experiments/lichess_db_ol6_p6.json"))
    us_opening = dicts_to_openings(json_to_dicts("opening_databases/Kasparov_ol6_p1000.json"))
    # graph_points(openings, WHITE, "exp_plots", "test_", True)
    openings_in_clusters = get_openings_in_clusters(c_model, openings, WHITE)
    us_in_clusts = get_openings_in_clusters(c_model, us_opening, WHITE)

    dims = get_cluster_dim_limits(openings_in_clusters, WHITE)

    for i, (cluster, us_openings) in enumerate(zip(openings_in_clusters, us_in_clusts)):
        plot_recommendations(
            us_openings, cluster, WHITE, dims[i],
            0, 1, "exp_plots", f"plt_c{i}_", True)
    return


def display_pandas_clusters():
    c_model = cluster_data("opening_databases/lichess_db_ol8_p30.json", WHITE, 4)

    if c_model is None:
        print("!!! Couldnt cluster from file !!!")
        return

    openings = dicts_to_openings(json_to_dicts("opening_databases/lichess_db_ol8_p30.json"))
    list_vals: List[List[float]] = []

    dims = get_cluster_dim_limits([openings], WHITE)[0]

    for opening in openings:
        list_vals.append(read_opening_data(opening, WHITE))

    to_clust = np.array(list_vals)

    clusts = c_model.predict(to_clust)

    plot_data = []
    for i, val in enumerate(list_vals):
        parsed_val = normalise_values(val, dims)
        parsed_val.append(clusts[i])
        plot_data.append(parsed_val)

    df = pd.DataFrame(
        np.array(plot_data),
        columns=[*AXIS_NAMES_SHORT, "Cluster"],
    )

    df = df.sort_values(by="Cluster")

    plt.figure(figsize=[12, 8])
    pd.plotting.parallel_coordinates(df, "Cluster")
    plt.show()


if __name__ == "__main__":
    display_pandas_clusters()
