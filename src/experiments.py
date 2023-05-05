import glob
import json
import subprocess
import sys
from os.path import basename
from typing import Any, List, Tuple

import matplotlib.markers as mk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chess import BLACK, WHITE
from sklearn.mixture import GaussianMixture

try:
    from opening import AXIS_NAMES_SHORT, Opening
    from pychora_db import process_database
    from pychora_rec import (
        check_opening_in_list,
        cluster_data,
        dicts_to_openings,
        extract_openings,
        find_fitting_openings,
        get_bic_score,
        get_centre_of_mass,
        get_cluster_dim_limits,
        get_cluster_game_counts,
        get_geometric_median,
        get_openings_in_clusters,
        get_valid_openings,
        jefferson_share_method,
        json_to_dicts,
        normalise_values,
        plot_points,
        plot_points_in_clusters,
        plot_recommendations,
        process_clustering,
        read_opening_data,
    )
except ImportError:
    from src.opening import AXIS_NAMES_SHORT, Opening
    from src.pychora_db import process_database
    from src.pychora_rec import (
        check_opening_in_list,
        cluster_data,
        dicts_to_openings,
        extract_openings,
        find_fitting_openings,
        get_bic_score,
        get_centre_of_mass,
        get_cluster_dim_limits,
        get_cluster_game_counts,
        get_geometric_median,
        get_openings_in_clusters,
        get_valid_openings,
        jefferson_share_method,
        json_to_dicts,
        normalise_values,
        plot_points,
        plot_points_in_clusters,
        plot_recommendations,
        process_clustering,
        read_opening_data,
    )


def test_bic_score():
    """Create a series of BIC scores for varying minimum proportions and
    opening lengths"""
    max_sample = 1_600_000

    database = "game_databases/lichess_db_2017-01.pgn"
    files = create_varying_opening_db(database, max_sample)

    with open("results/experiments_unweighted_b.txt", "w", encoding="utf-8") as out:
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
        process_database(
            database, "experiments", 2 / 100000, opening_length, max_sample
        )

        with open(
            f"experiments/lichess_db_ol{int(opening_length)}_p2.json",
            "r",
            encoding="utf-8",
        ) as in_file:
            data = json.load(in_file)

        for min_prop in range(4, 21, 2):
            temp_opening = []
            for opening in data["Openings"]:
                if opening["Total Results"][0] > (max_sample * min_prop / 100000):
                    temp_opening.append(opening)

            data["Openings"] = temp_opening
            data["Opening Length"] = opening_length
            data["Sample Rate"] = min_prop

            with open(
                f"experiments/lichess_db_ol{opening_length}_p{min_prop}.json",
                "w",
                encoding="utf-8",
            ) as out:
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

    plot_points_in_clusters(
        c_model, WHITE, "experiments/lichess_db_ol6_p6.json", "exp_plots", True
    )

    openings = dicts_to_openings(json_to_dicts("experiments/lichess_db_ol6_p6.json"))
    # graph_points(openings, WHITE, "exp_plots", "test_", True)
    openings_in_clusters = get_openings_in_clusters(c_model, openings, WHITE)

    for i, cluster in enumerate(openings_in_clusters):
        plot_points(cluster, WHITE, "exp_plots", f"test_c{i}_", True)


def plot_dist_funcs():
    samp_clust_data = rand_points()
    samp_user_data = rand_points(25, 60)

    com = get_centre_of_mass(samp_user_data, weight=True)

    exam_val = [
        list(a)
        for a in zip(
            [x_val for x_val in samp_user_data[0]],
            [y_val for y_val in samp_user_data[1]],
        )
    ]
    gmed = get_geometric_median(exam_val, list(samp_user_data[-1]))

    plot_data_and_point(
        "The Geometric Median of a Uniformly Random 2D Dataset",
        "Random X",
        "Random Y",
        samp_user_data,
        [com, gmed],
        ["Centre of Mass", "Geometric Median"],
    )
    return


def plot_data_and_point(
    title, x_lab, y_lab, data: List[np.ndarray], points: List[np.ndarray], meanings
):
    fig = plt.figure()
    ax = plt.subplot(111)

    scatter = ax.scatter(data[0], data[1], s=data[2])

    plotted_points = []
    for point in points:
        plotted_points.append(ax.scatter(*point, marker=mk.MarkerStyle("x"), s=[50]))

    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    leg = ax.legend(
        [scatter, *plotted_points],
        ["Sample Data", *meanings],
        loc="center left",
        bbox_to_anchor=(1, 0.8),
        shadow=True,
    )

    ax.add_artist(leg)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.33), shadow=True)
    plt.show()


def rand_points(n: int = 20, w_max: int = 10) -> List[np.ndarray]:
    """Generate 30 random x, y values between 0-1"""
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    if w_max <= 10:
        w = np.array([10 for _ in range(n)])
    else:
        w = np.random.randint(low=10, high=w_max, size=n)
    return [x, y, w]


def plot_each_cluster():
    c_model = cluster_data("experiments/lichess_db_ol6_p6.json", WHITE)

    if c_model is None:
        print("!!! Couldnt cluster from file !!!")
        return

    openings = dicts_to_openings(json_to_dicts("experiments/lichess_db_ol6_p6.json"))
    us_opening = dicts_to_openings(
        json_to_dicts("opening_databases/Kasparov_ol6_p1000.json")
    )
    # graph_points(openings, WHITE, "exp_plots", "test_", True)
    openings_in_clusters = get_openings_in_clusters(c_model, openings, WHITE)
    us_in_clusts = get_openings_in_clusters(c_model, us_opening, WHITE)

    dims = get_cluster_dim_limits(openings_in_clusters, WHITE)

    for i, (cluster, us_openings) in enumerate(zip(openings_in_clusters, us_in_clusts)):
        plot_recommendations(
            us_openings, cluster, WHITE, dims[i], 0, 1, "exp_plots", f"plt_c{i}_", True
        )
    return


def display_pandas_clusters():
    c_model = cluster_data("opening_databases/lichess_db_ol8_p30.json", WHITE, 4)

    if c_model is None:
        print("!!! Couldnt cluster from file !!!")
        return

    openings = dicts_to_openings(
        json_to_dicts("opening_databases/lichess_db_ol8_p30.json")
    )
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


def graph_values_of_openings(file_name: str, color: bool):
    """Graph each of the frequency of the values within an opening database file
    for each of the quantitative analysis of each position"""
    with open(file_name, "r") as f:
        file_decoded = json.load(f)

    value_names = file_decoded["Value Names"]

    values = []
    weights = []

    openings = dicts_to_openings(file_decoded["Openings"])

    for opening in openings:
        values.append(read_opening_data(opening, color))
        weights.append(opening.results_total[0])

    grouped_vals = zip(*values)

    for vals, name in zip(grouped_vals, value_names):
        plot_value_frequency(vals, name, weights, steps=30)


def plot_value_frequency(
    values: Tuple[float], values_name: str, weights: List[int] = [], steps: int = 15
):
    limits = [min(values), max(values) + 0.1]
    step_dif = (limits[1] - limits[0]) / steps

    x = [0 for _ in range(steps)]
    y = [(limits[0] + 0.5 * step_dif) + n * step_dif for n in range(steps)]

    if not weights:
        weights = [1 for _ in values]
        plt.ylabel("Number of openings")
    else:
        plt.ylabel("Number of games played")
    for value, weight in zip(values, weights):
        print(int((value - limits[0]) // step_dif))
        x[int((value - limits[0]) // step_dif)] += weight

    plt.plot(y, x)

    plt.xlabel(values_name)
    plt.title("Frequency of Values for an Opening Evaluation")
    plt.show()


def test_program(clust_file, input_dir, ouput_dir, prefix="GM"):
    """Do a complete test of the results of the program from start to finish"""
    proc_files = process_all_pgns(input_dir, ouput_dir, length=5, prefix=prefix)
    # proc_files = glob.glob(f"{user_output_dir}/{prefix}*")

    viable_results_quant = get_viable_opening_count(proc_files, clust_file, WHITE)
    random_expected = random_opening_expectation(viable_results_quant, 15)
    most_common_expected = [
        most_used_expectation(file, clust_file, WHITE) for file in proc_files
    ]

    trials = 20

    geo_res = [0.0 for _ in proc_files]
    common_res = [0.0 for _ in proc_files]
    rand_res = [0.0 for _ in proc_files]
    wrand_res = [0.0 for _ in proc_files]

    geo_div = []
    common_div = []
    wrand_div = []
    rand_div = []

    for t in range(trials):
        print(f"Trial Number : {t}")
        clust_model = cluster_data(clust_file, WHITE)
        if clust_model is None:
            continue

        temp_geo, temp_geo_div = test_cluster_methods(
            clust_model, clust_file, proc_files, "geo"
        )
        temp_com, temp_common_div = test_cluster_methods(
            clust_model, clust_file, proc_files, "common"
        )
        temp_wrand, temp_wrand_div = test_cluster_methods(
            clust_model, clust_file, proc_files, "wrand"
        )
        temp_rand, temp_rand_div = rand_in_cluster(
            clust_model, clust_file, proc_files, "rand"
        )

        geo_div.append(temp_geo_div)
        common_div.append(temp_common_div)
        wrand_div.append(temp_wrand_div)
        rand_div.append(temp_rand_div)

        for i, _ in enumerate(proc_files):
            geo_res[i] += temp_geo[i]
            common_res[i] += temp_com[i]
            wrand_res[i] += temp_wrand[i]
            rand_res[i] += temp_rand[i]

    geo_res = [round(100 * res / trials / 15, 2) for res in geo_res]
    common_res = [round(100 * res / trials / 15, 2) for res in common_res]
    wrand_res = [round(100 * res / trials / 15, 2) for res in wrand_res]
    rand_res = [round(100 * res / trials / 15, 2) for res in rand_res]

    geo_div = [round(d / len(proc_files) * 100, 1) for d in geo_div]
    common_div = [round(d / len(proc_files) * 100, 1) for d in common_div]
    wrand_div = [round(d / len(proc_files) * 100, 1) for d in wrand_div]
    rand_div = [round(d / len(proc_files) * 100, 1) for d in rand_div]

    print(f"Repertoire lengths: {find_repertoire_lengths(proc_files)}\n")

    print(f"Cluster Geometric Median Diversity: {geo_div}")
    print(f"Cluster Most Common Diversity: {common_div}")
    print(f"Cluster Uniform Random Diversity: {rand_div}")
    print(f"Cluster Weighted Random Diversity: {wrand_div}\n")

    print(f"Random Opening: {random_expected}")
    print(f"Most Common Opening: {most_common_expected}")
    print(f"Cluster Geometric Median: {geo_res}")
    print(f"Cluster Most Common Opening: {common_res}")
    print(f"Cluster Uniform Random Opening: {rand_res}")
    print(f"Cluster Weighted Random Opening: {wrand_res}")


def process_all_pgns(pgn_dir: str, dest: str, prefix: str = "", length=5) -> List[str]:
    """Do a complete test of the results of the program from start to finish"""
    pgn_files = glob.glob(f"{pgn_dir}/{prefix}*")
    print(pgn_files)
    processed_files = []
    for file in pgn_files:
        processed_files.append(
            process_database(file, dest, opening_length=length, min_proportion=0.0002)
        )
    return processed_files


def get_viable_opening_count(user_files: List[str], clu_file, color) -> List[List[int]]:
    viable_openings: List[List[int]] = [[0, 0, 0] for _ in user_files]
    with open(clu_file, "r") as clu_f:
        stored_openings = dicts_to_openings(json.load(clu_f)["Openings"])
    for i, file in enumerate(user_files):
        print(file)
        with open(file, "r") as use_f:
            user_openings = dicts_to_openings(json.load(use_f)["Openings"])
        viable_openings[i][0] = len(
            get_valid_openings(
                stored_openings, user_openings[: len(user_openings) // 2], color
            )
        )
        viable_openings[i][1] = len(
            get_valid_openings(stored_openings, user_openings, color)
        )
        viable_openings[i][2] = viable_openings[i][0] - viable_openings[i][1]

    return viable_openings


def most_used_expectation(op_file: str, db_file: str, color) -> int:
    print(op_file)
    exp = 0
    with open(op_file, "r") as op:
        user_openings = dicts_to_openings(json.load(op)["Openings"])
    with open(db_file, "r") as db:
        db_openings = dicts_to_openings(json.load(db)["Openings"])

    valid_openings = get_valid_openings(db_openings, user_openings, color)

    rec_openings = get_most_used_openings(valid_openings)

    for opening in rec_openings:
        if check_opening_in_list(opening, user_openings):
            exp += 1

    return exp


def rand_in_cluster(
    clust_model: GaussianMixture,
    cluster_file: str,
    proc_files: List[str],
    color,
    rec: int = 15,
) -> Tuple[List[float], float]:
    """Generate the average number of openings that are randomly generated from designated clusters
    would be present in the users opening repertoire"""
    print("--- Finding Recommended Openings ---")

    large_openings = dicts_to_openings(json_to_dicts(cluster_file))
    if not large_openings:
        return ([], 0)

    results = [0.0 for _ in proc_files]

    diversity = 0.0

    temp_suggested_openings: List[Opening] = []

    for i, file in enumerate(proc_files):
        user_openings = dicts_to_openings(json_to_dicts(file))
        if not user_openings:
            return ([], 0)

        short_u_op = user_openings[: len(user_openings) // 2]

        # Put all the openings into their clusters
        print("--- Putting Openings in Clusters ---")
        user_openings_in_clusters = get_openings_in_clusters(
            clust_model, user_openings, color
        )
        sht_U_op_clust = get_openings_in_clusters(clust_model, short_u_op, color)
        all_openings_in_clusters = get_openings_in_clusters(
            clust_model, large_openings, color
        )

        # Count each cluster for the amount of GAMES within them, NOT UNIQUE OPENINGS
        cluster_counts = get_cluster_game_counts(user_openings_in_clusters)

        # Decide how to share the recommendations between the openings based on one of two methods
        print("--- Apportioning Recommendations for Clusters ---")
        rec_openins_per_cluster = jefferson_share_method(cluster_counts, rec)

        for all_op, sh_u_op, user_op, rec_per_clust in zip(
            all_openings_in_clusters,
            sht_U_op_clust,
            user_openings_in_clusters,
            rec_openins_per_cluster,
        ):
            incorrect_openings = get_valid_openings(all_op, user_op, color)
            choosable_openins = get_valid_openings(all_op, sh_u_op, color)

            clust_vop = [
                len(choosable_openins),
                len(incorrect_openings),
                len(choosable_openins) - len(incorrect_openings),
            ]
            results[i] += random_opening_expectation([clust_vop], rec_per_clust)[0]

            if len(choosable_openins) <= rec_per_clust:
                for opening in choosable_openins:
                    if not check_opening_in_list(opening, temp_suggested_openings):
                        temp_suggested_openings.append(opening)

            for opening in np.random.choice(
                choosable_openins, size=rec_per_clust  # type:ignore
            ):
                if not check_opening_in_list(opening, temp_suggested_openings):
                    temp_suggested_openings.append(opening)

    diversity = 15 * len(proc_files) / len(temp_suggested_openings)
    return [round(er, 2) for er in results], diversity


def test_cluster_methods(
    clust_model: GaussianMixture, cluster_file: str, proc_files: List[str], method="geo"
) -> Tuple[List[float], float]:
    results = [0.0 for _ in proc_files]
    rec_openings: List[List[Opening]] = []
    for user_file in proc_files:
        rec_openings.append(
            find_fitting_openings(
                clust_model, user_file, cluster_file, WHITE, test=True, method=method
            )
        )

    all_ops: List[Opening] = []
    for openings in rec_openings:
        for opening in openings:
            if not check_opening_in_list(opening, all_ops):
                all_ops.append(opening)

    diversity = 15 * len(proc_files) / len(all_ops)
    for i, res in enumerate(test_results(proc_files, rec_openings)):
        results[i] += res

    return results, diversity


def find_repertoire_lengths(user_files: List[str]) -> List[int]:
    """Return the individual lengths of a list of opening repertoires"""
    lengths = [0 for _ in user_files]

    for index, user_file in enumerate(user_files):
        with open(user_file, "r") as file:
            lengths[index] = len(json.load(file)["Openings"])

    return lengths


def get_most_used_openings(openings: List[Opening]) -> List[Opening]:
    output_openings: List[Opening] = []
    for opening in openings:
        if len(output_openings) < 15:
            output_openings.append(opening)
            continue

        replace_index = -1
        min_used = opening.results_total[0]

        for i, stored_opening in enumerate(output_openings):
            if stored_opening.results_total[0] < min_used:
                replace_index = i
                min_used = stored_opening.results_total[0]

        if replace_index != -1:
            output_openings[replace_index] = opening

    return output_openings


def random_opening_expectation(
    viable_openings: List[List[int]], rec: int
) -> List[float]:
    expected_results: List[float] = [0.0 for _ in viable_openings]
    for pos, pair in enumerate(viable_openings):
        for quant in range(1, rec):
            try:
                chose_non_appearing_openings = [
                    (pair[1] - i) / (pair[0] - i) for i in range(rec - quant)
                ]
                chose_appearing_openings = [
                    (pair[2] - i) / (pair[0] - i) for i in range(quant)
                ]

                expected_results[pos] += (
                    quant
                    * np.prod(chose_non_appearing_openings)
                    * np.prod(chose_appearing_openings)
                )
            except ZeroDivisionError:
                break

    return [round(er, 2) for er in expected_results]


def create_test_recs(files: List[str], clu_file: str) -> List[str]:
    """Create a set of recommendations for the first half of the openings within an opening
    repertoire. Used for testing the viability of the recommendations"""
    rec_files = []

    for openings_db in files:
        rec_files.append(
            process_clustering(
                clu_file,
                openings_db,
                export_recs="exp_data/example_user_recs",
                test=True,
                color=WHITE,
            )[0]
        )

    return rec_files


def test_results(
    openings_dbs: List[str], opening_recs: List[List[Opening]]
) -> List[float]:
    results: List[float] = [0.0 for _ in openings_dbs]
    for res_index, (openings_file, recs) in enumerate(zip(openings_dbs, opening_recs)):
        with open(openings_file, "r") as ops:
            stored_openings = dicts_to_openings(json.load(ops)["Openings"])

        for opening in recs:
            if check_opening_in_list(opening, stored_openings):
                results[res_index] += 1

    return results


def run_command(cmd):
    print(cmd)
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in process.stdout:  # type: ignore
        line = line.decode(
            encoding=sys.stdout.encoding,
            errors="replace" if (sys.version_info) < (3, 5) else "backslashreplace",
        ).rstrip()

        print(f"Output: {line}")
    return


if __name__ == "__main__":
    test_program(
        clust_file="exp_data/example_large_jsons/lichess_db_ol5_p150.json",
        input_dir="exp_data/example_user_pgns",
        ouput_dir="exp_data/example_user_jsons",
        prefix="LBU",
    )
