"""Contains all the files relating to clustering the data, and utilising the clusters
to recommend chess games"""

import argparse
import json
import os.path
import random
from math import sqrt
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Used for creating clusters and selecting BIC scores
import scipy.stats
from chess import BLACK, WHITE

# Used for Geometric means
from scipy.spatial.distance import cdist, euclidean
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# Allow package import absolutely and relatively, allows file
# To be used in the package, and as __main__

try:
    from opening import AXIS_NAMES, AXIS_NAMES_SHORT, Opening, check_opening_in_list
except ImportError:
    from src.opening import AXIS_NAMES, AXIS_NAMES_SHORT, Opening, check_opening_in_list

### READING FILE FUNCTIONS ###


def read_opening_data(opening: Opening, color: bool) -> List[float]:
    """Gets to values for the opening for a opening and a color"""
    if color == WHITE:
        return opening.w_opening_values
    return opening.b_opening_values


def json_to_dicts(in_file: str) -> List[Dict[str, Union[str, List[Union[int, float]]]]]:
    """Converts Bitboard JSON object into Dict of data"""
    data: List[Dict[str, Union[str, List[Union[int, float]]]]]

    with open(in_file, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return []

    if not data:
        return []

    return data["Openings"]  # type: ignore


def dicts_to_openings(dict_list: List[Dict[str, Any]]) -> List[Opening]:
    """Convert a list of dict interpreations of openings into an opening list"""
    out_list: List[Opening] = []
    for opening in dict_list:
        out_opening = Opening()
        err = out_opening.from_dict(opening)

        if err:
            continue
        else:
            out_list.append(out_opening)

    return out_list


### CONTROL FLOW FUNCTIONS ###


def process_clustering(
    clust_file: str,
    user_file: str,
    color: Union[bool, None] = None,
    export_recs: Union[str, None] = None,
    export_plots: Union[str, None] = None,
    clust_count: int = 0,
    test: bool = False,
    method: str = "geo",
) -> List[str]:
    """Call all relevent functions to recommend a set of openings for any or both
    colors"""
    random.seed()
    output_files = []

    if not file_compatability(clust_file, user_file):
        print(
            "!!! Error: The methods of quantifying openings is not identical. Cannot compare repertoires"
        )
        print(
            "--- Quantifying functions have to be the same and in the same order between repertoires ---"
        )
        return output_files

    if color is None:
        output_files.append(
            process_for_color(
                clust_file,
                user_file,
                WHITE,
                export_recs,
                export_plots,
                clust_count,
                test,
                method=method,
            )
        )

        output_files.append(
            process_for_color(
                clust_file,
                user_file,
                BLACK,
                export_recs,
                export_plots,
                clust_count,
                test,
                method=method,
            )
        )
    else:
        output_files.append(
            process_for_color(
                clust_file,
                user_file,
                color,
                export_recs,
                export_plots,
                clust_count,
                test,
                method=method,
            )
        )

    print("--- Recommendations Completed! ---")
    return output_files


def file_compatability(file_1: str, file_2: str) -> bool:
    """Check if the two files methods of quantifying the openings into values are identical.
    If they are not, there is no way to reliably compare the two methods."""
    with open(file_1, "r", encoding="utf-8") as file:
        try:
            value_names_1 = json.load(file)["Value Names"]
        except json.JSONDecodeError:
            return False
        except KeyError:
            print(f"!!! Error: Value names not found, please reconvert {file_2}")
            return False

    with open(file_2, "r", encoding="utf-8") as file:
        try:
            value_names_2 = json.load(file)["Value Names"]
        except json.JSONDecodeError:
            return False
        except KeyError:
            print(f"!!! Error: Value names not found, please reconvert {file_2}")
            return False

    return value_names_1 == value_names_2


def process_for_color(
    clust_file: str,
    user_file: str,
    color: bool,
    export_recs: Union[str, None] = None,
    export_plots: Union[str, None] = None,
    clust_count: int = 0,
    test: bool = False,
    method: str = "geo",
) -> str:
    """Call all relevent functions to recommend a set of openings for a specific color
    Returns the path of thje file that contains the recommendations"""

    # Get the base file name for output purposes
    base_file_name = os.path.basename(user_file).split(".")[0]
    if color == WHITE:
        color_str = "White"
    else:
        color_str = "Black"

    clust_model = cluster_data(clust_file, color, clust_count)

    if clust_model is None:
        print("!!! Error clust_model is None !!!")
        return ""

    # If the user defines to export the plots
    if export_plots is not None:
        plot_points_in_clusters(clust_model, color, clust_file, export_plots)
        plot_points_in_clusters(clust_model, color, user_file, export_plots)

    # Find the openings
    openings = find_fitting_openings(
        clust_model, user_file, clust_file, color, test=test, method=method
    )

    # Convert openings into a dict type
    out_dict = {}
    out_dict["Recommended Openings"] = []

    for opening in openings:
        # Convert unusual openings into UCI notation for easier readability and conversion
        # into PGN or SAN if thge user wishes
        opening.gen_opening_move_string()

        out_dict["Recommended Openings"].append(opening.to_simple_dict())

    if export_recs is None:
        export_recs = os.path.dirname(user_file)

    output_file_name = f"{export_recs}/{base_file_name}_{color_str}.json"
    with open(output_file_name, "w", encoding="utf-8") as out:
        out.write(json.dumps(out_dict, indent=4))
    return output_file_name


### CLUSTERING FUNCTIONS ###


def cluster_data(
    clust_file: str, color: bool, num_components: int = 0
) -> Union[GaussianMixture, None]:
    """Create a Gaussian mixture model using the data within a cluster file and a
    players color. Can manually select the number of components"""

    openings: List[Opening] = []

    while not openings:
        openings = dicts_to_openings(json_to_dicts(clust_file))

        if not openings:
            return None

    games_recorded: int = 0

    for opening in openings:
        games_recorded += opening.results_total[0]

    # The cluster factor determines that if an opening is used more than
    # each one-thousanth of a time, an aditional point is placed within
    # the data to impact clustering. Weighting the clusters
    cluster_factor = float(games_recorded / 1000)
    print(f"--- {games_recorded} games are recorded in the opening database ---")
    print(f"--- {len(openings)} openings are recorded in the opening database ---")

    print("--- Extracting data from the openings ---")
    data_set = extract_openings(openings, color, cluster_factor)

    print("--- Extracted opening data ---")

    print("--- Calculating BIC score and covariance type ---")
    best_scoring = get_bic_score(data_set)
    # Run multiple times to make the score more reliable
    for _ in range(2):
        temp_best_score = get_bic_score(data_set)
        if temp_best_score["BIC score"] < best_scoring["BIC score"]:
            best_scoring = temp_best_score

    print("--- Finished calculating BIC score and covariance type ---")
    # If the number of clusters is not defined

    print("--- Clustering Data ---")
    if num_components == 0:
        cluster_model = create_clustering(
            data_set,
            int(best_scoring["Number of components"]),  # type: ignore
            str(best_scoring["Type of covariance"]),  # type: ignore
        )
    # If the number of components is defined but incorrectly
    elif num_components < 0:
        print("!!! Num_component is less than 0 !!!")
        return None

    else:
        cluster_model = create_clustering(
            data_set, num_components, str(best_scoring["Type of covariance"])
        )

    print("--- Finished Clustering Data ---")
    return cluster_model


def get_bic_score(data_set: np.ndarray) -> pd.DataFrame:
    """Get the best BIC scoring number of clusters and covariance type for a dataset"""
    param_grid = {
        "n_components": range(1, 14),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }

    # Use the sci-learn kit to calculate the BIC values for each of the values
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_BIC_scorer
    )
    grid_search.fit(data_set)

    data_frame = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    data_frame["mean_test_score"] = -data_frame["mean_test_score"]

    # Rename for displaying / printing the data
    data_frame = data_frame.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )

    # Order data by lowest BIC Score
    data_frame: pd.DataFrame = data_frame.sort_values(by="BIC score").head()

    print("--- Finished Computing BIC Scores ---")

    # Return best scoring method
    return data_frame.iloc[0]  # type: ignore


def create_clustering(
    data_set: np.ndarray, components: int, convariance_type: str
) -> GaussianMixture:
    """Create a clustering model on a data set for a covariance type and a number of clusters"""
    clust_model = GaussianMixture(
        n_components=components, covariance_type=convariance_type, n_init=5
    )  # type: ignore
    clust_model.fit(data_set)

    return clust_model


def gmm_BIC_scorer(estimator: GaussianMixture, X):
    """Use the SCI-Learn method to return the BIC values"""
    return -estimator.bic(X)


def extract_openings(
    openings: List[Opening], color: bool, weighting: float = 0
) -> np.ndarray:
    """Convert a list of openings into a list of the values for each opening"""

    # This is weird but done to create a np array of the correct dimensions
    print("--- Initialising data set and extracting first opening ---")
    first_opening_vals = read_opening_data(openings[0], color)
    data_set = np.array([first_opening_vals])
    data_set = np.append(data_set, [first_opening_vals], axis=0)

    # if the weighting is supplied, add a datapoint if it passes each threshhold
    if weighting:
        print("--- Weighting first opening ---")
        opening_count = int((openings[0].results_total[0] - 1) / weighting)
        for _ in range(opening_count):
            data_set = np.append(data_set, [first_opening_vals], axis=0)

    print("--- Processing further openings ---")
    for opening in openings[1:]:
        opening_values = read_opening_data(opening, color)
        data_set = np.append(data_set, [opening_values], axis=0)

        # if the weighting is supplied, add a datapoint if it passes each threshhold
        if weighting:
            opening_count = int(((opening.results_total[0]) / weighting))
            for _ in range(opening_count):
                data_set = np.append(data_set, [opening_values], axis=0)

    print(f"--- Opening data extracted. Dataset contains {len(data_set)} points")

    return data_set


## RECOMMEND OPENING FUNCTIONS ###


def find_fitting_openings(
    cluster_model: GaussianMixture,
    users_opening_file: str,
    large_game_file: str,
    color: bool,
    recommendations: int = 10,
    test: bool = False,
    method: str = "geo",
) -> List[Opening]:
    """Get a list of recommended openings within a opening database for a user repertoire"""
    # Check the repertoires and convert into lists of Openings

    print("--- Finding Recommended Openings ---")
    user_openings = dicts_to_openings(json_to_dicts(users_opening_file))
    if not user_openings:
        return []

    if test:
        user_openings = user_openings[: len(user_openings) // 2]

    large_openings = dicts_to_openings(json_to_dicts(large_game_file))
    if not large_openings:
        return []

    # Put all the openings into their clusters
    print("--- Putting Openings in Clusters ---")
    user_openings_in_clusters = get_openings_in_clusters(
        cluster_model, user_openings, color
    )
    all_openings_in_clusters = get_openings_in_clusters(
        cluster_model, large_openings, color
    )

    # Count each cluster for the amount of GAMES within them, NOT UNIQUE OPENINGS
    cluster_counts = get_cluster_game_counts(user_openings_in_clusters)

    # Decide how to share the recommendations between the openings based on one of two methods
    print("--- Apportioning Recommendations for Clusters ---")
    rec_openins_per_cluster = jefferson_share_method(cluster_counts, recommendations)

    recommended_openings = find_recommended_openings(
        user_openings_in_clusters,
        all_openings_in_clusters,
        rec_openins_per_cluster,
        color,
        method=method,
    )

    print("--- Found Recommended Openings! ---\n")
    return recommended_openings


def get_openings_in_clusters(
    cluster_model: GaussianMixture, openings: List[Opening], color: bool
) -> List[List[Opening]]:
    """Put each opening into its correstponding cluster index in a list"""

    openings_in_clusters = [[] for _ in range(len(cluster_model.means_))]  # type:ignore

    for opening in openings:
        openings_in_clusters[
            cluster_model.predict([read_opening_data(opening, color)])[0]  # type:ignore
        ].append(
            opening
        )  # type:ignore

    return openings_in_clusters


def get_cluster_game_counts(openings_in_clusters: List[List[Opening]]) -> List[int]:
    """Get the amount of games played within a cluster"""

    cluster_counts = [0 for _ in range(len(openings_in_clusters))]

    for i, cluster in enumerate(openings_in_clusters):
        for opening in cluster:
            cluster_counts[i] += opening.results_total[0]

    return cluster_counts


def jefferson_share_method(
    clusts_and_counts: List[int], number_of_recommendations: int
) -> List[int]:
    """Use the sharing method devised by jefferson to distribute the amount of recommendations
    between the clusters based on the amount of games shared. Information on the method can
    be found here:
    https://math.libretexts.org/Bookshelves/Applied_Mathematics/Math_in_Society_(Lippman)/04:_Apportionment/4.03:_Jeffersons_Method
    """

    # Get the total tally of games played
    total_games: int = 0
    for count in clusts_and_counts:
        total_games += count

    # Create a divisor which if all cluster values are rounded down, the total would be
    # less than the number of recommendations
    divisor: float = total_games / number_of_recommendations

    # Create a decrement, as decreasing the divisor would increase the number of recommendations
    divisor_decr: float = divisor * 0.05

    # Get the first values for the clusters (integer conversion always rounds down)
    rec_games_per_clust = [int(count / divisor) for count in clusts_and_counts]

    # Escape count incase the loop gets stuck
    escape_int: int = 0
    while sum(rec_games_per_clust) != number_of_recommendations:
        # Reprocess the counts
        rec_games_per_clust = [
            int(count / (divisor - divisor_decr)) for count in clusts_and_counts
        ]

        # If there are more recommendations, reduce the decrement and retry
        if sum(rec_games_per_clust) > number_of_recommendations:
            divisor_decr /= 2
        if escape_int >= 2000:
            break
        else:
            # Otherwise, reduce the divisor
            divisor -= divisor_decr

        escape_int += 1

    return rec_games_per_clust


def find_recommended_openings(
    user_openings_in_clusters: List[List[Opening]],
    all_openings_in_clusters: List[List[Opening]],
    rec_openings_per_cluster: List[int],
    color: bool,
    method: str = "geo",
) -> List[Opening]:
    """Find the recommended openings for a users openings within clusters, all openings
    in clusters, and with the ammount of reccommended openings per cluster"""

    cluster_dime_lims = get_cluster_dim_limits(all_openings_in_clusters, color)
    recommended_openings = []

    for i, rec_ammount in enumerate(rec_openings_per_cluster):
        if rec_ammount == 0:
            continue
        # for each clusters if it is apportioned to have an opening
        # either use lowest_dists_in_cluster or closests_to_geo_median
        if method == "rand":
            rec_cluster_openings = get_rand_in_cluster(
                user_openings_in_clusters[i],
                all_openings_in_clusters[i],
                rec_ammount,
                color,
                w=False,
            )
        elif method == "common":
            rec_cluster_openings = most_common_in_cluster(
                user_openings_in_clusters[i], all_openings_in_clusters[i], rec_ammount
            )
        elif method == "wrand":
            rec_cluster_openings = get_rand_in_cluster(
                user_openings_in_clusters[i],
                all_openings_in_clusters[i],
                rec_ammount,
                color,
                w=True,
            )

        else:
            rec_cluster_openings = closests_to_geo_median(
                user_openings_in_clusters[i],
                all_openings_in_clusters[i],
                rec_ammount,
                cluster_dime_lims[i],
                color,
            )

        for opening in rec_cluster_openings:
            # Add each recommended opening to the list
            recommended_openings.append(opening)

    return recommended_openings


def get_cluster_dim_limits(
    openings_in_clusters: List[List[Opening]], color
) -> List[List[List[float]]]:
    """
    Returns the minimum and maximum for each dimension for each cluster
    """
    len_open_data = len(read_opening_data(openings_in_clusters[0][0], color))
    cluster_dimensions = [
        [[0.0, 0.0] for _ in range(len_open_data)]
        for _ in range(len(openings_in_clusters))
    ]

    # Initialise all the cluster's dimensions to contain the first opening values
    for clust_no, opening_list in enumerate(openings_in_clusters):
        first_opening_vals = read_opening_data(opening_list[0], color)
        for clust_dimension, opening_val in zip(
            cluster_dimensions[clust_no], first_opening_vals
        ):
            # Index 0 contains the minimum, index 1 is the maximum
            clust_dimension[0] = opening_val
            clust_dimension[1] = opening_val

        # Go through all the next openings
        for opening in opening_list[1:]:
            for clust_dimension, val in zip(
                cluster_dimensions[clust_no], read_opening_data(opening, color)
            ):
                # Check if the values exceed the boundaries before.
                # Replace them if they do
                if val < clust_dimension[0]:
                    clust_dimension[0] = val
                elif val > clust_dimension[1]:
                    clust_dimension[1] = val

    return cluster_dimensions


def get_rand_in_cluster(
    user_openings: List[Opening],
    all_openings: List[Opening],
    rec_quant: int,
    color: bool,
    w=False,
) -> List[Opening]:
    """Get a given ammount of random openings within a cluster"""
    valid_openings = get_valid_openings(all_openings, user_openings, color)
    if not w:
        return np.random.choice(valid_openings, size=rec_quant)  # type: ignore

    weights = [op.results_total[0] for op in valid_openings]
    total = sum(weights)
    weights = [val / total for val in weights]
    return np.random.choice(valid_openings, size=rec_quant, p=weights)  # type: ignore


def most_common_in_cluster(
    user_openings: List[Opening], all_openings: List[Opening], rec_quant: int
) -> List[Opening]:
    """Get the most common openings within a cluster"""

    output: List[Opening] = []
    for opening in all_openings:
        if check_opening_in_list(opening, user_openings):
            continue
        if len(output) < rec_quant:
            output.append(opening)
            continue

        replace_index = -1
        min_games = opening.results_total[0]
        for check_index, op in enumerate(output):
            if op.results_total[0] < min_games:
                replace_index = check_index
                min_games = op.results_total[0]

        if replace_index != -1:
            output[replace_index] = opening

    return output


def lowest_dists_in_cluster(
    user_openings: List[Opening],
    all_openings: List[Opening],
    rec_quant: int,
    cluster_dim_limits: List[List[float]],
    color: bool,
) -> List[Opening]:
    """Find the best given number openings for a cluster for a user repertoire
    from a large opening database"""

    # Keep track of the count each set of opening values is used
    # This is for using weighting for recommendations
    user_opening_values: List[List[float]] = []
    user_opening_weights: List[int] = []

    for opening in user_openings:
        user_opening_values.append(
            normalise_values(read_opening_data(opening, color), cluster_dim_limits)
        )
        user_opening_weights.append(opening.results_total[0])

    # Keep track of the score of each opening. could be a tuple to save scope space
    opening_scores = []
    recommened_openings = []
    valid_openings = get_valid_openings(all_openings, user_openings, color)

    for opening in valid_openings:
        # Get the distance score for each opening
        score = find_distance(
            read_opening_data(opening, color), user_opening_values, user_opening_weights
        )

        # If there are less openings than the amount apportioned, add it to the recommended
        if len(recommened_openings) < rec_quant:
            recommened_openings.append(opening)
            opening_scores.append(score)

        else:
            check_score = score
            index = -1
            # Find the opening with the largest score to replace
            for replace_index, old_score in enumerate(opening_scores):
                if old_score > check_score:
                    index = replace_index
                    check_score = old_score

            # If there is an opening that this opening can replace, replace it
            if index != -1:
                recommened_openings[index] = opening
                opening_scores[index] = score

    return recommened_openings


def closests_to_geo_median(
    user_openings: List[Opening],
    all_openings: List[Opening],
    rec_quant: int,
    cluster_dim_limits: List[List[float]],
    color: bool,
) -> List[Opening]:
    """Find openings not used by the user which are closest to the geometric
    median of the values of the weighted openings used by the user"""

    user_opening_values: List[List[float]] = []
    user_opening_weights: List[int] = []

    for opening in user_openings:
        user_opening_values.append(
            normalise_values(read_opening_data(opening, color), cluster_dim_limits)
        )
        user_opening_weights.append(opening.results_total[0])

    user_geometric_median = get_geometric_median(
        user_opening_values, user_opening_weights
    )

    closest = []
    closest_scores = []

    for opening in all_openings:
        opening_values = normalise_values(
            read_opening_data(opening, color), cluster_dim_limits
        )
        dist_score = find_distance(user_geometric_median, [opening_values], [1])

        if check_opening_in_list(opening, user_openings):
            continue

        if len(closest) < rec_quant:
            closest.append(opening)

        else:
            check_score = dist_score
            index = -1
            # Find the opening with the largest score to replace
            for replace_index, old_score in enumerate(closest_scores):
                if old_score > check_score:
                    index = replace_index
                    check_score = old_score

            # If there is an opening that this opening can replace, replace it
            if index != -1:
                closest[index] = opening
                closest_scores[index] = dist_score

    return closest


def find_distance(
    point_data: Union[List[float], np.ndarray],
    data_points: List[List[float]],
    weights: List[int],
) -> float:
    """Get the total distance between two points on a graph with the same dimensions"""
    score: float = 0.0

    # Just pythag multiplied by the weighting of the point
    for values, weight in zip(data_points, weights):
        temp_score: float = 0.0

        for val, datum in zip(values, point_data):
            temp_score += (val - datum) ** 2

        temp_score = sqrt(temp_score)

        temp_score *= weight

        score += temp_score
    return score


def get_centre_of_mass(point_data: List[np.ndarray], weight: bool = True) -> np.ndarray:
    """Get the centre of mass of a series of points"""
    com = []
    if weight:
        tot_mass = sum(point_data[-1])
        for axis in point_data[:-1]:
            axis_com = 0
            for value, weight in zip(axis, point_data[-1]):
                axis_com += value * weight
            com.append(axis_com / tot_mass)
    else:
        tot_mass = len(point_data[0])
        for axis in point_data:
            axis_com = 0
            for value in axis:
                axis_com += value
            com.append(axis / tot_mass)

    return np.array(com)


def get_geometric_median(
    values_list: List[List[float]], weights: List[int], epsillon: float = 1e-5
) -> np.ndarray:
    """Calculate the geometric median for a set of values and weights.
    This method is an altered version of the one found here: https://stackoverflow.com/a/30305181
    which is based on the algorithm found here: https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423

    The function is altered to allow the implementaion of weights for each point"""

    # Add weightings
    unpacked_data: List[List[float]] = []
    for values, weight in zip(values_list, weights):
        for _ in range(weight):
            unpacked_data.append(values)

    np_data = np.array(unpacked_data)

    means = np.mean(np_data, 0)
    # While the difference is greater than epsilon (as dealing with floats)
    while True:
        distances = cdist(np_data, [means])
        nonzeros = (distances != 0)[:, 0]

        inv_dists = 1 / distances[nonzeros]
        sum_inv_dists = np.sum(inv_dists)

        weights = inv_dists / sum_inv_dists
        totals = np.sum(weights * np_data[nonzeros], 0)

        num_zeros = len(np_data) - np.sum(nonzeros)

        if num_zeros == 0:
            test_medians = totals

        elif num_zeros == len(np_data):
            return means

        else:
            R = (totals - means) * sum_inv_dists
            r = np.linalg.norm(R)

            rinv = 0 if r == 0 else num_zeros / r

            test_medians = max(0, 1 - rinv) * totals + min(1, rinv) * means

        if euclidean(means, test_medians) < epsillon:
            return test_medians

        means = test_medians


def normalise_values(values: List[Union[int, float]], dimensions: List[List[float]]):
    """Change the values of an opening to be between 0-1 for a given cluster"""
    normalised_values = []
    for dimension, value in zip(dimensions, values):
        # Check if the dimension for each value exceed the usual boundry for the cluster
        # If it does, bring it inbounds
        if value < dimension[0]:
            value = (dimension[1] - dimension[0]) * 0.25
        elif value > dimension[1]:
            value = (dimension[1] - dimension[0]) * 0.75

        # Calculate the value as a number between 0-1
        normal_value = value - dimension[0]

        normal_range = dimension[1] - dimension[0]

        # Check if the range is 0 to prevent div by 0
        if normal_range != 0:
            normal_value /= dimension[1] - dimension[0]
        else:
            normal_value = dimension[0]

        normalised_values.append(normal_value)

    return normalised_values


def get_valid_openings(
    all_openings: List[Opening], user_openings: List[Opening], color: bool
) -> List[Opening]:
    """Get openings from a list of openings that aren't part of the users openings
    and that have a >40% chance of not losing"""
    valid_openings: List[Opening] = []
    for opening in all_openings:
        if check_opening_in_list(opening, user_openings):
            continue

        # Check if the opening is competative, does not lose more than 60% of games
        if color == WHITE:
            loss_index = 3
        else:
            loss_index = 1

        if (opening.results_total[loss_index] / opening.results_total[0]) > 0.6:
            continue

        valid_openings.append(opening)

    return valid_openings


### PLOTING FUNCTIONS ###


def plot_points_in_clusters(
    clust_model: GaussianMixture,
    color: bool,
    points_file: str,
    dest: str,
    prop_sizing: bool = False,
) -> int:
    """Plot openings from a JSON file against a Gaussian model for a given chess players color.
    Each opening is colored according to its cluster given by a cluster model argument.
    """

    openings: List[Opening]
    openings = dicts_to_openings(json_to_dicts(points_file))

    if not openings:
        # The openings could not be extracted
        return 1

    b_f_name = os.path.basename(points_file).split(".")[0][:8]

    cluster_colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "orange",
        "black",
        "grey",
        "goldenrod",
        "fuchsia",
        "purple",
        "chartreuse",
    ]

    if prop_sizing:
        total_games = 0
        min_opening_games = openings[0].results_total[0]
        for opening in openings:
            total_games += opening.results_total[0]

            if opening.results_total[0] < min_opening_games:
                min_opening_games = opening.results_total[0]

        sizing = []
        for opening in openings:
            sizing.append((opening.results_total[0] / min_opening_games) * 15)

    else:
        sizing = 20

    first_opening_vals = read_opening_data(openings[0], color)

    # Create a corresponding array for the color each opening plot belongs in
    # It does not need to repeat for every axis, as only the positon of the points
    # change, not the cluster each opening belongs in
    plot_colors = np.array(
        cluster_colors[clust_model.predict([first_opening_vals])[0]]  # type:ignore
    )  # type:ignore

    for opening in openings[1:]:
        opening_vals = read_opening_data(opening, color)

        # Add the data points to the plots and predict the cluster
        plot_colors = np.append(
            plot_colors,
            cluster_colors[clust_model.predict([opening_vals])[0]],  # type:ignore
        )

    for x_i, x_val in enumerate(first_opening_vals):
        for y_i, y_val in enumerate(first_opening_vals):
            # For each distinct pair of x and y axis values
            if y_i <= x_i:
                # Catches non-distinct pairs
                continue

            print(f"--- Plotting {AXIS_NAMES[x_i]} against {AXIS_NAMES[y_i]} --- ")

            # Create a new array of values for plotting
            x_values = np.array([x_val])
            y_values = np.array([y_val])
            # cluster = clust_model.predict([first_opening_vals])][0]

            for opening in openings[1:]:
                opening_vals = read_opening_data(opening, color)

                # Add the data points to the plots and predict the cluster
                x_values = np.append(x_values, opening_vals[x_i])
                y_values = np.append(y_values, opening_vals[y_i])

            # Plot the graph
            plt.scatter(x_values, y_values, c=plot_colors, s=sizing, alpha=0.7)

            # Plot centroids
            for clust, center in enumerate(clust_model.means_):  # type:ignore
                vals = list(center)
                plt.scatter(
                    vals[x_i],
                    vals[y_i],
                    s=70,
                    marker="^",  # type:ignore
                    color=cluster_colors[clust],
                    alpha=0.9,
                )

            if color == WHITE:
                str_color = "White"
            else:
                str_color = "Black"

            plt.title(
                f"Plot of clusters relating to chess opening styles for {str_color} openings"
            )

            # Lable the axis
            plt.xlabel(AXIS_NAMES[x_i])
            plt.ylabel(AXIS_NAMES[y_i])

            # Save toi custom file name
            file_name = (
                dest
                + "/"
                + f"{b_f_name}_{str_color[0]}_{AXIS_NAMES_SHORT[x_i]}_{AXIS_NAMES_SHORT[y_i]}.svg"
            )
            plt.savefig(file_name, format="svg")
            plt.clf()

    return 0


def plot_points(
    openings: List[Opening],
    color: bool,
    dest: str,
    b_f_name: str,
    prop_sizing: bool = False,
) -> int:
    """Plot the values for each opening on each unique parings of openings. Does not color
    points depending on clusters"""

    if prop_sizing:
        total_games = 0
        min_opening_games = openings[0].results_total[0]
        for opening in openings:
            total_games += opening.results_total[0]

            if opening.results_total[0] < min_opening_games:
                min_opening_games = opening.results_total[0]

        sizing = []
        for opening in openings:
            sizing.append((opening.results_total[0] / min_opening_games) * 15)

    else:
        sizing = 20

    first_opening_vals = read_opening_data(openings[0], color)

    for x_i, x_val in enumerate(first_opening_vals):
        for y_i, y_val in enumerate(first_opening_vals):
            # For each distinct pair of x and y axis values
            if y_i <= x_i:
                # Catches non-distinct pairs
                continue

            print(f"--- Plotting {AXIS_NAMES[x_i]} against {AXIS_NAMES[y_i]} --- ")

            # Create a new array of values for plotting
            x_values = np.array([x_val])
            y_values = np.array([y_val])
            # cluster = clust_model.predict([first_opening_vals])][0]

            for opening in openings[1:]:
                opening_vals = read_opening_data(opening, color)

                # Add the data points to the plots and predict the cluster
                x_values = np.append(x_values, opening_vals[x_i])
                y_values = np.append(y_values, opening_vals[y_i])

            # Plot the graph
            plt.scatter(x_values, y_values, s=sizing, alpha=1)

            if color == WHITE:
                str_color = "White"
            else:
                str_color = "Black"

            plt.title(
                f"Plot of clusters relating to chess opening styles for {str_color} openings"
            )

            # Lable the axis
            plt.xlabel(AXIS_NAMES[x_i])
            plt.ylabel(AXIS_NAMES[y_i])

            # Save toi custom file name
            file_name = (
                dest
                + "/"
                + f"{b_f_name}_{str_color[0]}_{AXIS_NAMES_SHORT[x_i]}_{AXIS_NAMES_SHORT[y_i]}.png"
            )
            plt.savefig(file_name, format="png")
            plt.clf()

    return 0


def plot_recommendations(
    user_openings: List[Opening],
    other_openings: List[Opening],
    color: bool,
    dimensions: List[List[Union[int, float]]],
    x_i: int,
    y_i: int,
    dest: str,
    b_f_name: str,
    prop_sizing: bool = False,
):
    """Plot the users openings and other openings on the same graph.
    User openings will appear in fuchsia and the other openings will appear in orange"""

    # Remove duplicate openings from other list
    unique_openings = [
        op for op in other_openings if not check_opening_in_list(op, user_openings)
    ]

    if len(user_openings) == 0 or len(other_openings) == 0:
        print("!!! Not enough openings to cluster !!!")
        return

    plot_colors = np.array("fuchsia")  # type:ignore

    if prop_sizing:
        total_games = 0
        min_opening_games = user_openings[0].results_total[0]
        for opening in user_openings:
            total_games += opening.results_total[0]

            if opening.results_total[0] < min_opening_games:
                min_opening_games = opening.results_total[0]

        sizing = []
        for opening in user_openings:
            sizing.append((opening.results_total[0] / min_opening_games) * 20)

        for _ in unique_openings:
            sizing.append(20)

    else:
        sizing = 20

    first_opening_vals = normalise_values(
        read_opening_data(user_openings[0], color), dimensions
    )

    print(f"--- Plotting {AXIS_NAMES[x_i]} against {AXIS_NAMES[y_i]} --- ")

    # Create a new array of values for plotting
    x_values = np.array([first_opening_vals[x_i]])
    y_values = np.array([first_opening_vals[y_i]])
    # cluster = clust_model.predict([first_opening_vals])][0]

    # Plot points for users
    for opening in user_openings[1:]:
        plot_colors = np.append(plot_colors, "fuchsia")  # type:ignore

        opening_vals = normalise_values(read_opening_data(opening, color), dimensions)

        # Add the data points to the plots and predict the cluster
        x_values = np.append(x_values, opening_vals[x_i])
        y_values = np.append(y_values, opening_vals[y_i])

    # Plot points for unique openings
    for opening in unique_openings:
        plot_colors = np.append(plot_colors, "orange")

        opening_vals = normalise_values(read_opening_data(opening, color), dimensions)

        # Add the data points to the plots and predict the cluster
        x_values = np.append(x_values, opening_vals[x_i])
        y_values = np.append(y_values, opening_vals[y_i])

    # This image tended to appear more confusing than helpful for a representation
    # of the distance function. A better example can be seen in experiments as
    # demonstrate_recommendations()

    # Plot lines to best recommended openings
    # dim_limits = get_cluster_dim_limits([other_openings], color)
    # rec_openings = find_best_openings_in_cluster(user_openings,
    #                                              other_openings,
    #                                              2,
    #                                              dim_limits[0],
    #                                              color)

    # for rec_opening in rec_openings:
    #     rec_vals = read_opening_data(rec_opening, color)
    #     for us_op in user_openings:
    #         user_vals = read_opening_data(us_op, color)
    #         x_coord = [rec_vals[x_i], user_vals[x_i]]
    #         y_coord = [rec_vals[y_i], user_vals[y_i]]
    #         plt.plot(x_coord, y_coord, color='black', linewidth=0.3)

    # Plot the graph
    plt.scatter(x_values, y_values, s=sizing, c=plot_colors, alpha=1)

    # Plit lines toward lowest scoring opening

    if color == WHITE:
        str_color = "White"
    else:
        str_color = "Black"

    plt.title(
        f"Plot of clusters relating to chess opening styles for {str_color} openings"
    )

    # Lable the axis
    plt.xlabel(AXIS_NAMES[x_i])
    plt.ylabel(AXIS_NAMES[y_i])

    # Save toi custom file name
    file_name = (
        dest
        + "/"
        + f"{b_f_name}_{str_color[0]}_{AXIS_NAMES_SHORT[x_i]}_{AXIS_NAMES_SHORT[y_i]}.png"
    )
    plt.savefig(file_name, format="png")
    plt.clf()


def main():
    """The main function called from the command line.
    Used within the GUI as it imitates a terminal"""
    parser = argparse.ArgumentParser(
        prog="pychora_rec.py",
        usage="python %(prog)s [cluster_file] [user_file] [options]",
        description="A program to recommend additions to a chess opening repertoire, using clusters formed from a large opening database",
    )

    clu_arg = parser.add_argument(
        "cluster_file", help="Path of an opening database to form clusters from"
    )
    us_arg = parser.add_argument(
        "user_file",
        help="Path of a users opening repertoire to get recommendations for",
    )

    # Default values are None Type
    col_arg = parser.add_argument(
        "-c",
        "--color",
        help="Generate recommendations for a specific color. 0: Black, 1: White",
        nargs="?",
    )
    rec_arg = parser.add_argument(
        "-r",
        "--exp_rec",
        help="Path of the destination for opening recommendations",
        nargs="?",
    )
    plt_arg = parser.add_argument(
        "-p",
        "--exp_plt",
        help="Path of the destination for graphical plots from clustering",
        nargs="?",
    )

    # Default type is 0
    qua_arg = parser.add_argument(
        "-q",
        "--clusters",
        help="The ammount of clusters to process the opening database into. Must be between 2-15",
        nargs="?",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    if not os.path.exists(args.cluster_file):
        raise argparse.ArgumentError(clu_arg, "File not found")

    if not os.path.exists(args.user_file):
        raise argparse.ArgumentError(us_arg, message="File not found")

    if args.exp_rec is not None and args.color not in ("0", "1"):
        raise argparse.ArgumentError(
            col_arg, message="Is not valid. Must be 0 (Black) or 1 (White)"
        )

    if args.exp_rec is not None and not os.path.exists(args.exp_rec):
        raise argparse.ArgumentError(rec_arg, message="File not found")

    if args.exp_plt is not None and not os.path.exists(args.exp_plt):
        raise argparse.ArgumentError(plt_arg, message="File not found")

    if args.clusters not in range(2, 16) and args.clusters != 0:
        raise argparse.ArgumentError(
            qua_arg, message="Out of bounds. Must be between 2-15"
        )

    process_clustering(
        args.cluster_file,
        args.user_file,
        args.color,
        args.exp_rec,
        args.exp_plt,
        args.clusters,
    )


if __name__ == "__main__":
    main()
