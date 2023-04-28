import json

import clustering
from chess import BLACK, WHITE


def main():
    clust_model = clustering.cluster_data("opening_databases/lichess_db_2017-01.json", BLACK)

    if clust_model is None:
        print("error clust_model is none")
        return

    clustering.plot_data(clust_model, WHITE, "opening_databases/lichess_db_2017-01.json")
    clustering.plot_data(clust_model, BLACK, "opening_databases/lichess_db_2017-01.json")

    clustering.plot_data(clust_model, WHITE, "opening_databases/s_example.json")
    clustering.plot_data(clust_model, BLACK, "opening_databases/s_example.json")

    openings = clustering.find_fitting_openings(clust_model,
                                                "opening_databases/s_example.json",
                                                "opening_databases/lichess_db_2017-01.json",
                                                BLACK)

    for opening in openings:
        uci_notation = ''

        white_moves_split = opening.get_most_common_opening(WHITE).split('|')[:-1]
        black_moves_split = opening.get_most_common_opening(BLACK).split('|')[:-1]

        for i in range(len(white_moves_split)):
            uci_notation += white_moves_split[i] + " " + black_moves_split[i] + " "

        opening["UCI Opening"] = uci_notation

        opening.pop("Most Common Opening White", None)
        opening.pop("Most Common Opening Black", None)

    out_dict = {}
    out_dict["Openings"] = openings
    with open('results/test_b.txt', "w") as out:
        out.write(json.dumps(out_dict, indent=4))
    return


if __name__ == '__main__':
    main()
