"""Contains Simple Repertoire class and all functions relating to processing of a PGN database"""
import argparse
import glob
import json
import os
from math import ceil
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, Union

import chess.pgn

try:
    from bitboard import FullBitboard
    from opening import AXIS_NAMES, Opening
except ImportError:
    from src.bitboard import FullBitboard
    from src.opening import AXIS_NAMES, Opening

### CONFIG VALUES ###
CONFIG: Dict[str, Union[int, float]] = {
    "DATABASE_MAX_OPENINGS_PROCCESSED": 10_000_000,
    "DATABASE_OPENING_MIN_PROPORTION": 0.000_030,
    "DATABASE_SAMPLE_SIZE": 200_000,
    "OPENING_MOVES_PER_PLAYER": 6,
    "THREAD_COUNT": 8
}


def process_database(pgn_db: str, dest: str,
                     min_proportion: float = float(CONFIG["DATABASE_OPENING_MIN_PROPORTION"]),
                     opening_length: int = int(CONFIG["OPENING_MOVES_PER_PLAYER"]),
                     max_games: int = int(CONFIG["DATABASE_MAX_OPENINGS_PROCCESSED"])) -> str:
    """Main function which processes and converts a PGN database into a opening JSON database

    @params
    pgn_db:     The path of a pgn database
    dest:       The destination of the JSON opening database, defaults to dir of pgn database
    min_proportion:     The minimum proportion of games each opening must have to be recorded
    opening_length:     The amount of moves per player to be counted as the opening stage
    max_games:  The maximum amount of games to be processed
    max_threads:        The maximum amount of threads for multiprocessing

    @returns
    A string of the path of the created opening json database"""

    max_threads: int = int(CONFIG["THREAD_COUNT"])

    # Split the main database into multiple files to all for multiprocessing
    # Each file has a single thread to prevent collisions.

    print(f"--- Splitting file into a maximum {max_threads} files for threads ---")

    thread_file_names, number_of_games = split_file_for_threads(pgn_db, max_games)

    print(f"--- Completed Splitting file into {len(thread_file_names)} files for threads ---")

    # Sample the openings initially to detect which openings are likely to be used more than
    # the minimum proportion. Could be removed through restructuring.

    with Pool(max_threads) as pool:

        # Each thread contains a dictionary of all the string representaions of the opening moves
        # and how often they are used. They then combine to form a single large dict.

        # Sampling is done first so during the processing stage, where the openings are counted
        # in every file, each opening is counted completely. Processing and sampling separately
        # would mean openings which are recorded have games which may not being included due to
        # not being over the minimum proportions in certain files.

        print("--- Beginning Opening Sampling ---")

        cached_opening_str: Dict[str, int]

        cached_opening_str = combine_dicts(pool.starmap(
            sample_openings,
            [(tfn, min_proportion, opening_length * 2)
                for tfn in thread_file_names]))

        print("--- Opening Sampling Completed ---")

        # Count each opening in every file to make sure all games with sampled openings are
        # counted

        processed_openings: Dict[str, Opening]

        print("\n--- Beginning Opening Bitboard Processing ---")

        processed_openings = combine_opening_dicts(pool.starmap(
            process_openings,
            [(tfn, cached_opening_str, opening_length*2, max_games)
                for tfn in thread_file_names]))

        print("--- Opening Bitboard Processing Completed ---")

    # File Cleanup

    print("--- Temporary File Clean-up ---")
    remove_old_thread_files()

    print("\n--- Beginning Writing Data ---")

    out_dict: Dict[str, Any] = {}

    # Configure output dict
    out_dict["Opening Length"] = int(opening_length)
    out_dict["Sample Rate"] = int(min_proportion*1000000)
    out_dict["Value Names"] = AXIS_NAMES
    out_dict["Openings"] = []

    # Convert each opening into its dict format if its used more than the minimum amount
    for opening in processed_openings.values():
        if opening.results_total[0] >= ceil(number_of_games * min_proportion):
            opening.process_bitboard_values()
            out_dict["Openings"].append(opening.to_dict())

    if dest is None:
        dest = os.path.dirname(pgn_db)

    file_base_name = os.path.basename(pgn_db).split('.')[0][:10]

    # Save it to the given folder
    dest_name = (
        str(dest) + "/" +
        f"{file_base_name}_ol{int(opening_length)}_p{int(min_proportion*1_000_000)}.json")

    with open(dest_name, "w") as out_file:  # type:ignore
        out_file.write(json.dumps(out_dict, indent=4))

    print("\n--- Writing Data Completed ---")
    print("\n--- Conversion Completed ---")

    return dest_name


def sample_openings(pgn_db: str, min_proportion: int, op_len: int) -> Dict[str, int]:
    """Check the occurance of each opening within a file, then if it is more than
    the minimum proportion, record the opening within a dictionary, along with its
    count. Used within multithreading"""
    print(f"--- Initialised Sampling Thread for {pgn_db} ---")
    game_no: int = 0
    cached_opening_str: Dict[str, int] = {}

    with open(pgn_db, "r") as open_pgn_db:  # type:ignore
        while True:
            new_game = chess.pgn.read_game(open_pgn_db)

            # If the sample size is reached. Go through each opening and remove those
            # that occur less than the minimum proportion
            if game_no % CONFIG["DATABASE_SAMPLE_SIZE"] == 0 and game_no != 0:

                # Temp dict prevents update the dictionary while iterating over it
                temp_dict = {}
                for open_str, count in cached_opening_str.items():
                    if count >= ceil(int(min_proportion * CONFIG["DATABASE_SAMPLE_SIZE"])):
                        temp_dict[open_str] = count

                cached_opening_str = temp_dict.copy()
                temp_dict.clear()

                print(f"--- Thread {pgn_db} sampled {game_no} games ---")

            if new_game is None:
                # There are no more games in the database
                break

            # It is impossible within standard running of the program that the file this
            # function uses has more openings than the maximum allowed to be processed.

            opening_string = retrieve_opening_string(new_game, op_len)
            game_no += 1

            if not opening_string:
                continue

            if opening_string in cached_opening_str:
                cached_opening_str[opening_string] += 1
            else:
                cached_opening_str[opening_string] = 1

    # This section allows openings if they were in a smaller sampling size due to the way
    # The database was split, and they appear more than the proportion of that smaller size
    temp_dict = {}

    for open_str, count in cached_opening_str.items():
        if count >= int(min_proportion * (game_no % CONFIG["DATABASE_SAMPLE_SIZE"])):
            temp_dict[open_str] = count

    cached_opening_str = temp_dict.copy()
    temp_dict.clear()

    print(f"--- Thread for {pgn_db} Completed: {len(cached_opening_str)} Openings recorded. ---")
    return cached_opening_str


def process_openings(pgn_db: str, chosen_openings: Dict[str, int],
                     op_len: int, max_games: int) -> Dict[str, Opening]:
    """Convert each opening within a dict of opening strings into an Opening type. A count
    for each time an opening occurs is stored within the opening type
    Used within multithreading"""

    stored_openings: Dict[str, Opening] = {}
    game_no: int = 1

    print(f"--- Thread for Processing {pgn_db} Created ---")

    with open(pgn_db, "r") as open_pgn_db:  # type:ignore
        while True:
            new_game = chess.pgn.read_game(open_pgn_db)

            # Check if reached the end of the database
            if new_game is None:
                break

            if game_no % max_games == 0:
                break

            # REad the string
            opening_string = retrieve_opening_string(new_game, op_len)

            game_no += 1

            if game_no % CONFIG["DATABASE_SAMPLE_SIZE"] == 0:
                print(f"--- Thread {pgn_db} sampled {game_no} games ---")

            if not opening_string:
                # If retrieve_opening_string() could not complete
                continue

            if opening_string not in chosen_openings:
                continue

            if opening_string in stored_openings:
                # If the game has already been stored, process the new game for
                # The opening object
                opening = stored_openings[opening_string]
                opening.process_game_data(new_game, op_len)
                continue

            # If its in the chosen openings, but it has not already been stored

            board = new_game.board()
            opening = Opening()
            new_bb = FullBitboard()

            # Add the moves to the board
            for _, move in zip(range(op_len), new_game.mainline_moves()):
                board.push(move)

            # Process the game and opening object
            new_bb.create_from_board(board)
            opening.attach_bitboard(new_bb)
            opening.process_game_data(new_game, op_len)

            # Add it to the stored openings
            stored_openings[opening_string] = opening

    print(f"--- Thread for Processing {pgn_db} Completed ---")
    return stored_openings


def combine_dicts(dicts: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Combine a list of dictionaries into a single dictionary, prioritises later dictionary
    values where keys match. DO NOT USE FOR COMBINING LISTS OF OPENINGS"""
    output_dict: Dict[Any, Any] = {}
    for dictionary in dicts:
        output_dict.update(dictionary)
    return output_dict


def combine_opening_dicts(dicts: List[Dict[str, Opening]]) -> Dict[str, Opening]:
    """Combine dicts of openings together, making sure that the total results between
    openings are transfered into the final dict"""
    output_dict: Dict[str, Opening] = {}

    for p_op_dict in dicts:
        for key, value in p_op_dict.items():
            # If the opening is already present, increase the total by the values
            if key in output_dict:
                output_dict[key].add_to_results_total(value.results_total)
                output_dict[key].add_to_results_2400(value.results_2400)
                output_dict[key].add_to_results_2400(value.results_1800)
                continue

            # If the opening is not present, add it.
            output_dict[key] = value

    return output_dict


def split_file_for_threads(pgn_db: str, max_games: int) -> Tuple[List[str], int]:
    """Split a large pgn database file into multiple smaller files to allow for
    multiprocessing"""
    file_base_name = os.path.basename(pgn_db)
    thread_value: int = 0
    game_count: int = 0
    last_line_empty: bool = True
    game_start = True

    # Remove any threads in the thread file
    remove_old_thread_files()

    if not os.path.exists('threads'):
        os.mkdir('threads')

    base_thread_pgn_db = "threads/" + "_" + file_base_name + "_T_"

    thread_names: List[str] = []

    with open(pgn_db, "r", errors="ignore") as in_db:  # type:ignore
        # While you haven't reached the end of the file

        line = in_db.readline()
        out_pgn_db = base_thread_pgn_db + str(thread_value) + ".txt"
        thread_names.append(out_pgn_db)

        out_file = open(out_pgn_db, "a+")  # type:ignore

        while line:
            # If the last line was empty and the new line starts with [
            # there is a new game

            if line[0] == "[" and last_line_empty:
                last_line_empty = False
                game_count += 1
                game_start = True

            last_line_empty = bool(line == '\n')

            # If the game count reaches the sample size, change to another outfile
            if game_count % CONFIG["DATABASE_SAMPLE_SIZE"] == 0 and game_start:
                thread_value = (thread_value + 1) % int(CONFIG["THREAD_COUNT"])
                # Close old file
                out_file.close()

                out_pgn_db = base_thread_pgn_db + str(thread_value) + ".txt"

                if out_pgn_db not in thread_names:
                    thread_names.append(out_pgn_db)

                # Reopen new file
                out_file = open(out_pgn_db, "a+")  # type:ignore

            game_start = False
            # Break if the number of games is greater than the maximum allowed

            if game_count >= max_games:
                break

            out_file.write(line)
            line = in_db.readline()

    return (thread_names, game_count)


def retrieve_opening_string(game: chess.pgn.Game, op_len: int) -> str:
    """Return the opening string for a chess game. Returns "" if not valid
    """
    opening_string_w: str = ""
    opening_string_b: str = ""
    zipped = zip(game.mainline_moves(), range(op_len))
    if len(list(zipped)) < op_len:
        # The length of the game is smaller than the number that determines each opening
        # No information can be gained from here
        return ""

    for i, move in zip(range(op_len), game.mainline_moves()):
        if str(move) == "0000":
            # "0000" is the uci form for no move, so the game ended before the end of the opening
            return ""
        if i % 2 == 0:
            opening_string_w = opening_string_w + str(move)
        else:
            opening_string_b = opening_string_b + str(move)

    # This is an old way of doing this from when these were split. Update if possible
    op_str: str = opening_string_w + opening_string_b
    return op_str


def remove_old_thread_files():
    """Detele thread files which are in the threads directory"""
    files = glob.glob("threads/*.txt")
    for file in files:
        try:
            os.remove(file)
        except OSError:
            print("!!! Error deleting Thread file {f} !!!")

### MAIN FUNCTION ###


def main():
    """The main function called from the command line.
    Used within the GUI as it imitates a terminal"""
    parser = argparse.ArgumentParser(
        prog='pychora_db.py', usage='python %(prog)s [cluster_file] [user_file] [options]',
        description='A program to convert chess PGN databases into an JSON opening repertoire')

    db_arg = parser.add_argument('pgn_db', help='Path of the PGN database')

    dest_arg = parser.add_argument(
        '-d', '--dest', nargs='?',
        help='Path to export the JSON opening database to. Defaults to same directory as database')
    prop_arg = parser.add_argument(
        '-p', '--min_prop', nargs='?',
        help='The minimum number of games per 1,000,000 that an opening is used in to store. Between 0 - 2000',
        type=int, default=float(CONFIG["DATABASE_OPENING_MIN_PROPORTION"]*1_000_000))

    len_arg = parser.add_argument(
        '-l', '--op_len', nargs='?',
        help='Length in moves to be considered an opening. Between 4 - 12', type=int,
        default=CONFIG["OPENING_MOVES_PER_PLAYER"])
    parser.add_argument(
        '-g', '--max_games', nargs='?',
        help='The maximum amount of games to process within the PGN database', type=int,
        default=CONFIG["DATABASE_MAX_OPENINGS_PROCCESSED"])

    args = parser.parse_args()

    if not os.path.exists(args.pgn_db):
        raise argparse.ArgumentError(db_arg, 'File not found')

    if args.dest is not None and not os.path.exists(args.dest):
        raise argparse.ArgumentError(dest_arg, 'File not found')

    if args.min_prop < 0 or args.min_prop > 2000:
        raise argparse.ArgumentError(prop_arg, 'Out of bounds. Must be between 0 - 2000 ')

    if args.op_len < 4 or args.op_len > 12:
        raise argparse.ArgumentError(len_arg, 'Out of bounds. Must be between 4 - 12 ')

    process_database(
        args.pgn_db,
        args.dest,
        args.min_prop/1_000_000,
        args.op_len,
        args.max_games
    )


if __name__ == '__main__':
    main()
