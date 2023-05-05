""" Class contianing the methods and values relating to the openings from each player"""

from typing import Any, Dict, List, Union

from chess import BLACK, WHITE, Board, Move
from chess.pgn import Game

try:
    from bitboard import FullBitboard
except ImportError:
    from src.bitboard import FullBitboard


class Opening:
    """Class containing all information relating to openings for storage and evaluation"""

    def __init__(self) -> None:
        # String interpretaion of the opening
        self.bitboard_string: str = ""
        self.bitboard: FullBitboard

        # Designated by Total ammount of games, white wins, draws, black wins
        self.results_total: List[int] = [0, 0, 0, 0]

        # Designate results where the average elo is greater than a certain elo score
        self.results_2400: List[int] = [0, 0, 0, 0]
        self.results_1800: List[int] = [0, 0, 0, 0]

        # The openings which give the same bitboard, and how often they are used
        self.w_openings_count: Dict[str, int] = {}
        self.b_openings_count: Dict[str, int] = {}

        # The following lists contian the data for the openings used in clustering
        # Form : TO_DO
        self.w_opening_values: List[float] = [0 for _ in AXIS_FUNCTIONS]
        self.b_opening_values: List[float] = [0 for _ in AXIS_FUNCTIONS]

        # The following is only used when exporting recommended Games
        self.uci_opening: str = ""
        self.san_opening: str = ""

    def process_opening_moves(self, game: Game, op_len: int) -> None:
        """Find what a set of opening"""
        white_moves: str = ""
        black_moves: str = ""
        for i, move in zip(range(op_len), game.mainline_moves()):
            if i % 2 == 0:
                white_moves = white_moves + move.uci() + "|"
            else:
                black_moves = black_moves + move.uci() + "|"

        self.update_uci_transpositions(white_moves, black_moves)

    def update_uci_transpositions(self, white_moves: str, black_moves: str):
        """Process the opening moves and add them to the count"""
        if white_moves in self.w_openings_count:
            self.w_openings_count[white_moves] += 1
        else:
            self.w_openings_count[white_moves] = 1

        if black_moves in self.b_openings_count:
            self.b_openings_count[black_moves] += 1
        else:
            self.b_openings_count[black_moves] = 1

        return

    def process_game_data(self, game: Game, op_len: int) -> int:
        """Process the information about the game from the pgn file"""
        total_elo: float = 0
        players_with_elo: int = 0

        avg_elo: float = 0

        if "WhiteElo" in game.headers:
            if game.headers["WhiteElo"]:
                if game.headers["WhiteElo"] != "?":
                    try:
                        total_elo += int(game.headers["WhiteElo"])
                        players_with_elo += 1
                    except ValueError:
                        print(f"Invalid White elo: {game.headers['WhiteElo']}")
        if "BlackElo" in game.headers:
            if game.headers["BlackElo"]:
                if game.headers["BlackElo"] != "?":
                    try:
                        total_elo += int(game.headers["BlackElo"])
                        players_with_elo += 1
                    except ValueError:
                        print(f"Invalid Black elo: {game.headers['WhiteElo']}")

        if total_elo != 0:
            avg_elo = total_elo / players_with_elo

        if "Result" not in game.headers:
            input("No result")
            return -1

        result = _get_outcome(game.headers["Result"])

        self.process_opening_moves(game, op_len)

        if avg_elo >= 2400:
            self.results_2400[0] += 1
            self.results_2400[result] += 1

        if avg_elo >= 1800:
            self.results_1800[0] += 1
            self.results_1800[result] += 1

        self.results_total[0] += 1
        self.results_total[result] += 1
        return 0

    def attach_bitboard(self, bitboard: FullBitboard) -> None:
        """Create the bitboard for a given opening and its transpositions"""
        self.bitboard = bitboard
        self.bitboard_string = bitboard.bb_to_store_string()
        return

    def add_to_results_total(self, other_results: List[int]) -> int:
        """Increment the results  for another result"""
        if len(other_results) != 4:
            return -1

        for i in range(4):
            self.results_total[i] += other_results[i]

        return 0

    def add_to_results_2400(self, other_results: List[int]) -> int:
        """Increment the results for 2400+ ratings for another result"""
        if len(other_results) != 4:
            return -1

        for i in range(4):
            self.results_2400[i] += other_results[i]

        return 0

    def add_to_results_1800(self, other_results: List[int]) -> int:
        """Increment the results for 1800+ ratings for another result"""
        if len(other_results) != 4:
            return -1

        for i in range(4):
            self.results_1800[i] += other_results[i]

        return 0

    def process_bitboard_values(self) -> None:
        """Get the values for a game based on the values given at the top of this file"""

        for index, func in enumerate(AXIS_FUNCTIONS):
            self.w_opening_values[index] = func(self, WHITE)
            self.b_opening_values[index] = func(self, BLACK)

    def to_dict(self) -> Dict[str, Union[str, List[int], List[float]]]:
        """Converts an opening type to a dictionary type. All but the most used
        opening is lost during this process however

        @return:
        Dict : contains keys:
            'Most Common Opening [White/Black]'
            '[White/Black] Opening Values'
            '[2400+/1800+/Total] Results'
        """
        out_dict: Dict[str, Union[str, List[int], List[float]]] = {}
        out_dict["Bitboard String"] = self.bitboard_string

        out_dict["Most Common Opening White"] = self.get_most_common_opening(WHITE)
        out_dict["Most Common Opening Black"] = self.get_most_common_opening(BLACK)

        out_dict["White Opening Values"] = self.w_opening_values
        out_dict["Black Opening Values"] = self.b_opening_values

        out_dict["2400+ Results"] = self.results_2400
        out_dict["1800+ Results"] = self.results_1800
        out_dict["Total Results"] = self.results_total

        if self.uci_opening:
            out_dict["UCI Opening"] = self.uci_opening

        if self.san_opening:
            out_dict["SAN Opening"] = self.san_opening

        return out_dict

    def to_simple_dict(self) -> Dict[str, Any]:
        out_dict: Dict[str, Any] = {}
        if self.uci_opening:
            out_dict["UCI Opening"] = self.uci_opening

        if self.san_opening:
            out_dict["SAN Opening"] = self.san_opening

        out_dict["Values Format"] = str(AXIS_NAMES_SHORT)
        out_dict["White Opening Values"] = str(self.w_opening_values)
        out_dict["Black Opening Values"] = str(self.b_opening_values)

        out_dict["Result Format"] = "[Total, White, Draws, Black]"
        out_dict["Total Results"] = str(self.results_total)
        out_dict["1800+ Results"] = str(self.results_1800)
        out_dict["2400+ Results"] = str(self.results_2400)

        return out_dict

    def get_most_common_opening(self, color: bool) -> str:
        """Find the most common opening for a color and return the string"""
        most_common_opening: str = ""
        most_common_opening_count: int = 0

        if color == WHITE:
            for opening, count in self.w_openings_count.items():
                if count > most_common_opening_count:
                    most_common_opening = str(opening)
        else:
            for opening, count in self.b_openings_count.items():
                if count > most_common_opening_count:
                    most_common_opening = str(opening)

        return most_common_opening

    def from_dict(self, dictionary: Dict[str, Any]) -> bool:
        """Convert an opening represented as a dictionary into an opening type

        @return:
        bool : Determines whether an error occured."""

        # Define temp variables to prevent partial writing if an exception is hit
        bb_str: str

        mco_white: str
        mco_black: str

        ov_white: List[float]
        ov_black: List[float]

        res_2400: List[int]
        res_1800: List[int]
        res_tot: List[int]

        try:
            bb_str = dictionary["Bitboard String"]

            mco_white = dictionary["Most Common Opening White"]
            mco_black = dictionary["Most Common Opening Black"]

            ov_white = dictionary["White Opening Values"]
            ov_black = dictionary["Black Opening Values"]

            res_2400 = dictionary["2400+ Results"]
            res_1800 = dictionary["1800+ Results"]
            res_tot = dictionary["Total Results"]

        except (KeyError, TypeError) as _:
            print("Dictionary Conversion Failed")
            return True

        self.bitboard_string = bb_str
        self.bitboard = FullBitboard()
        self.bitboard.store_string_to_bb(bb_str)

        self.w_opening_values = ov_white
        self.b_opening_values = ov_black

        self.w_openings_count[mco_white] = 1
        self.b_openings_count[mco_black] = 1

        self.results_total = res_tot
        self.results_1800 = res_1800
        self.results_2400 = res_2400

        return False

    def gen_opening_move_string(self):
        """Generate a user-readable series of opening moves for this opening"""
        temp_board = Board()
        uci_notation = ""

        white_moves_split = list(self.w_openings_count.keys())[0].split("|")[:-1]
        black_moves_split = list(self.b_openings_count.keys())[0].split("|")[:-1]

        board_moves: List[Move] = []
        for w_move, b_move in zip(white_moves_split, black_moves_split):
            uci_notation += w_move + " " + b_move + " "

            board_moves.append(Move.from_uci(w_move))
            board_moves.append(Move.from_uci(b_move))

        self.uci_opening = uci_notation[:-1]
        try:
            self.san_opening = temp_board.variation_san(board_moves)
        except ValueError as err:
            print(err)

    def __eq__(self, __value: object) -> bool:
        if __value is None:
            return False

        if not isinstance(__value, Opening):
            return False

        return __value.bitboard == self.bitboard


def get_draw_rate(
    opening: Opening, color: bool
) -> float:  # color allows func to be run
    # with same arguments as other evaluatory functions
    """Get the rate that an opening draws"""
    return opening.results_total[2] / opening.results_total[0]


def get_pawn_structure(opening: Opening, color: bool) -> float:
    """Get the pawn structure for the given side"""
    return opening.bitboard.eval_pawn_structure(color)


def get_centre_prev(opening: Opening, color: bool) -> float:
    """Get the value for the piece prevalence in the centre"""
    return opening.bitboard.eval_center_prevailence(color)


def get_left_side_prev(opening: Opening, color: bool) -> float:
    """Get the value for the prevalence of the pieces on the left side of the board"""
    return opening.bitboard.eval_side_prevalence(color, False)


def get_right_side_prev(opening: Opening, color: bool) -> float:
    """Get the value for the prevalence of the pieces on the right side of the board"""
    return opening.bitboard.eval_side_prevalence(color, True)


def get_developed_pieces(opening: Opening, color: bool) -> float:
    """Get the total value of the pieces that are not on the back rank"""
    return opening.bitboard.eval_developed_pieces(color)


def check_opening_in_list(check_opening: Opening, opening_list: List[Opening]) -> bool:
    """Check if an opening appears in a list of openings"""
    appearence: bool = False
    for opening in opening_list:
        if opening == check_opening:
            appearence = True
            break

    return appearence


def _get_outcome(result: str):
    if result == "1-0":
        out_position = 1
    elif result == "1/2-1/2":
        out_position = 2
    else:
        out_position = 3

    return out_position


# Clustering Values configurations
AXIS_NAMES = [  # "Rate of Drawing Game",
    "Pawn Structure Evaluation",
    "Points of Material on Centre Squares",
    "Left Side Material Worth",
    "Right Side Material Worth",
    "Developed Piece Material Worth",
]

AXIS_NAMES_SHORT = ["Pawn", "Centre", "Left", "Right", "Develop"]  # "Draw",

AXIS_FUNCTIONS = [  # get_draw_rate,
    get_pawn_structure,
    get_centre_prev,
    get_left_side_prev,
    get_right_side_prev,
    get_developed_pieces,
]
