"""File containing the Bitboard class and relevent functions to allow FEN conversion
to bitboard, comparison between chess positions, and evaluation for the chess positions"""
from __future__ import annotations

from typing import Dict, List

from chess import (BB_A1, BB_A8, BB_H1, BB_H8, BISHOP, BLACK, KNIGHT, PAWN,
                   QUEEN, ROOK, WHITE, Board)
from piece_type import PieceType


class FullBitboard():
    """A data structure representing the state of the board
    at a given time further reading into bitboards is found
    here: https://www.chessprogramming.org/Bitboards"""

    def __init__(self) -> None:
        self.piece_bitboards: Dict[PieceType, int] = {}

        # Represented through integer 0-3 showing which direction each king can castle
        # b01: can only castle right. b10: can only castle left. b11 can castle both ways.
        # b00 can't castle
        self.black_castling: int = 0
        self.white_castling: int = 0

    def create_from_board(self, board_state: Board) -> None:
        """Create a bitboard from a board state type within python_chess"""
        self._create_piece_bitboards(board_state)
        self._get_meta_attributes(board_state)

    def _create_piece_bitboards(self, board_state: Board) -> None:
        """Produce a dictionary of 8x8 bitboards which state the positions of pieces"""
        self.piece_bitboards = {piece_type: int(board_state.pieces(
            *piece_type.value)) for piece_type in PieceType}

    def _get_meta_attributes(self, board_state: Board) -> None:
        """Add variables to the class representing castling, and other information
        not represented through the bitboard"""
        self.white_castling = (2 if bool(board_state.castling_rights & BB_A1) else 0 +
                               1 if bool(board_state.castling_rights & BB_H1) else 0)

        self.black_castling = (2 if bool(board_state.castling_rights & BB_A8) else 0 +
                               1 if bool(board_state.castling_rights & BB_H8) else 0)

    # Returning bitboard information with the following methods

    def get_one_piece_bitboard(self, piece_type: PieceType) -> int:
        """Gets the bitboard for a specific piece"""
        return self.piece_bitboards[piece_type]

    def get_color_bitboards(self, color: bool):
        """Gets the bitbaord of all pieces for a specific color"""
        color_dict: Dict[int, int] = {}
        for piece_type in PieceType:
            if piece_type.value[1] == color:
                color_dict[piece_type.value[0]] = self.piece_bitboards[piece_type]
                # Returning color information for consistency with bitboard types

        return color_dict

    def get_all_piece_bitboard(self) -> Dict[PieceType, int]:
        """Returns all of pieces bitboards in one dictionary"""
        return self.piece_bitboards

    def color_equals(self, other: FullBitboard, color: bool) -> bool:
        """Determines whether the given colors state is the same as another bitboard

        @params
        other: Bitboard of another opening
        color: The color of another opening to check against, False = White, True = Black

        """
        if color:
            if other.white_castling != self.white_castling:
                return False

        else:
            if other.black_castling != self.black_castling:
                return False

        for piece_type in PieceType:
            if piece_type.value[1] == color:
                if self.piece_bitboards[piece_type] != other.piece_bitboards[piece_type]:
                    return False

        return True

    # Processing bitboard to and from strings and files

    def bb_to_store_string(self) -> str:
        """Convert the bitboard to a string to encode the information in a 
        small and concise string for storing information"""

        outstring: str = ""

        for piece_type in PieceType:
            board_value: int = self.piece_bitboards[piece_type]
            piece_str_encoding: str = str(hex(board_value))[2:]

            outstring += piece_str_encoding
            outstring += "/"
        outstring += str(self.white_castling)
        outstring += str(self.black_castling)

        return outstring

    def store_string_to_bb(self, instring: str) -> None:
        """Process a stored bitboard string and attribute the information to this bitboard
        object."""
        char_index: int = 0
        split_instring: List[str] = instring.split("/")
        for piece_type in PieceType:
            self.piece_bitboards[piece_type] = int(split_instring[char_index], 16)
            char_index += 1

        self.white_castling = int(str(split_instring[char_index])[0])
        self.black_castling = int(str(split_instring[char_index])[1])

    # Evaluate each bitboard with the following methods

    def eval_total_piece_value(self, color: bool) -> float:
        """Evaluate the overall static worth of the pieces for a player"""
        total_value: float = 0
        for piece, bitboard in self.get_color_bitboards(color).items():
            # Can be replaced by a switch statement for Python 3.10+
            total_value += get_piece_bb_total(piece, bitboard)
        return total_value

    def eval_side_prevalence(self, color: bool, side: bool) -> float:
        """Evaluate the presence of pieces in one side of the board
        the outermost files are worth 1x, second most, 0.75x and third most 0.5x
        True is left right side of the board"""
        default_file = make_default_file() - 2**56 - 1

        if side:
            edgemost_bb = default_file
            second_bb = default_file << 1
            thrid_bb = default_file << 2

        else:
            edgemost_bb = default_file << 7
            second_bb = default_file << 6
            thrid_bb = default_file << 5

        side_value: float = 0.0

        for piece, bitboard in self.get_color_bitboards(color).items():
            if piece == PAWN:
                continue

            side_value += 1 * get_piece_bb_total(piece, (bitboard & edgemost_bb))
            side_value += 0.75 * get_piece_bb_total(piece, (bitboard & second_bb))
            side_value += 0.5 * get_piece_bb_total(piece, (bitboard & thrid_bb))

        return side_value

    def eval_developed_pieces(self, color: bool) -> float:
        """Get the value of all pieces that have moved from the back ranks 
        and are still on the board"""
        developed_value: float = 0
        developed_squares = (
            255 << 8) | (
            255 << 16) | (
            255 << 24) | (
            255 << 32) | (
            255 << 40) | (
            255 << 48)

        for piece, bitboard in self.get_color_bitboards(color).items():
            if piece == PAWN:
                continue
            developed_value += get_piece_bb_total(piece, (bitboard & developed_squares))

        return developed_value

    def eval_center_prevailence(self, color: bool) -> float:
        """Evaluate the total amount of pieces can attack each of the centre squares"""
        default_rank: int = 255
        default_file = make_default_file()

        # Const Values, should be moved to decrease unnecessary processing
        rank_centre = (
            default_rank << 16) | (
            default_rank << 24) | (
            default_rank << 32) | (
            default_rank << 40)
        file_centre = (
            default_file << 2) | (
            default_file << 3) | (
            default_file << 4) | (
            default_file << 5)

        centre_squares = rank_centre & file_centre

        # Actual Vars
        centre_value = 0
        for piece, bitboard in self.get_color_bitboards(color).items():
            centre_value += get_piece_bb_total(piece, (bitboard & centre_squares))

        return centre_value

    def eval_pawn_structure(self, color: bool) -> float:
        """Evaluate pawn structure for the given color. Algorithm derived from
        stockfish pawn strength evaluation found at 
        https://github.com/official-stockfish/Stockfish/blob/master/src/pawns.cpp"""

        score: int = 0

        pawn_bb = self.get_color_bitboards(color)[PAWN]
        our_pawn_bb = self.get_color_bitboards(color)[PAWN]

        # Values taken from stockfish S(mg)
        doubled_value: int = 11
        isolated_value: int = 9
        default_pawn_value = 15

        if (color == WHITE):
            connected_value: List[int] = [0, 0, 0, 3, 7, 7, 15, 54, 86]
        else:
            connected_value: List[int] = [0, 86, 54, 15, 7, 7, 3, 0]

        while pawn_bb:
            # Find rank and file or current pawn
            f_piece_pos = find_first_piece_pos(pawn_bb)

            next_pawn_rank = find_piece_rank(f_piece_pos)
            next_pawn_file = find_piece_file(f_piece_pos)

            bitboard_rank = 255 << (8*(next_pawn_rank-1))
            # Create default rank and shift to correct rank

            bitboard_file = make_default_file()
            # Create default file (H)
            bitboard_file = bitboard_file << (next_pawn_file)
            # Shift to correct file

            # Remove pawn from pawn bb, prevents comparison against itself for doubled
            pawn_bb -= (bitboard_rank & bitboard_file)

            # Set flags
            if (color):
                supported: bool = bool(our_pawn_bb
                                       & get_rank_behind(bitboard_rank)
                                       & get_file_neighbors(bitboard_file))
            else:
                supported: bool = bool(our_pawn_bb
                                       & get_rank_front(bitboard_rank)
                                       & get_file_neighbors(bitboard_file))

            phalanx: bool = bool(our_pawn_bb
                                 & bitboard_rank
                                 & get_file_neighbors(bitboard_file))

            doubled: bool = pawn_bb & bitboard_file
            isolated: bool = not our_pawn_bb & get_file_neighbors(bitboard_file)

            if phalanx | supported:
                score += connected_value[next_pawn_rank] * (
                    2 + bool(phalanx) + 22 * bool(supported))

            elif isolated:
                score -= isolated_value

            if doubled:
                score -= doubled_value

            # Add Score for each Pawn
            score += default_pawn_value

        return score / 450

    # Dunder methods

    def __eq__(self, other: object) -> bool:
        """Override default equality function"""
        if not isinstance(other, FullBitboard):
            return False
        return (self.color_equals(other, WHITE) and
                self.color_equals(other, BLACK)
                )


def find_first_piece_pos(piece_bb: int):
    """Get the position in the bitboard of the first piece in that bitboard. Order:
    (A8 = 64  > ... > H8 > A7 > ... > A1 > ... > H1 = 1)"""
    if piece_bb < 1:
        return -1

    bit_string: str = str(bin(piece_bb))[2:]  # Convert to binary and remove leading 0b
    pos = len(bit_string)  # Leading 0s are removed by default in bin conv. Pos is remaining length

    return pos


def find_piece_rank(piece_pos: int):
    """Find the row of the position of a piece on a bitboard. 1 = 1, 8 = 8"""
    row: int = ((piece_pos - 1) // 8) + 1
    return row


def find_piece_file(piece_pos: int):
    """Find the file of the position of a piece on a bitboard A = 7, H = 0 ()"""
    file: int = (piece_pos - 1) % 8
    return file


def get_rank_front(rank: int):
    """Get the rank in front of the input rank"""
    if rank == (255 << (8*7)):
        return 0
    return rank << (8)


def get_rank_behind(rank: int):
    """Get the behind in front of the input rank"""
    if rank == (255):
        return 0
    return rank >> 8


def get_file_neighbors(file: int):
    """Get the neighbouring files to an input file"""
    if file % 256 == 1:
        return file << 1
    if file % 256 == 0:
        return file >> 1
    return file << 1 | file >> 1


def make_default_file():
    """Create a bitboard for the file H"""
    return 2**56 + 2**48 + 2 ** 40 + 2 ** 32 + 2 ** 24 + 2 ** 16 + 2**8 + 1


def get_piece_bb_total(piece: int, bitboard: int) -> float:
    """Evaluate points of the total ammount pieces on a bitboard"""
    side_value: float = 0.0
    if piece == PAWN:
        piece_value = 1.0
    elif piece == ROOK:
        piece_value = 5.63
    elif piece == KNIGHT:
        piece_value = 3.05
    elif piece == BISHOP:
        piece_value = 3.33
    elif piece == QUEEN:
        piece_value = 9.5
    else:
        # Since the kings will always be on the board, the value for the piece is 0
        piece_value = 0.0

    # Count the number of pieces and times by the piece value
    side_value += piece_value * bin(bitboard).count("1")

    return side_value
