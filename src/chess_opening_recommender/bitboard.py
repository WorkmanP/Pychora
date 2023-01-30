"""File containing the Bitboard class and relevent functions to allow FEN conversion
to bitboard, comparison between chess positions, and evaluation for the chess positions"""
from typing import Dict

from chess import Board
from piece_type import PieceType


class Bitboard():
    """A data structure representing the state of the board
    at a given time further reading into bitboards is found
    here: https://www.chessprogramming.org/Bitboards"""

    def __init__(self, board_state: Board) -> None:
        self.piece_bitboards: Dict[PieceType, int]
        self.create_piece_bitboards(board_state)

    def create_piece_bitboards(self, board_state: Board) -> None:
        """Produce a dictionary of 8x8 bitboards which state the positions of pieces"""
        self.piece_bitboards = {piece_type: int(board_state.pieces(
            *piece_type.value)) for piece_type in PieceType}

    def get_one_piece_bitboard(self, piece_type: PieceType) -> int:
        """Gets the bitboard for a specific piece"""
        return self.piece_bitboards[piece_type]

    def get_color_bitboards(self, color: bool):
        """Gets the bitbaord of all pieces for a specific color"""
        color_dict: Dict[int, int] = {}
        for piece_type in PieceType:
            if piece_type.value[1] == color:
                color_dict[piece_type.value[0]] = self.piece_bitboards[piece_type]
                # No need to retain color information if only returning one color type

        return color_dict

    def get_all_piece_bitboard(self) -> Dict[PieceType, int]:
        """Returns all of pieces bitboards in one dictionary"""
        return self.piece_bitboards
