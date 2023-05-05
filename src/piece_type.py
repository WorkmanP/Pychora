"""A file containing all of the low level classes for the chess opening recomender project"""
from enum import Enum

from chess import BISHOP, BLACK, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE


class PieceType(Enum):
    """Chess piece type containing the specific piece and the color"""
    PAWN_W = (PAWN, WHITE)
    PAWN_B = (PAWN, BLACK)
    ROOK_W = (ROOK, WHITE)
    ROOK_B = (ROOK, BLACK)
    KNIGHT_W = (KNIGHT, WHITE)
    KNIGHT_B = (KNIGHT, BLACK)
    BISHOP_W = (BISHOP, WHITE)
    BISHOP_B = (BISHOP, BLACK)
    QUEEN_W = (QUEEN, WHITE)
    QUEEN_B = (QUEEN, BLACK)
    KING_W = (KING, WHITE)
    KING_B = (KING, BLACK)
