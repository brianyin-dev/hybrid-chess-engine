"""Experimental square-specific relationships; legacy checkpoints stay unchanged."""
import chess
import numpy as np
import torch
from ml.model import ChessNet, RELATIONAL_INPUT_SIZE, board_to_array
from engine.evaluation import MATERIAL

THREAT_INPUT_SIZE = RELATIONAL_INPUT_SIZE + 512 + 16
THREAT_VERSION = 5


def pin_aware_attackers(board, color, target):
    """Pseudo attackers constrained by absolute pins, not full capture legality.

    Kings retain geometric attacks. En passant and discovered king exposure are
    not simulated here; these are features, not legal-move or SEE substitutes.
    """
    return sum(chess.BB_SQUARES[square]
               for square in chess.scan_forward(board.attackers_mask(color, target))
               if board.pin_mask(color, square) & chess.BB_SQUARES[target])


def threat_features(board):
    planes = np.zeros((2, 4, 64), dtype=np.float32)
    kings = np.zeros((2, 8), dtype=np.float32)
    for ci, color in enumerate((chess.WHITE, chess.BLACK)):
        for square in chess.scan_forward(board.occupied_co[color]):
            attackers = pin_aware_attackers(board, not color, square)
            defenders = pin_aware_attackers(board, color, square)
            planes[ci, 0, square] = min(attackers.bit_count(), 3) / 3
            planes[ci, 1, square] = min(defenders.bit_count(), 3) / 3
            planes[ci, 2, square] = float(board.is_pinned(color, square))
            victim = board.piece_type_at(square)
            planes[ci, 3, square] = float(any(
                MATERIAL.get(board.piece_type_at(a), 20000) < MATERIAL.get(victim, 20000)
                for a in chess.scan_forward(attackers)))
        king = board.king(color)
        if king is None:
            continue
        ring = chess.BB_KING_ATTACKS[king]
        undefended = 0
        for target in chess.scan_forward(ring):
            attackers = pin_aware_attackers(board, not color, target)
            for pt in chess.PIECE_TYPES:
                kings[ci, pt - 1] += (attackers & board.pieces_mask(pt, not color)).bit_count() / 24
            undefended += bool(attackers) and not bool(pin_aware_attackers(board, color, target))
        kings[ci, 6] = float(bool(pin_aware_attackers(board, not color, king)))
        kings[ci, 7] = undefended / 8
    return np.concatenate((planes.reshape(-1), kings.reshape(-1)))


def board_to_threat_array(board):
    return np.concatenate((board_to_array(board, RELATIONAL_INPUT_SIZE), threat_features(board)))


def mirror_array(values):
    """Match python-chess Board.mirror without building another board."""
    out = values.copy()
    out[:768] = values[:768].reshape(2, 6, 8, 8)[::-1, :, ::-1, :].reshape(-1)
    out[768] = 1 - values[768]
    out[769:773] = values[[771, 772, 769, 770]]
    out[782:792] = values[[787, 788, 789, 790, 791, 782, 783, 784, 785, 786]]
    out[792] = -values[792]
    out[794:892] = values[794:892].reshape(2, 49)[::-1].reshape(-1)
    out[892:1404] = values[892:1404].reshape(2, 4, 8, 8)[::-1, :, ::-1, :].reshape(-1)
    out[1404:] = values[1404:].reshape(2, 8)[::-1].reshape(-1)
    return out


class ThreatNet(ChessNet):
    def __init__(self):
        super().__init__(THREAT_INPUT_SIZE, 250, False, (64, 32))

    def forward(self, x):
        mirrored = x.clone()
        mirrored[..., :768] = x[..., :768].reshape(*x.shape[:-1], 2, 6, 8, 8).flip((-4, -2)).reshape(*x.shape[:-1], 768)
        mirrored[..., 768] = 1 - x[..., 768]
        mirrored[..., 769:773] = x[..., [771, 772, 769, 770]]
        mirrored[..., 782:792] = x[..., [787, 788, 789, 790, 791, 782, 783, 784, 785, 786]]
        mirrored[..., 792] = -x[..., 792]
        mirrored[..., 794:892] = x[..., 794:892].reshape(*x.shape[:-1], 2, 49).flip(-2).reshape(*x.shape[:-1], 98)
        mirrored[..., 892:1404] = x[..., 892:1404].reshape(*x.shape[:-1], 2, 4, 8, 8).flip((-4, -2)).reshape(*x.shape[:-1], 512)
        mirrored[..., 1404:] = x[..., 1404:].reshape(*x.shape[:-1], 2, 8).flip(-2).reshape(*x.shape[:-1], 16)
        white = x[..., 768] > .5
        return super().forward(torch.where(white.unsqueeze(-1), x, mirrored)) * torch.where(white, 1., -1.)
