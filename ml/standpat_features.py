"""Exact threat features from shared attack masks; no model/feature changes."""
import chess
import numpy as np
from engine.evaluation import MATERIAL


def threat_features_with_attacks(board):
    effective = {c:{pt:[] for pt in chess.PIECE_TYPES} for c in chess.COLORS}
    count_masks = {}; pseudo = {}; pins = {}
    for color in (chess.WHITE,chess.BLACK):
        once = twice = three = combined = 0
        for square in chess.scan_forward(board.occupied_co[color]):
            attack = board.attacks_mask(square); combined |= attack
            pin = board.pin_mask(color,square); pins[square] = pin != chess.BB_ALL
            mask = attack & pin
            effective[color][board.piece_type_at(square)].append(mask)
            three |= twice & mask; twice |= once & mask; once |= mask
        count_masks[color] = (once,twice,three); pseudo[color] = combined
    planes = np.zeros((2,4,64),dtype=np.float32); kings = np.zeros((2,8),dtype=np.float32)
    for ci,color in enumerate((chess.WHITE,chess.BLACK)):
        cheaper = {}
        for victim in chess.PIECE_TYPES:
            mask = 0
            for pt,masks in effective[not color].items():
                if MATERIAL.get(pt,20000) < MATERIAL.get(victim,20000):
                    for attack in masks:
                        mask |= attack
            cheaper[victim] = mask
        for square in chess.scan_forward(board.occupied_co[color]):
            bit = chess.BB_SQUARES[square]
            planes[ci,0,square] = sum(bool(mask & bit) for mask in count_masks[not color])/3
            planes[ci,1,square] = sum(bool(mask & bit) for mask in count_masks[color])/3
            planes[ci,2,square] = float(pins[square])
            planes[ci,3,square] = float(bool(cheaper[board.piece_type_at(square)] & bit))
        king = board.king(color)
        if king is None:
            continue
        undefended = 0
        for target in chess.scan_forward(chess.BB_KING_ATTACKS[king]):
            bit = chess.BB_SQUARES[target]
            for pt in chess.PIECE_TYPES:
                # Preserve the reference float32 accumulation order exactly.
                kings[ci,pt-1] += sum(bool(mask & bit) for mask in effective[not color][pt])/24
            undefended += bool(count_masks[not color][0]&bit) and not bool(count_masks[color][0]&bit)
        kings[ci,6] = float(bool(count_masks[not color][0]&chess.BB_SQUARES[king]))
        kings[ci,7] = undefended/8
    return np.concatenate((planes.reshape(-1),kings.reshape(-1))),pseudo
