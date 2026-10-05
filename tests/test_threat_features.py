import random
import tempfile
import unittest
from pathlib import Path
import chess
import numpy as np
import torch
from ml.threat_model import (ThreatNet, board_to_threat_array, mirror_array,
                            pin_aware_attackers, threat_features)
from ml.threat_hybrid import ThreatHybrid
from ml.run_threat_v31 import metadata
from ml.run_search_correction_v29 import quiet
from ml.model import SCORE_SCALE


class ThreatFeatureTests(unittest.TestCase):
    def test_runtime_matches_torch_and_preserves_boards_through_gates(self):
        torch.manual_seed(3131)
        model = ThreatNet().eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.pt'
            torch.save(metadata(model.state_dict()), path)
            evaluator = ThreatHybrid(path)
            positions = [chess.Board(), chess.Board('8/5k2/8/8/3N4/8/3P1K2/8 w - - 0 1'),
                         chess.Board('6k1/3P4/8/8/8/8/5K2/8 w - - 0 1'),
                         chess.Board('4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1')]
            for original in positions:
                for b in (original, original.mirror()):
                    fen = b.fen()
                    with torch.inference_mode():
                        correction = model(torch.from_numpy(board_to_threat_array(b))).item()
                    expected = round(evaluator.baseline(b) + (.25 if quiet(b) else 0)*correction*SCORE_SCALE)
                    self.assertLessEqual(abs(expected-evaluator.evaluate_position(b)), 1)
                    self.assertEqual(evaluator.evaluate_position(b), -evaluator.evaluate_position(b.mirror()))
                    self.assertEqual(b.fen(), fen)

    def test_rejects_runtime_blend_different_from_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.pt'
            torch.save(metadata(ThreatNet().state_dict()), path)
            with self.assertRaises(ValueError):
                ThreatHybrid(path, .1)

    def test_pinned_knight_is_not_an_effective_defender(self):
        b = chess.Board('4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1')
        self.assertTrue(b.is_pinned(chess.WHITE, chess.E2))
        self.assertTrue(b.attackers_mask(chess.WHITE, chess.F4))
        self.assertFalse(pin_aware_attackers(b, chess.WHITE, chess.F4))

    def test_pinned_rook_keeps_attacks_along_pin(self):
        b = chess.Board('4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1')
        self.assertTrue(pin_aware_attackers(b, chess.WHITE, chess.E8))
        self.assertFalse(pin_aware_attackers(b, chess.WHITE, chess.F2) & chess.BB_E2)

    def test_attack_by_cheaper_piece_and_pin_have_spatial_locations(self):
        b = chess.Board('4r1k1/8/8/8/3p4/2Q5/4N3/4K3 w - - 0 1')
        p = threat_features(b)[:512].reshape(2, 4, 64)
        self.assertEqual(p[0, 3, chess.C3], 1)
        self.assertEqual(p[0, 2, chess.E2], 1)
        self.assertEqual(p[0, 2, chess.D2], 0)

    def test_mirror_encoding_and_network_symmetry_across_legal_play(self):
        rng = random.Random(310031)
        torch.manual_seed(31)
        model = ThreatNet().eval()
        b = chess.Board()
        for _ in range(60):
            before = b.fen()
            x, y = board_to_threat_array(b), board_to_threat_array(b.mirror())
            np.testing.assert_array_equal(mirror_array(x), y)
            with torch.inference_mode():
                self.assertAlmostEqual(model(torch.from_numpy(x)).item(),
                                       -model(torch.from_numpy(y)).item(), places=6)
            self.assertEqual(before, b.fen())
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
