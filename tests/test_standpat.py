import tempfile
import random
import unittest
from pathlib import Path
import chess
import torch
import numpy as np
from ml.static_teacher import parse_static_line
from ml.standpat_hybrid import StandPatHybrid, GATE_ID, gate_active
from ml.threat_hybrid import ThreatHybrid
from ml.threat_model import ThreatNet, board_to_threat_array, threat_features
from ml.standpat_features import threat_features_with_attacks
from ml.run_threat_v31 import metadata
from ml.model import SCORE_SCALE
from ml.run_standpat_v32 import StaticSampler
from engine.search import _position_key, _Search, INF
from engine.strategic import PRIORS, StrategicEvaluator


class StandPatTests(unittest.TestCase):
    def test_shared_attack_encoder_is_exact_across_play_and_special_positions(self):
        rng=random.Random(320032); b=chess.Board()
        positions=[chess.Board('4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1'),
                   chess.Board('6k1/3P4/8/8/8/8/5K2/8 w - - 0 1')]
        for _ in range(150):
            positions.append(b.copy(stack=False))
            if b.is_game_over():
                b=chess.Board()
            else:
                b.push(rng.choice(list(b.legal_moves)))
        for original in positions:
            for board in (original,original.mirror()):
                actual,attacks=threat_features_with_attacks(board)
                np.testing.assert_array_equal(actual,threat_features(board))
                for color in chess.COLORS:
                    expected=0
                    for s in chess.scan_forward(board.occupied_co[color]):
                        expected|=board.attacks_mask(s)
                    self.assertEqual(attacks[color],expected)

    def test_sampler_preserves_search_repetition_state_and_tree(self):
        b=chess.Board()
        for _ in range(3):
            for uci in ('g1f3','g8f6','f3g1','f6g8'):
                b.push_uci(uci)
        sampler=StaticSampler(b,32)
        reference=_Search(b,StrategicEvaluator(PRIORS),None,True,2000)
        reference.use_lmr=True
        self.assertEqual(sampler.counts[_position_key(b)],4)
        self.assertEqual(sampler.negamax(b,1,-INF,INF,0),reference.negamax(b,1,-INF,INF,0))
        self.assertEqual(sampler.nodes,reference.nodes)
        self.assertEqual(sampler.root_candidate,reference.root_candidate)
        self.assertEqual(sampler.counts,reference.counts)

    def test_parses_final_white_score_not_raw_nnue(self):
        self.assertEqual(parse_static_line('Final evaluation      -1.37 (white side) [with scaled NNUE, ...]'),-137)
        self.assertEqual(parse_static_line('Final evaluation      +0.09 (white side)'),9)
        self.assertIsNone(parse_static_line('NNUE evaluation        +4.62 (white side)'))
        self.assertIsNone(parse_static_line('Final evaluation: none (in check)'))
        self.assertIsNone(parse_static_line('Final evaluation +1.00 (side to move)'))

    def test_capture_position_is_active_but_check_is_not(self):
        capture = chess.Board('6k1/8/8/8/3p4/2P5/8/6K1 w - - 0 1')
        self.assertTrue(next(capture.generate_legal_captures(),None))
        self.assertTrue(gate_active(capture))
        self.assertFalse(gate_active(chess.Board('4r1k1/8/8/8/8/8/8/4K3 w - - 0 1')))

    def test_runtime_matches_torch_on_capture_positions_and_respects_bound(self):
        torch.manual_seed(3232)
        model = ThreatNet().eval()
        saved = metadata(model.state_dict())
        saved.update(training_gate=GATE_ID,training_quiet_only=False,training_target='stockfish_static_white_cp')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.pt'; torch.save(saved,path)
            runtime = StandPatHybrid(path)
            for original in (chess.Board('6k1/8/8/8/3p4/2P5/8/6K1 w - - 0 1'),
                             chess.Board('4r1k1/8/8/8/8/8/8/4K3 w - - 0 1')):
                for b in (original,original.mirror()):
                    fen = b.fen(); base = runtime.baseline(b)
                    with torch.inference_mode():
                        delta = .25*model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE
                    expected = round(base+(delta if gate_active(b) else 0))
                    actual = runtime.evaluate_position(b)
                    self.assertLessEqual(abs(expected-actual),1)
                    self.assertLessEqual(abs(actual-base),63)
                    self.assertEqual(actual,-runtime.evaluate_position(b.mirror()))
                    self.assertEqual(fen,b.fen())
            with self.assertRaises(ValueError):
                ThreatHybrid(path)
            saved['training_gate']='quiet_v1'; torch.save(saved,path)
            with self.assertRaises(ValueError):
                StandPatHybrid(path)
