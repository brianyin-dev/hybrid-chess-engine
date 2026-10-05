import tempfile,unittest
from pathlib import Path
import chess,numpy as np,torch
from ml.decision_hybrid import DecisionNet,DecisionHybrid,metadata
from ml.threat_model import board_to_threat_array
from ml.model import SCORE_SCALE

class DecisionTests(unittest.TestCase):
    def test_output_rule_and_bounds(self):
        x=torch.from_numpy(board_to_threat_array(chess.Board()))
        for mode in ('tanh','linear_clip'):
            model=DecisionNet(mode)
            for p in model.parameters():torch.nn.init.zeros_(p)
            with torch.no_grad():model.net[4].bias.fill_(.3)
            expected=.625*np.tanh(.3/.625) if mode=='tanh' else .3
            self.assertAlmostEqual(model(x).item(),expected,places=6)
            with torch.no_grad():model.net[4].bias.fill_(5.)
            self.assertLessEqual(abs(model(x).item()*SCORE_SCALE),250)
    def test_runtime_training_parity_symmetry_and_zero(self):
        torch.manual_seed(33)
        boards=[chess.Board(),chess.Board('8/8/1R6/4k3/6K1/8/4p3/8 b - - 0 67'),
                chess.Board('r3rk2/ppp2ppp/8/3p1b2/1P3P2/4B1P1/PPP1K1P1/R2B4 b - - 0 17')]
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('tanh','linear_clip'):
                model=DecisionNet(mode);path=Path(tmp)/'model.pt'
                torch.save(metadata(model.state_dict(),mode),path);adapter=DecisionHybrid(path)
                for b in boards:
                    before=b.fen()
                    with torch.inference_mode():expected=round(adapter.baseline(b)+.25*model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE)
                    actual=adapter.evaluate_position(b)
                    self.assertLessEqual(abs(actual-expected),1)
                    self.assertEqual(actual,-adapter.evaluate_position(b.mirror()))
                    self.assertLessEqual(abs(actual-adapter.baseline(b)),63)
                    self.assertEqual(before,b.fen())
                torch.nn.init.zeros_(model.net[4].weight);torch.nn.init.zeros_(model.net[4].bias)
                torch.save(metadata(model.state_dict(),mode),path);zero=DecisionHybrid(path)
                for b in boards:self.assertEqual(zero(b),zero.baseline(b))
    def test_metadata_rejects_wrong_target_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';saved=metadata(DecisionNet().state_dict(),'linear_clip')
            saved['training_target']='stockfish_static_white_cp';torch.save(saved,path)
            with self.assertRaises(ValueError):DecisionHybrid(path)
            saved=metadata(DecisionNet().state_dict(),'linear_clip');saved['output_mode']='unbounded';torch.save(saved,path)
            with self.assertRaises(ValueError):DecisionHybrid(path)
    def test_check_gate(self):
        b=chess.Board('4k3/8/8/8/8/8/4r3/4K3 w - - 0 1')
        self.assertTrue(b.is_check())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';m=DecisionNet()
            with torch.no_grad():m.net[4].bias.fill_(5.)
            torch.save(metadata(m.state_dict(),'linear_clip'),path);adapter=DecisionHybrid(path)
            self.assertEqual(adapter.evaluate_position(b),adapter.baseline.evaluate_position(b))

    def test_trace_leaf_preserves_history_and_score_perspective(self):
        from ml.run_decision_v33 import leaf
        from engine.strategic import StrategicEvaluator,PRIORS
        from benchmarks.build_strategy_v26 import reconstruct
        base=StrategicEvaluator(PRIORS)
        for opening,move in [([],"e2e4"),(["e2e4"],"e7e5")]:
            b=chess.Board()
            for u in opening:b.push_uci(u)
            root={"fen":b.fen(),"initial_fen":chess.STARTING_FEN,"history":opening,
                  "game_id":1,"family":"fixture","split":"train"}
            result=leaf(root,chess.Move.from_uci(move),base)
            self.assertIsNotNone(result)
            endpoint=reconstruct(result)
            self.assertEqual(endpoint.fen(),result["fen"])
            self.assertEqual(result["history"][:len(opening)],opening)
            self.assertEqual(result["history"][len(opening)],move)
            self.assertEqual(result["strategic_base_cp"],base(endpoint))
            self.assertFalse(endpoint.is_check())
            self.assertEqual(root["fen"],b.fen())
