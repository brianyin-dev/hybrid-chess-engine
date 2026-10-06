import tempfile,unittest
from pathlib import Path
import chess,torch
from ml.calibrated_hybrid import CalibratedNet,CalibratedHybrid,metadata
from ml.model import SCORE_SCALE
from ml.threat_model import board_to_threat_array
from ml.run_evaluator_v34 import feasible_target

class CalibratedTests(unittest.TestCase):
    def test_hidden_features_frozen_and_margin_feasible(self):
        m=CalibratedNet()
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),33)
        before={k:v.clone() for k,v in m.state_dict().items() if not k.startswith('net.4.')}
        opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=.01)
        m(torch.from_numpy(board_to_threat_array(chess.Board()))).square().backward();opt.step()
        for k,v in before.items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
        refs=torch.tensor([-47.,-40.,-5.,0.]);target=feasible_target(refs)
        self.assertTrue(torch.all(target<=refs+48))
        self.assertTrue(torch.all(target>0))
    def test_bounds_symmetry_exact_zero_and_terminal_gate(self):
        boards=[chess.Board(),chess.Board('8/8/1R6/4k3/6K1/8/4p3/8 b - - 0 67'),chess.Board('4k3/8/8/8/8/8/4r3/4K3 w - - 0 1')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.pt';model=CalibratedNet()
            for p in model.parameters():torch.nn.init.zeros_(p)
            with torch.no_grad():model.net[4].bias.fill_(5.)
            torch.save(metadata(model.state_dict()),path);adapter=CalibratedHybrid(path)
            for b in boards:
                before=b.fen();actual=adapter.evaluate_position(b);base=adapter.baseline.evaluate_position(b)
                self.assertLessEqual(abs(actual-base),25)
                self.assertEqual(actual,-adapter.evaluate_position(b.mirror()))
                if b.is_check():self.assertEqual(actual,base)
                else:
                    with torch.inference_mode():expected=round(base+.25*model(torch.from_numpy(board_to_threat_array(b))).item()*SCORE_SCALE)
                    self.assertLessEqual(abs(actual-expected),1)
                self.assertEqual(before,b.fen())
            with torch.no_grad():model.net[4].bias.zero_()
            torch.save(metadata(model.state_dict()),path);zero=CalibratedHybrid(path)
            for b in boards:self.assertEqual(zero(b),zero.baseline(b))
            terminal=chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
            from engine.app_evaluation import evaluate
            self.assertTrue(terminal.is_checkmate());self.assertEqual(zero(terminal),evaluate(terminal))
    def test_rejects_wrong_bound_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.pt'
            for field,value in [('final_correction_bound_cp',63),('training_gate','quiet'),('frozen_hidden_layers',False)]:
                saved=metadata(CalibratedNet().state_dict());saved[field]=value;torch.save(saved,path)
                with self.assertRaises(ValueError):CalibratedHybrid(path)
