import tempfile
import unittest
from pathlib import Path
import chess
import torch
from engine.strategic import PRIORS, StrategicEvaluator
from engine.evaluation import MATE_SCORE
from ml.model import ChessNet, RELATIONAL_INPUT_SIZE, RELATIONAL_MODEL_VERSION, SCORE_SCALE, board_to_tensor
from ml.strategic_hybrid import StrategicHybrid


class StrategicHybridTests(unittest.TestCase):
    def checkpoint(self,path,baseline='strategic-v26'):
        model=ChessNet(RELATIONAL_INPUT_SIZE,250,True,(64,32))
        for parameter in model.parameters():
            parameter.data.zero_()
        model.net[4].bias.data.fill_(.2)
        torch.save({'version':RELATIONAL_MODEL_VERSION,'input_size':RELATIONAL_INPUT_SIZE,
                    'score_scale':SCORE_SCALE,'target_mode':'residual','hidden_sizes':[64,32],
                    'correction_limit_cp':250,'color_consistent':True,'baseline_id':baseline,
                    'baseline_weights':list(PRIORS),'training_correction_weight':.25,
                    'training_quiet_only':True,'state_dict':model.state_dict()},path)
        return model

    def test_runtime_matches_training_blend_and_quiet_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';model=self.checkpoint(path);hybrid=StrategicHybrid(path,.25)
            baseline=StrategicEvaluator(PRIORS)
            for fen in (chess.STARTING_FEN,'7k/8/P7/8/8/8/8/7K b - - 0 1'):
                b=chess.Board(fen)
                with torch.inference_mode():
                    residual=model(board_to_tensor(b,RELATIONAL_INPUT_SIZE).unsqueeze(0)).item()*SCORE_SCALE
                self.assertEqual(hybrid.evaluate_position(b),round(baseline(b)+.25*residual))
                self.assertEqual(hybrid(b),-hybrid(b.mirror()))
            for fen in ('7k/8/8/8/8/8/4r3/4K3 w - - 0 1',
                        '7k/8/8/8/8/3p4/2P5/K7 w - - 0 1'):
                b=chess.Board(fen)
                self.assertEqual(hybrid.evaluate_position(b),baseline(b))
            self.assertEqual(hybrid(chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')),MATE_SCORE)

    def test_rejects_wrong_baseline_or_runtime_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';self.checkpoint(path,'original')
            with self.assertRaises(ValueError):StrategicHybrid(path,.25)
            self.checkpoint(path)
            with self.assertRaises(ValueError):StrategicHybrid(path,.1)
