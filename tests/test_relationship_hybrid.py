import tempfile,unittest
from pathlib import Path
import chess,numpy as np,torch
from ml.relationship_features import relationship_features,mirror_relationship,KING_FEATURES,PAWN_FEATURES,RELATIONSHIP_SIZE
from ml.relationship_hybrid import RelationshipNet,RelationshipHybrid,metadata,features,mirror_features,ARMS
from ml.model import SCORE_SCALE

class RelationshipTests(unittest.TestCase):
    def test_color_geometry_symmetry_and_translation(self):
        boards=[chess.Board(),chess.Board('6k1/8/8/8/4n3/2P5/4R3/1K6 w - - 0 1'),chess.Board('k2r4/8/8/2p5/3Q4/4P3/8/3K4 w - - 0 1')]
        for b in boards:
            x=relationship_features(b)
            self.assertEqual(x.shape,(RELATIONSHIP_SIZE,));self.assertTrue(np.isfinite(x).all())
            np.testing.assert_allclose(relationship_features(b.mirror()),mirror_relationship(x),atol=1e-7)
            for arm in ARMS:
                np.testing.assert_allclose(features(b.mirror(),arm),mirror_features(features(b,arm),arm),atol=1e-6)
        b=boards[1];shift=b.transform(chess.shift_right)
        np.testing.assert_allclose(relationship_features(b),relationship_features(shift),atol=1e-7)
    def test_promotion_distance_and_defender_values(self):
        b=chess.Board('7k/4P3/8/8/3K4/8/8/8 w - - 0 1')
        p=relationship_features(b)[KING_FEATURES:KING_FEATURES+PAWN_FEATURES].reshape(2,8,3)
        self.assertEqual(p[0,1,0],1/8);self.assertEqual(p[0,1,1],1/8)
        b=chess.Board('k2r4/8/8/2p5/3Q4/4P3/8/3K4 w - - 0 1')
        v=relationship_features(b)[KING_FEATURES+PAWN_FEATURES:].reshape(2,5,6)
        self.assertEqual(v[0,4,2],0);self.assertGreater(v[0,4,3],0);self.assertGreater(v[0,4,4],0);self.assertGreater(v[0,4,5],0)
        b.remove_piece_at(chess.E3);v=relationship_features(b)[KING_FEATURES+PAWN_FEATURES:].reshape(2,5,6)
        self.assertGreater(v[0,4,2],0);self.assertEqual(v[0,4,4],0)
    def test_numpy_training_parity_bounds_zero_and_terminal(self):
        boards=[chess.Board(),chess.Board('8/8/1R6/4k3/6K1/8/4p3/8 b - - 0 67'),chess.Board('4k3/8/8/8/8/8/4r3/4K3 w - - 0 1')]
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ARMS:
                torch.manual_seed(35);m=RelationshipNet(arm);path=Path(tmp)/f'{arm}.pt';torch.save(metadata(m.state_dict(),arm),path);adapter=RelationshipHybrid(path)
                self.assertTrue(all(p.requires_grad for p in m.parameters()))
                for b in boards:
                    before=b.fen();actual=adapter.evaluate_position(b);base=adapter.baseline.evaluate_position(b)
                    self.assertEqual(actual,-adapter.evaluate_position(b.mirror()));self.assertLessEqual(abs(actual-base),63)
                    if b.is_check():self.assertEqual(actual,base)
                    else:
                        with torch.inference_mode():expected=round(base+.25*m(torch.from_numpy(features(b,arm))).item()*SCORE_SCALE)
                        self.assertLessEqual(abs(actual-expected),1)
                    self.assertEqual(before,b.fen())
                torch.nn.init.zeros_(m.net[4].weight);torch.nn.init.zeros_(m.net[4].bias);torch.save(metadata(m.state_dict(),arm),path);zero=RelationshipHybrid(path)
                for b in boards:self.assertEqual(zero(b),zero.baseline(b))
                from engine.app_evaluation import evaluate
                terminal=chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
                self.assertEqual(zero(terminal),evaluate(terminal))
    def test_rejects_wrong_target_and_feature_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.pt'
            for field,value in [('input_size',1420),('training_target','old'),('training_gate','quiet')]:
                saved=metadata(RelationshipNet('relationships').state_dict(),'relationships');saved[field]=value;torch.save(saved,path)
                with self.assertRaises(ValueError):RelationshipHybrid(path)
    def test_batched_mirror_matches_reference(self):
        for arm in ARMS:
            m=RelationshipNet(arm);b=chess.Board();b.push_uci('e2e4')
            x=torch.from_numpy(np.stack([features(b,arm),features(b.mirror(),arm)]))
            transformed=x[...,m.mirror_index]*m.mirror_sign+m.mirror_offset
            np.testing.assert_allclose(transformed.numpy()[0],x.numpy()[1],atol=1e-6)
            with torch.inference_mode():y=m(x)
            self.assertAlmostEqual(y[0].item(),-y[1].item(),places=6)

    def test_new_collector_trace_keeps_history_and_white_score(self):
        from ml.collect_relationship_v35 import trace
        from benchmarks.build_strategy_v26 import reconstruct
        from engine.strategic import StrategicEvaluator,PRIORS
        base=StrategicEvaluator(PRIORS)
        for prefix,move in [([],"e2e4"),(["e2e4"],"e7e5")]:
            b=chess.Board()
            for u in prefix:b.push_uci(u)
            r={"fen":b.fen(),"initial_fen":chess.STARTING_FEN,"history":prefix,"game_id":1,"family":"fixture","split":"train"}
            leaf,samples=trace(r,chess.Move.from_uci(move),base,2)
            self.assertIsNotNone(leaf);lb=reconstruct(leaf)
            self.assertEqual(lb.fen(),leaf["fen"]);self.assertEqual(base.evaluate_position(lb),leaf["strategic_base_cp"])
            self.assertEqual(leaf["history"][:len(prefix)],prefix);self.assertEqual(leaf["history"][len(prefix)],move)
            self.assertLessEqual(len(samples),4);self.assertFalse(lb.is_check())
