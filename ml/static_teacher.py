"""Stockfish debug-eval labels: current-position static score, not search value.

SF19 evaluate.cpp trace() reports Final evaluation in White-perspective pawn
units after to_cp normalization, with zero optimism. The two printed decimals
give integer cp precision. Its in-check trace has no static score.
"""
import re
from decimal import Decimal
import chess
import chess.engine

_FINAL = re.compile(r'^Final evaluation\s+([+-]?\d+\.\d{2})\s+\(white side\)')


def parse_static_line(line):
    match = _FINAL.match(line.strip())
    return int(Decimal(match.group(1))*100) if match else None


def static_evaluation(engine, board):
    if board.is_check():
        raise ValueError('Stockfish has no stand-pat score in check')
    class StaticCommand(chess.engine.BaseCommand):
        def start(self):
            self.value = None; self.final_line = None
            self._engine._position(board)
            self._engine.send_line('eval')
            self._engine.send_line('isready')

        def line_received(self, line):
            value = parse_static_line(line)
            if value is not None:
                if self.value is not None:
                    raise chess.engine.EngineError('Multiple final static evaluations')
                self.value = value; self.final_line = line
            if line.strip() == 'readyok':
                if self.value is None:
                    raise chess.engine.EngineError('No White-perspective Final evaluation in Stockfish output')
                self.result.set_result({'score_cp':self.value, 'teacher_final_line':self.final_line,
                                        'label_kind':'stockfish_static_white_cp'})
                self.set_finished()
    return engine.communicate(StaticCommand)
