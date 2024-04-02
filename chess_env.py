import gym
from gym import spaces
import chess
import numpy as np

class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(64 * 73)  # Considerando que hay 64 casillas de origen y 73 posibles casillas de destino en notación 0x88
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)  # Ejemplo simplificado

    def step(self, action):
    # Convertir acción a movimiento en el tablero de ajedrez
        try:
            move = self.action_to_move(action)
            print(f"Intentando movimiento: {move}")
            if move in self.board.legal_moves:
                self.board.push(move)
            else:
                # Manejo de un movimiento inválido sin terminar la partida
                print("Movimiento inválido. Selecciona otro movimiento.")
                return self.board_to_observation(), -1, False, {}  # Penalización y la partida continúa
        except ValueError as e:
            print(f"Error al ejecutar movimiento: {e}")
            return self.board_to_observation(), -1, False, {}  # Penalización y la partida continúa
        
        done = self.board.is_game_over()
        reward = self.calculate_reward()
        observation = self.board_to_observation()

        print(self.board)  # Imprime el tablero actual
        print(f"Juego terminado? {done}")
        
        return observation, reward, done, {}

    def reset(self):
        self.board = chess.Board()
        return self.board_to_observation()

    def render(self, mode='human'):
        print(self.board)

    def board_to_observation(self):
        observation = np.zeros((8, 8, 12), dtype=np.float32)
        piece_to_layer = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                layer_index = piece_to_layer[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
                observation[row, col, layer_index] = 1
        return observation

    def action_to_move(self, action):
        # Asegurarse de que la acción se pueda mapear a un movimiento legal
        try:
            legal_moves = list(self.board.legal_moves)
            move = legal_moves[action]
            return move
        except IndexError:
            # La acción no se pudo convertir a un movimiento legal
            raise ValueError("Action is invalid")

    def calculate_reward(self):
        if self.board.is_checkmate():
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.1
        else:
            return -0.01
        
    def get_game_result(self):
        if self.board.is_checkmate():
            return "Negro" if self.board.turn == chess.WHITE else "Blanco", self.board.fullmove_number
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_draw():
            return "Tablas", self.board.fullmove_number
        return "En juego", self.board.fullmove_number
    
    def seed(self, seed=None):
        return

