"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
import numpy as np
from tensorflow.keras.models import load_model
import gomoku as gm


def myfun(state):
    model = load_model('policies/models/20231209_022154.h5')
    actions = state.valid_actions()
    input = state.board.copy()
    inputE = state.board[gm.EMPTY]
    inputO = state.board[gm.MIN]
    inputX = state.board[gm.MAX]
    inp = np.zeros((15, 15),dtype=int)
    inp[inputO == 1] = -1
    inp[inputX == 1] = 1
    # input[(input != 1) & (input != 0)] = -1
    # input[(input == 1) & (input != 0)] = 1
    inp = np.expand_dims(inp, axis=(0, -1)).astype(np.float32)

    tup = calculate(inp,model)
    reps =0
    while( tup not in actions and reps <10):
        print("Duplicate Problem")
        #tup = calculate(inp,model)
        tup = abc(state)
        reps +=1
    return tup
def calculate(inp,model):
    output = model.predict(inp).squeeze()
    output = output.reshape((15, 15))
    return np.unravel_index(np.argmax(output), output.shape)

def abc(state):
    rank = -state.corr[:, 1:].sum(axis=(0, 1)) - np.random.rand(*state.board.shape[1:])
    rank = rank[state.board[gm.EMPTY] > 0]  # only empty positions are valid actions
    scrambler = np.argsort(rank)

class Submission:
    def __init__(self, board_size, win_size):
        self.win_size = win_size
        self.board_size = board_size

    def __call__(self, state):

        action = myfun(state)
        return action
