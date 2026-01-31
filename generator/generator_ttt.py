import random
import numpy as np
from numba import njit

# --- CORE GAME LOGIC (Numba-compiled) ---

@njit
def check_winner(board):
    """Returns: 1=X wins, 2=O wins, 0=no winner"""
    for i in range(3):
        if board[i,0] == board[i,1] == board[i,2] != 0: return board[i,0]
    for j in range(3):
        if board[0,j] == board[1,j] == board[2,j] != 0: return board[0,j]
    if board[0,0] == board[1,1] == board[2,2] != 0: return board[1,1]
    if board[0,2] == board[1,1] == board[2,0] != 0: return board[1,1]
    return 0

@njit
def is_full(board):
    for i in range(3):
        for j in range(3):
            if board[i,j] == 0: return False
    return True

@njit
def generate_game(seed, min_moves=3, max_moves=7):
    """Generate random valid game."""
    np.random.seed(seed)
    board = np.zeros((3,3), dtype=np.int32)
    moves = []
    player = 1

    num_moves = np.random.randint(min_moves, max_moves+1)

    for _ in range(num_moves):
        empty = []
        for i in range(3):
            for j in range(3):
                if board[i,j] == 0:
                    empty.append((i,j))
        if len(empty) == 0: break

        idx = np.random.randint(0, len(empty))
        r, c = empty[idx]
        board[r,c] = player
        moves.append((r, c, player))

        if check_winner(board) != 0: break
        player = 3 - player

    return board, moves

# --- FORMATTING ---

def board_str(board):
    s = {0:' ', 1:'X', 2:'O'}
    return '\n'.join(['|'.join([s[board[i,j]] for j in range(3)]) for i in range(3)])

def pos(r,c):
    return [['top-left','top-center','top-right'],
            ['mid-left','center','mid-right'],
            ['bot-left','bot-center','bot-right']][r][c]

def status(board):
    w = check_winner(board)
    if w == 1: return "X wins"
    if w == 2: return "O wins"
    if is_full(board): return "Draw"
    return ""

def generate_random_problem():
    """Generate multi-turn tic-tac-toe conversation."""
    while True:
        seed = np.random.randint(0, 1000000000)
        final_board, moves = generate_game(seed)

        if len(moves) < 2: continue  # Need at least X move + O move

        # Build conversation
        board = np.zeros((3,3), dtype=np.int32)
        conv = []

        # First X move
        r, c, p = moves[0]
        board[r,c] = p
        conv.append({
            'role': 'user',
            'content': f"Tic-tac-toe. I'm X at {pos(r,c)}."
        })

        # Subsequent moves
        for i in range(1, len(moves)):
            r, c, player = moves[i]

            if player == 2:  # O's turn (assistant)
                # Build full board state in code (all moves so far + this one)
                code_lines = []
                for mr, mc, mp in moves[:i]:  # All previous moves
                    sym = 'X' if mp == 1 else 'O'
                    code_lines.append(f"board[{mr}][{mc}]='{sym}'")
                code_lines.append(f"board[{r}][{c}]='O'")  # This move
                code_lines.append("print(board)")

                board[r,c] = player  # Update our tracking

                if i == len(moves)-1 or check_winner(board) != 0:
                    code_lines.append("check_winner(board)")

                code = '\n'.join(code_lines)
                msg = f"<|python_start|>\n{code}\n<|python_end|>\n"
                msg += f"<|output_start|>\n{board_str(board)}"
                st = status(board)
                if st: msg += f"\n{st}"
                msg += "\n<|output_end|>"

                conv.append({'role': 'assistant', 'content': msg})

                if check_winner(board) != 0 or is_full(board):
                    break
            else:  # X's turn (user)
                board[r,c] = player  # Update tracking
                conv.append({
                    'role': 'user',
                    'content': f"I play {pos(r,c)}."
                })

        return {
            'conversation': conv,
            'final_board': final_board.tolist(),
            'outcome': status(final_board)
        }


# --- TEST ---
if __name__ == '__main__':
    for i in range(3):
        p = generate_random_problem()
        print(f"\n{'='*50}\nGame {i+1}: {p['outcome']}\n{'='*50}")
        for msg in p['conversation']:
            print(f"\n[{msg['role'].upper()}]\n{msg['content']}")
