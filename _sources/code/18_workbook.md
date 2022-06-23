---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Reinforcement Learning Example - Tic Tac Toe

```python
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
```

```python
n_states = 9*8*7*6*5*4*3*2
state_values = np.zeros(n_states)
p2_state_values = np.zeros(n_states)
# outer loop is the number of sets of inner loops to run
# inner loop is just to print out stats for the given number of
# games at a time
outer_loops = 100 # random training
outer_model_loops = 100 # training vs model
inner_loop = 1000
test_games = 100  # games to test after training inner loop
learning_rate = 0.15
win_score = 1
lose_score = -1
draw_score = 0.9
learn_while_testing = True
debug = False
```

```python
# helper function to return the state id given a sequence of moves
def state_id(moves):
    coef = 1
    count = 0
    result = 0
    all_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for id in moves:
        idx = all_moves.index(id)
        all_moves.pop(idx)
        result += coef * idx
        coef *= (9-count)
        count += 1

    return result
```

```python
def get_valid_moves(state):
    all_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    return list(set(all_moves) - set(state))

# assumes board is labeled like
#   0  |  1  |  2
#   --------------
#   3  |  4  |  5
#   --------------
#   6  |  7  |  8
#
# Player 1 moves are in the even indexes (0,2,4,6,8)
# Player 2 moves are in the odd indexes (1,3,5,7)
#
```

```python
# returns:
#   0 -- no winner
#   1 -- player 1 wins
#   2 -- player 2 wins
#   3 -- draw


def check_for_winner(state):
    result = 0
    win1 = set([0, 1, 2])
    win2 = set([3, 4, 5])
    win3 = set([6, 7, 8])
    win4 = set([0, 3, 6])
    win5 = set([1, 4, 7])
    win6 = set([2, 5, 8])
    win7 = set([0, 4, 8])
    win8 = set([2, 4, 6])

    p1_moves = set(state[::2])
    p2_moves = set(state) - p1_moves

    # this could be simpliefied as a loop ...
    if (win1.issubset(p1_moves) or win2.issubset(p1_moves) or win3.issubset(p1_moves) or win4.issubset(p1_moves) or win5.issubset(p1_moves) or win6.issubset(p1_moves) or win7.issubset(p1_moves) or win8.issubset(p1_moves)):

        result = 1

    elif (win1.issubset(p2_moves) or
          win2.issubset(p2_moves) or
          win3.issubset(p2_moves) or
          win4.issubset(p2_moves) or
          win5.issubset(p2_moves) or
          win6.issubset(p2_moves) or
          win7.issubset(p2_moves) or
          win8.issubset(p2_moves)):

        result = 2

    elif len(state) == 9:
        result = 3

    return result

```

```python
def update_values(state, learning_rate, debug=False):

    game = copy.deepcopy(state)
    winner = check_for_winner(game)
    if (winner == 1):
        p1_val = win_score
        p2_val = lose_score
    elif (winner == 2):
        p1_val = lose_score
        p2_val = win_score
    else:
        p1_val = draw_score
        p2_val = draw_score

    id = state_id(game)

    if debug:
        print("game: ", game)
        print("winner: ", winner)
        print("p1 val:", p1_val)
        print("p2_val: ", p2_val)
        print("id: ", id)
        print("before state_values: ", state_values[id])
        print("before p2 state value: ", p2_state_values[id])
        
    # final state gets value assigned. Either win or draw
    state_values[id] = p1_val
    p2_state_values[id] = p2_val
    
    prev_val = p1_val
    p2_prev_val = p2_val

    # rest of values are defined as
    # v(s_n) = v(s_n) + learning_rate * (v(s_n+1) - v(s_n))
    while len(game) > 1:
        game.pop()
        id = state_id(game)
        cur_val = state_values[id]
        p2_cur_val = p2_state_values[id]
        state_values[id] += learning_rate*(prev_val - cur_val)
        p2_state_values[id] += learning_rate*(p2_prev_val - p2_cur_val)
        if debug:
            print("game: ", game)
            print("id: ", id)
            print("prev val:", prev_val)
            print("p2 prev val: ", p2_prev_val)
            print("cur val:", cur_val)
            print("p2 cur val: ", p2_cur_val)
            print("after state_values: ", state_values[id])
            print("after p2 state value: ", p2_state_values[id])
        prev_val = state_values[id]
        p2_prev_val = p2_state_values[id]

```

```python
def random_move(state, force_win=False):

    # play winning move if there is one otherwise play randomly
    valid_moves = get_valid_moves(state)
    move_id = -1
    if force_win:
        for i in range(0, len(valid_moves)):
            test_move = copy.deepcopy(state)
            test_move.append(valid_moves[i])
            if (check_for_winner(test_move) != 0):
                move_id = i
                break
    if move_id == -1:
        move_id = random.randint(0, len(valid_moves) - 1)
    state.append(valid_moves[move_id])


```

```python
def get_scores(state):
    valid_moves = get_valid_moves(state)

    # compute values of each move
    scores = []
    for move in valid_moves:
        test_state = copy.deepcopy(state)
        test_state.append(move)
        id = state_id(test_state)
        if (len(state) % 2 == 0):
            score = state_values[id]
        else:
            score = p2_state_values[id]
        scores.append(score)
    return scores
```

```python
def model_move(state):

    scores = get_scores(state)
    valid_moves = get_valid_moves(state)
    move_id = np.argmax(scores)
    state.append(valid_moves[move_id])
```

```python
def mcts_move(state, iters=25):

    # just allocate max space, it would certainly be smarter to
    # set up a real tree ...
    mcts_scores = np.zeros(n_states)
    mcts_visits = np.zeros(n_states)
    mcts_wins = np.zeros(n_states)

    c = np.sqrt(2.0)

    # play requested iterations
    for i in range(0,iters):

        # play out unvisited moves
        test_game = copy.deepcopy(state)
        node_id = -1
        p_id = state_id(test_game)
        valid_moves = get_valid_moves(test_game)

        while check_for_winner(test_game) == 0:

            # check for unvisited moves at this level
            unvisited_move = False
            for move in valid_moves:
                test_move = copy.deepcopy(test_game)
                test_move.append(move)
                id = state_id(test_move)
                if mcts_visits[id] == 0:
                    test_game = copy.deepcopy(test_move)
                    # randomly play out game
                    while (check_for_winner(test_game) == 0):
                        random_move(test_game)
                    unvisited_move = True
                    node_id = id
                    break
            
            # compute scores for each possible move
            if not unvisited_move:
                parent_n = np.log(mcts_visits[p_id])
                scores = []
                for move in valid_moves:
                    test_move = copy.deepcopy(test_game)
                    test_move.append(move)
                    id = state_id(test_move)
                    score = mcts_wins[id] / mcts_visits[id] + c*np.sqrt(parent_n / mcts_visits[id])
                    scores.append(score)
                # select move with highest score
                id = np.argmax(scores)
                test_game.append(valid_moves[id])
                p_id = state_id(test_game)
                valid_moves = get_valid_moves(test_game)

        winner = check_for_winner(test_game)
        win = 0
        score = 0
        if (winner == 3):
            score = draw_score
        elif (len(state) % 2 + 1 == winner):
            score = win_score
            win = 1
        else:
            score = lose_score
  
        
        # update vals
        id = state_id(state)
        mcts_scores[id] += score
        mcts_visits[id] += 1
        mcts_wins[id] += win

        test_move = copy.deepcopy(state)
        done = False
        for j in range(len(state), len(test_game)):
            if done:
                break
            test_move.append(test_game[j])
            id = state_id(test_move)
            if mcts_visits[id] == 0:
                done = True
            mcts_scores[id] += score
            mcts_visits[id] += 1
            mcts_wins[id] += win

    # update state with highest score
    valid_moves = get_valid_moves(state)
    scores = []
    # print("state: ", state)
    for move in valid_moves:
        test_game = copy.deepcopy(state)
        test_game.append(move)
        id = state_id(test_game)
        scores.append(mcts_scores[id]/mcts_visits[id])
        # print("move: ", move)
        # print("\tmcts_score: ", mcts_scores[id])
        # print("\tmcts_visits: ", mcts_visits[id])
        # print("\tmcts_wins: ", mcts_wins[id])
        # print("\tavg mcts_score: ", mcts_scores[id]/mcts_visits[id])
    id = np.argmax(scores)
    state.append(valid_moves[id])
```

```python
def play_games(n_games, learning_rate=0.15, update_model=False, p1_mode='random', p2_mode='random'):

    p1_wins = 0
    p2_wins = 0
    draws = 0
    for i in range(0,n_games):

        state = []
        while (check_for_winner(state) == 0):

                # Player 1 move
                if p1_mode=='random':
                    random_move(state)
                elif p1_mode=='model':
                    model_move(state)
                elif p1_mode=='mcts':
                    mcts_move(state)

                # stop if p1 won
                if check_for_winner(state) != 0:
                    break

                # player 2 move
                if p2_mode=='random':
                    random_move(state)
                elif p2_mode=='model':
                    model_move(state)
                elif p2_mode=='mcts':
                    mcts_move(state)

        if update_model:
            update_values(state, learning_rate)

        winner = check_for_winner(state)
        if (winner == 1):
            p1_wins += 1
        elif (winner == 2):
            p2_wins += 1
        else:
            draws += 1

    return p1_wins, p2_wins, draws
```

```python
def print_move_sequence(state):
    print("state: ", state)
    for i in range(0, len(state)):

        print("Move ", i+1)
        chars = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        next_play = 'X'
        for play in state[0:i+1]:
            chars[play] = next_play
            if (next_play == 'X'):
                next_play = 'O'
            else:
                next_play = 'X'

        # print the scores and the next state
        scores = get_scores(state[0:i])
        score_chars = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        next_play = '  X '
        for play in state[0:i]:
            score_chars[play] = next_play
            if (next_play == '  X '):
                next_play = '  O '
            else:
                next_play = '  X '
        moves = get_valid_moves(state[0:i])
        for j in range(0, len(moves)):
            id = moves[j]
            score_chars[id] = "{:.2f}".format(scores[j])

        print(" %s | %s | %s \t %s | %s | %s " % (
                score_chars[0], score_chars[1], score_chars[2], chars[0], chars[1], chars[2]))
        print("------------------- \t -----------")
        print(" %s | %s | %s \t %s | %s | %s " % (
                score_chars[3], score_chars[4], score_chars[5], chars[3], chars[4], chars[5]))
        print("------------------- \t -----------")
        print(" %s | %s | %s \t %s | %s | %s " % (
                score_chars[6], score_chars[7], score_chars[8], chars[6], chars[7], chars[8]))
        print("")
```

```python
p1_wins_v_random = []
p2_wins_v_random = []
p1_losses_v_random = []
p2_losses_v_random = []
training_games = []

# Train and test 
for outer in range(0, outer_loops):

    # play random games and update model
    play_games(inner_loop, learning_rate, update_model=True)

    training_games.append((outer+1)*inner_loop)

    # test with player 2 using model, player 1 random
    player_1_wins, player_2_wins, draws = play_games(test_games, learning_rate, update_model=learn_while_testing, p1_mode='random', p2_mode='model')

    p2_wins_v_random.append(player_2_wins)
    p2_losses_v_random.append(player_1_wins)

    # test with player 1 using model, player 2 random
    player_1_wins, player_2_wins, draws = play_games(test_games, learning_rate, update_model=learn_while_testing, p1_mode='model', p2_mode='random')

    p1_wins_v_random.append(player_1_wins)
    p1_losses_v_random.append(player_2_wins)
```

```python
# Train and test using model
for outer in range(0, outer_model_loops):

    # play random games and update model
    play_games(inner_loop, learning_rate, update_model=True)

    # train with player 2 using model, player 1 random
    play_games(inner_loop, learning_rate, update_model=True, p1_mode='random',  p2_mode='model')
    # train with player 1 using model, player 2 random
    play_games(inner_loop, learning_rate, update_model=True, p1_mode='model', p2_mode='random')

    training_games.append(outer_loops*inner_loop + 2*(outer+1)*inner_loop)

    # test with player 2 using model, player 1 mcts
    player_1_wins, player_2_wins, draws = play_games(test_games, learning_rate, update_model=learn_while_testing, p1_mode='random', p2_mode='model')

    p2_wins_v_random.append(player_2_wins)
    p2_losses_v_random.append(player_1_wins)

    # test with player 1 using model, player 2 mcts
    player_1_wins, player_2_wins, draws = play_games(test_games, learning_rate, update_model=learn_while_testing, p1_mode='model', p2_mode='random')

    p1_wins_v_random.append(player_1_wins)
    p1_losses_v_random.append(player_2_wins)
```

```python
# try to visualize a loss
for i in range(0, test_games):

    # player 1 plays randomly
    state = []

    while (check_for_winner(state) == 0):

        # Just play completely randomly
        random_move(state)

        # stop if p1 won
        if check_for_winner(state) != 0:
            break

        # player 2 uses model
        model_move(state)

    winner = check_for_winner(state)
    if winner == 1:
        print_move_sequence(state)
        update_values(state, learning_rate, True)
        break
```

```python
if len(p1_wins_v_random) > 0:
    fig, ax = plt.subplots()
    line1, = ax.plot(np.array(training_games) / 1000, np.array(p1_wins_v_random) /
                    test_games * 100, 'g-', label='Player 1 wins vs Random')
    line2, = ax.plot(np.array(training_games) / 1000, np.array(p2_wins_v_random) /
                    test_games * 100, 'b-', label='Player 2 wins vs Random')
    line3, = ax.plot(np.array(training_games) / 1000, np.array(p1_losses_v_random) /
                    test_games * 100, 'g--', label='Player 1 losses vs Random')
    line4, = ax.plot(np.array(training_games) / 1000, np.array(p2_losses_v_random) /
                    test_games * 100, 'b--', label='Player 2 losses vs Random')
    ax.set_xlabel('Random Training Games (1000s)')
    ax.set_ylabel('Win Percentage')
    ax.set_title('Model Performance Based on Training Size')
    ax.legend()
    plt.show()
```

```python

```

```python

```
