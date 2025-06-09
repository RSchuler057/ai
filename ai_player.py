from stable_baselines3 import DQN
import numpy as np
from engine import Game, FOLD, CHECK, CALL, RAISE

model = DQN.load("poker_dqn")

def ai_action(game, player):
    state = game.get_state()
    obs = np.zeros(20)
    obs[:6] = state['stacks']
    obs[6:12] = state['bets']

    action, _ = model.predict(obs, deterministic=True)

    if action == 0:
        return FOLD, 0, None
    
    elif action == 1:
        return CALL, 0 , None
    
    elif action == 2:
        return RAISE, 0, 20
    
    else:
        return CHECK, 0 , None