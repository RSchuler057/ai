from engine import Game, FOLD
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import logging
import os
from treys import Card

if os.path.exists("poker_ai.log"):
    os.remove("poker_ai.log")

logging.basicConfig(
    level=logging.INFO,
    filename="poker_ai.log",
    filemode="a", #Append mode to keep logs
    format="%(asctime)s - %(message)s",
)

class PokerEnv(gym.Env):
    def __init__(self):
        super(PokerEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(self._obs_size(),), dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([0,0.0]), high=np.array([2.99, 100.0], dtype=np.float32))

        self.prev_stacks = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        state, _ = self.game.start()

        total_chips = sum(state['stacks']) + sum(p['amount'] for p in state['pots'])
        logging.info(f"[RESET] --- New Game ---")
        logging.info(f"[RESET] Total chips: {total_chips}")
        logging.info(f"[RESET] Stacks: {state['stacks']}")
        logging.info(f"[RESET] Pots: {[p['amount'] for p in state['pots']]}")
        logging.info(f"[RESET] Stage: {state['stage']}")

        self.current_state = state
        self.prev_stacks = state['stacks'].copy()
        return self._encode_state(state), {}

    def step(self, action):
        current_player = self.game.current_player
        stacks_before = self.current_state['stacks'].copy()
        pots_before = [p['amount'] for p in self.current_state['pots']]
        total_chips_before = sum(stacks_before) + sum(pots_before)

        action_type = int(np.clip(action[0], 0, 2))
        raw_raise = float(action[1])

        bets = self.current_state['bets']
        amount_to_call = max(bets) - bets[current_player]
        min_raise = amount_to_call + self.game.big_blind
        max_raise = stacks_before[current_player] + bets[current_player]
        raise_amount = min_raise + raw_raise * (max_raise - min_raise)

        hand = self.current_state['player_hands'][current_player]
        community = self.current_state['community_cards']

        logging.info(f"[STEP] --- Player {current_player} Turn ---")
        logging.info(f"[STEP] Hand: {hand}")
        logging.info(f"[STEP] Community: {community}")
        logging.info(f"[STEP] Action requested: ({action[0]}, {action[1]} -> Executed: ({action_type}, {raise_amount})")
        logging.info(f"Stack: {stacks_before[current_player]}")
        logging.info(f"Bets: {bets}")
        logging.info(f"Amount to Call: {amount_to_call}")
        logging.info(f"Pot before action: {pots_before}")
        logging.info(f"Total chips before action: {total_chips_before}")

        if not self.game.is_action_valid(action_type, amount_to_call, raise_amount):
            logging.warning(f"[INVALID ACTION] ({action_type}, {raise_amount} is invalid. Forcing FOLD.)")
            self.game.action(FOLD, amount_to_call)
        else:
            self.game.action(action_type, amount_to_call, raise_amount)
        
        new_state = self.game.get_state()
        
        stacks_after = new_state['stacks']
        pots_after = [p['amount'] for p in new_state['pots']]
        total_chips_after = sum(stacks_after) + sum(pots_after)

        logging.info(f"[STEP] Stack after action: {stacks_after}")
        logging.info(f"[STEP] Pot after action: {pots_after}")
        logging.info(f"[STEP] Total chips after action: {total_chips_after}")

        if total_chips_after != 600:
            logging.warning(f"[WARNING] CHIP LEAKAGE DETECTED! Total chips changed from {total_chips_before} to {total_chips_after}.")

        done = new_state['done']
        reward = self._calculate_reward(new_state)
        self.current_state = new_state

        terminated = done
        truncated = False

        return self._encode_state(new_state), reward, terminated, truncated, {}
    
    def _obs_size(self):
        return 30
    
    def _encode_state(self, state):
        """Convert raw game state into a neural network input vector"""
        player = state['current_player']
        hand = state['player_hands'][player]
        community = state['community_cards']
        stack = state['stacks'][player]
        bets = state['bets']
        pot = sum(p['amount'] for p in state['pots'])

        def card_to_index(card_int):
            rank = Card.get_rank_int(card_int)
            suit = Card.get_suit_int(card_int)
            idx = rank * 4 + suit
            if idx < 0 or idx > 51:
                logging.warning(f"Card index out of range: {idx} for card {card_int}")
            return min(idx, 51)  # Ensure index is within bounds (Clamps index to 51)

        vec = []

        # --- Encode hand cards (scaled 0-1) ---
        vec += [card_to_index(c) / 51.0 for c in hand]
        vec += [0.0] * (2 - len(hand))  # Pad to 2 cards

        # --- Encode community cards (scaled 0-1) ---
        vec += [card_to_index(c) / 51.0 for c in community]
        vec += [0.0] * (5 - len(community))  # Pad to 5 cards

        # --- Encode stack and pot size ---
        vec.append(stack / 100.0)
        vec.append(pot / 100.0)
        
        # --- Encode bets ---
        vec += [b / 100.0 for b in bets]

        # --- Encode game stage one-hot ---
        stage_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        stage_vector = [0.0] * 5
        stage_vector[stage_map[state['stage']]] = 1.0
        vec += stage_vector

        # --- Pad to 30 total ---
        while len(vec) < 30:
            vec.append(0.0)

        return np.array(vec, dtype=np.float32)
    
    def _calculate_reward(self, new_state):
        current_player = new_state['current_player']
        prev_stack = self.prev_stacks[current_player]
        new_stack = new_state['stacks'][current_player]

        stack_diff = (new_stack - prev_stack) / 100.0  # Normalize stack change

        reward = stack_diff

        if new_state['done']:
            if 'winner' in new_state and new_state['winner'] == current_player:
                reward += 1.0
        
        folded = new_state['folded'][current_player]
        if folded and stack_diff < 0:
            reward -= 0.1
        
        self.prev_stacks = new_state['stacks'].copy()

        return reward

env = PokerEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
