from engine import Game, FOLD, CALL, RAISE, CHECK
from typing import Optional, List
import gymnasium as gym
from gymnasium import spaces, ActionWrapper
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from treys import Card
import logging
import os

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
        self.game: Optional[Game] = None

        self.observation_space = spaces.Box(low=0, high=1, shape=(self._obs_size(),), dtype=np.float32)

        self.action_space = spaces.MultiDiscrete([4, 601])  # (action_type, raise_amount_level)

        self.prev_stacks = None
        self.current_state = None
        self.game = None

        self.iteration = 0

        self.total_chips_expected = None
    
    def get_action_mask(self) -> List[bool]:
        """Return a mask of valid actions for the current player."""
        """
        Returns a 1D boolean array of length sum(action_space.nvec).
        First 4 entries: mask for [FOLD, CALL, RAISE, CHECK]
        Next 601 entries: mask for the second dimension (raise_level)
        """

        # 0) Prevent crash from get_action_mask call before reset
        if self.game is None or self.current_state is None:
            return np.ones(4 + 601, dtype=bool)

        # 1) mask for the first dimension (action_type)
        base_mask = np.array(self.game.get_valid_actions_mask(), dtype=bool)

        # 2) mask for the second dimension (raise_level)
        # MaskablePPO samples all dimensions at once so this mask must be valid regardless of what was picked for action_type
        level_mask = np.ones(601, dtype=bool)

        # Restrict RAISE to min_raise and max_raise
        if base_mask[RAISE]:
            state = self.current_state
            actor = self.game.current_player
            bets = state['bets']
            stacks = state['stacks']

            amount_to_call = max(bets) - bets[actor]
            min_raise = amount_to_call + self.game.big_blind
            max_raise = stacks[actor] + bets[actor]

            lo = max(0, int(min_raise))
            hi = min(600, int(max_raise))

            level_mask[:] = False
            if hi >= lo:
                level_mask[lo:hi+1] = True
        
        return np.concatenate([base_mask, level_mask])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        state, _ = self.game.start()

        self.total_chips_expected = sum(state['stacks']) + sum(p['amount'] for p in state['pots'])
        
        if not hasattr(self, 'iteration'):
            self.iteration = 1  
        else:
            self.iteration += 1

        logging.info(f"[RESET] Iteration: {self.iteration}")
        logging.info(f"[RESET] --- New Game ---")
        logging.info(f"[RESET] Total chips: {self.total_chips_expected}")
        logging.info(f"[RESET] Stacks: {state['stacks']}")
        logging.info(f"[RESET] Pots: {[p['amount'] for p in state['pots']]}")
        logging.info(f"[RESET] Stage: {state['stage']}")

        self.current_state = state
        self.prev_stacks = state['stacks'].copy()

        return self._encode_state(state), {}

    def step(self, action):
        actor = self.game.current_player
        stacks_before = self.current_state['stacks'].copy()
        pots_before = [p['amount'] for p in self.current_state['pots']]
        total_chips_before = sum(stacks_before) + sum(pots_before)

        action_type = int(action[0])
        raise_level = int(action[1])

        bets = self.current_state['bets']
        amount_to_call = max(bets) - bets[actor]
        min_raise = amount_to_call + self.game.big_blind
        max_raise = stacks_before[actor] + bets[actor]

        valid_mask = self.game.get_valid_actions_mask()
        final_action = FOLD
        final_raise = None
        
        
        if action_type == RAISE:
            raise_span = max_raise - min_raise
            
            if raise_span < 0:
                candidate = int(max_raise)

            else:
                candidate = min_raise + (raise_level % (raise_span + 1))
            
            candidate = max(min_raise, min(candidate, max_raise))

            if self.game.is_action_valid(RAISE, amount_to_call, candidate):
                final_action = RAISE
                final_raise = candidate
        
        elif action_type == CALL and self.game.is_action_valid(CALL, amount_to_call):
            final_action = CALL
        
        elif action_type == CHECK and self.game.is_action_valid(CHECK, amount_to_call):
            final_action = CHECK

        elif action_type == FOLD:
            final_action = FOLD

        if final_action == FOLD:
            _ = self.game.action(FOLD, amount_to_call)

        elif final_action == CHECK:
            _ = self.game.action(CHECK, amount_to_call)

        elif final_action == CALL:
            _ = self.game.action(CALL, amount_to_call)

        elif final_action == RAISE:
            _ = self.game.action(RAISE, amount_to_call, final_raise)

        if self.game.eliminated[actor] or self.game.folded[actor] or self.current_state['stacks'][actor] == 0:
            logging.info(f"[STEP] Auto-advance: Player {actor} cannot act.")

            self.game.acted[actor] = True

            if not self.game.betting_round_over():
                self.game.next_player()

        hand = self.current_state['player_hands'][actor]
        community = self.current_state['community_cards']

        logging.info(f"[STEP] --- Player {actor} Turn ---")
        logging.info(f"[STEP] Hand: {hand}")
        logging.info(f"[STEP] Community: {community}")
        logging.info(f"[STEP] Action requested: (action_type={action_type}, raise_level={raise_level}) -> Executed: ({action_type}, {final_raise})")
        logging.info(f"Stack: {stacks_before[actor]}")
        logging.info(f"Bets: {bets}")
        logging.info(f"Amount to Call: {amount_to_call}")
        logging.info(f"Pot before action: {pots_before}")
        logging.info(f"Total chips before action: {total_chips_before}")

        new_state = self.game.get_state()
        stacks_after = new_state['stacks']
        pots_after = [p['amount'] for p in new_state['pots']]
        total_chips_after = sum(stacks_after) + sum(pots_after)

        logging.info(f"[STEP] Stack after action: {stacks_after}")
        logging.info(f"[STEP] Pot after action: {pots_after}")
        logging.info(f"[STEP] Total chips after action: {total_chips_after}")

        if total_chips_after != self.total_chips_expected:
            logging.warning(f"[WARNING] CHIP LEAKAGE DETECTED! Total chips changed from {total_chips_before} to {total_chips_after}.")
        
        reward = self._calculate_reward(actor, new_state)
        self.current_state = new_state
        self.prev_stacks = new_state['stacks'].copy()

        terminated = bool(new_state['done'])
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
        vec.append(stack / 600.0)
        vec.append(pot / 600.0)
        
        # --- Encode bets ---
        vec += [b / 600.0 for b in bets]

        # --- Encode game stage one-hot ---
        stage_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        stage_vector = [0.0] * 5
        stage_vector[stage_map[state['stage']]] = 1.0
        vec += stage_vector

        # --- Pad to 30 total ---
        while len(vec) < 30:
            vec.append(0.0)

        return np.array(vec, dtype=np.float32)
    
    def _calculate_reward(self, actor_idx, new_state):
        prev_stack = self.prev_stacks[actor_idx]
        new_stack = new_state['stacks'][actor_idx]
        stack_diff = (new_stack - prev_stack) / 600.0

        reward = stack_diff

        folded = new_state['folded'][actor_idx]

        if folded and stack_diff < 0:
            reward -= 0.1

        return reward

def mask_fn(env: PokerEnv):
    return env.get_action_mask()

env = PokerEnv()
env = ActionMasker(env, mask_fn)
check_env(env)

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500)
