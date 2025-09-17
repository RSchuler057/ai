import numpy as np
from treys import Card
from engine import Game, FOLD, CALL, RAISE, CHECK
from poker_logger import PokerLogger

class PokerEnv:
    """
    Lightweight wrapper that exposes:
        - reset() -> obs
        - step(action_idx) -> (obs, reward, done, info)
    The action space is discrete (6 actions):
        0: fold
        1: call
        2: raise_min (all raise options will be changed later)
        3: raise_half_pot
        4: raise_pot
        5: all_in
    The wrapper maps that discrete choice into (engine_action_int, raise_total_or_None).
    """

    ACTION_NAMES = ["fold", "call", "raise_min", "raise_half_pot", "raise_pot", "all_in"]

    def __init__(self, player_index: int = 0, logger=None):
        self.logger = logger or PokerLogger("poker.log")
        self.game = Game(logger=self.logger)
        self.player_index = player_index
        self.prev_stacks = None

    def reset(self):
        """Start a new hand and return an observation"""
        state = self.game.start_hand()
        self.prev_stacks = list(state['stacks'])
        return self.encode_obs(state)
    
    def step(self, action_idx: int):
        """
        action_idx is an int 0..5 from ACTION_NAMES. This wrapper maps to engine.step(action, amount)
        """
        state = self.game.get_state()
        curr = state['current_player']

        if curr is None:
            obs = self.encode_obs(state)
            done = state.get("done", False) or state.get("game_over", False)
            reward = float(state['stacks'][self.player_index] - self.prev_stacks[self.player_index])
            self.prev_stacks = list(state['stacks'])
            return obs, reward, done, {}
        
        engine_action, raise_total = self.map_action(state, curr, action_idx)

        self.game.step(engine_action, raise_total)

        new_state = self.game.get_state()
        obs = self.encode_obs(new_state)
        done = new_state.get("done", False) or new_state.get("game_over", False)

        reward = float(new_state['stacks'][self.player_index] - self.prev_stacks[self.player_index])
        self.prev_stacks = list(new_state['stacks'])
        return obs, reward, done, {}

    
    def encode_obs(self, state):
        """Simple numeric observation: stacks, bets, total_pot, stage one-hot, player's hole and community cards as 52-bit vectors."""
        num_players = len(state['player_names'])
        stacks = np.array(state['stacks'], dtype=np.float32) / 1000.0
        bets = np.array(state['bets'], dtype=np.float32) / 1000.0
        total_pot = sum(p["amount"] for p in state['pots']) if state['pots'] else 0
        pot_vec = np.array([total_pot / 1000.0], dtype=np.float32)

        # Stages one-hot (preflop, flop, turn, river, showdown)
        stages = ['preflop', 'flop', 'turn', 'river', 'showdown']
        stage_onehot = np.zeros(len(stages), dtype=np.float32)
        try:
            stage_onehot[stages.index(state["stage"])] = 1.0
        except Exception:
            pass

        def encode_card_list(cards):
            vec = np.zeros(52, dtype=np.float32)
            for c in cards:
                idx = self.card_to_index(c)
                vec[idx] = 1.0
            return vec
        
        hole = encode_card_list(state['player_hands'][self.player_index]) if state.get('player_hands') else np.zeros(52, dtype=np.float32)
        community = encode_card_list(state['community_cards']) if state.get('community_cards') else np.zeros(52, dtype=np.float32)

        obs = np.concatenate([stacks, bets, pot_vec, stage_onehot, hole, community]).astype(np.float32)
        return obs

    def card_to_index(self, card_int):
        """treys.Card.get_rank_int -> 2..14, get_suit_int -> 1..4
        Map to 0..51 index: (rank-2)*4 + (suit-1)
        """

        r = Card.get_rank_int(card_int)
        s = Card.get_suit_int(card_int)
        return (r-2) * 4 + (s-1)
    
    def map_action(self, state, curr_player_idx, action_idx):
        """Map discrete action to (engine_action_int, raise_total_or_None)"""
        va = state['valid_actions']
        amount_to_call = va['amount_to_call']
        current_bet = state['bets'][curr_player_idx]
        total_pot = sum(p['amount'] for p in state['pots']) if state['pots'] else 0
        stack = state['stacks'][curr_player_idx]

        if action_idx == 0:
            return FOLD, None
    
        if action_idx == 1:
            if va["call"]:
                return CALL, None
            if va["check"]:
                return CHECK, None
            return FOLD, None
        
        if action_idx == 2:
            if not va['raise']:
                if va['call']:
                    return CALL, None
                return FOLD, None
            
            min_total, max_total = va['min_total_bet'], va['max_total_bet']
            if min_total > max_total:
                total = current_bet + stack
                return RAISE, int(total)

            target = min_total
            target = max(min_total, min(target, max_total))
            return RAISE, int(target)

        if action_idx == 3:
            if not va['raise']:
                if va['call']:
                    return CALL, None
                return FOLD, None
            desired = current_bet + int(0.5 * max(1, total_pot))
            min_total, max_total = va['min_total_bet'], va["max_total_bet"]
            if min_total > max_total:
                return RAISE, int(current_bet + stack)
            target = max(min_total, min(desired, max_total))
            return RAISE, int(target)
        
        if action_idx == 4:
            if not va['raise']:
                if va['call']:
                    return CALL, None
                return FOLD, None
            desired = current_bet + max(1, total_pot)
            min_total, max_total = va['min_total_bet'], va['max_total_bet']
            if min_total > max_total:
                return RAISE, int(current_bet + stack)
            target = max(min_total, min(desired, max_total))
            return RAISE, int(target)
        
        if action_idx == 5:
            total = current_bet + stack
            return RAISE, int(total)
        
        return FOLD, None

