from treys import Card, Deck, Evaluator
from poker_logger import PokerLogger
import random
from typing import List, Dict, Optional, Tuple, Any

FOLD = 0
CALL = 1
RAISE = 2
CHECK = 3

class Game():
    def __init__(
        self,
        player_names: Optional[List[str]] = None,
        starting_stack: int = 100,
        small_blind: int = 5,
        big_blind: int = 10,
        logger: Optional[PokerLogger] = None,
        seed: Optional[int] = None,
        debug=False,
    ) -> None:
        self.rng = random.Random(seed)
        self.evaluator = Evaluator()
        self.player_names = player_names or [f"Player {i+1}" for i in range(6)]
        self.num_players = len(self.player_names)
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.logger = logger
        self.debug = debug

        self.button = self.rng.randrange(self.num_players)
        self.stacks: List[int] = [starting_stack] * self.num_players
        self.eliminated: List[bool] = [False] * self.num_players
        self.hands_played = 0
        self.game_over = False
        self.winner: Optional[int] = None

        self.reset_hand_state(first_hand=True)
    
    # ---------------------------
    # Public API (engine drives the flow)
    # ---------------------------
    
    def start_hand(self) -> Dict[str, Any]:
       """Starts (or restart) a new hand. Returns the initial state."""

       if self.game_over:
           return self.get_state()
       
       self.reset_hand_state()
       self.deal_hole_cards()
       self.post_blinds()
       self.set_first_to_act_preflop()

       if self.logger:
           self.logger.logger.info(f"--- New Hand --- (Button: {self.player_names[self.button]})")
       
       return self.get_state()
    
    def step(self, action: int, raise_amount: Optional[int] = None) -> Dict[str, Any]:
        """
        Apply an action for the current player (one atomic move), then progress rounds/showdown as needed.
        Returns the updated state.
        """
        # If hand or game over, return snapshot immediately
        if getattr(self, "hand_over", False) or self.game_over:
            return self.get_state()
        
        # apply action (no exceptions thrown for invalid actions; they are simply ignored)
        self.apply_action(action, raise_amount)
        # after applying, try to advance the round (flop/turn/river/showdown)
        self.maybe_advance_round_or_showdown()

        return self.get_state()
    
    def action(self, action: int, amount_to_call: Optional[int] = None, raise_amount: Optional[int] = None):
        """
        Compatibility wrapper: older callers called action(action, amount_to_call, raise_amount).
        We acceppt positional `action` and ignore the second positional parameter (amount_to_call), using step instead.
        """
        return self.step(action, raise_amount=raise_amount)
    
    def valid_actions(self) -> Dict[str, bool]:
        """Human/agent-friendly dict describing which actions are currently available and min/max raise totals."""
        i = self.current_player

        if i is None:
            return {"fold": False, "check": False, "call": False, "raise": False, "amount_to_call": 0, "min_total_bet": 0, "max_total_bet": 0}
        
        amt_to_call = self.amount_to_call(i)
        can_check = amt_to_call == 0 and self.stacks[i] > 0
        can_call = amt_to_call > 0 and self.stacks[i] > 0
        can_fold = not self.eliminated[i] and not self.folded[i] and not can_check
        can_raise, min_total, max_total = self.raise_window(i, amt_to_call)

        if min_total > max_total:
            if self.stacks[i] > 0:
                can_raise = True
                min_total = self.bets[i] + self.stacks[i]
                max_total = min_total
            
            else:
                can_raise = False
                min_total, max_total = 0, 0

        return {
            "fold": can_fold,
            "check": can_check,
            "call": can_call,
            "raise": can_raise,
            "amount_to_call": amt_to_call,
            "min_total_bet": min_total,
            "max_total_bet": max_total,
        }
    
    def get_valid_actions_mask(self) -> List[bool]:
        """Return mask [FOLD< CALL, RAISE, CHECK] for the current player (useful for RL wrappers)."""
        v = self.valid_actions()

        return [bool(v["fold"]), bool(v["call"]), bool(v["raise"]), bool(v["check"])]
    
    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot dictionary of the current game state (for UI/training/logging)."""
        return {
            "player_hands": [list(h) for h in self.player_hands],
            "community_cards": [c for c in self.community_cards],
            "stacks": list(self.stacks),
            "bets": list(self.bets),
            "total_bets": list(self.total_bets),
            "current_player": self.current_player,
            "player_names": list(self.player_names),
            "stage": self.stage,
            "pots": [dict(p) for p in self.pots],
            "display_pots": self.get_display_pots(),
            "acted": list(self.acted),
            "folded": list(self.folded),
            "hand_over": getattr(self, "hand_over", False),
            "game_over": self.game_over,
            "winner": self.winner,
            "hands_played": self.hands_played,
            "showdown_results": self.showdown_results,
            "valid_actions": self.valid_actions()
        }
    
    # ---------------------------
    # Internal: hand reset / deal / blinds
    # ---------------------------

    def reset_hand_state(self, first_hand: bool = False) -> None:
        """Initialize per-hand variables. Call this to start a new hand."""
        self.deck = Deck()
        self.community_cards: List[int] = []
        self.player_hands: List[List[int]] = [[] for _ in range(self.num_players)]
        self.stage = "preflop"

        self.bets = [0] * self.num_players
        self.total_bets = [0] * self.num_players
        self.acted = [False] * self.num_players
        self.folded = [False] * self.num_players
        self.pots: List[Dict] = [] # side pots as dicts {"amount":int, "eligible": set(...)}
        self.current_player: Optional[int] = None
        self.last_raise_size = self.big_blind

        self.hand_over = False
        self.showdown_results = None

        if not first_hand:
            self.advance_button()
        
        if self.logger:    
            self.logger.log_state(self)

    def advance_button(self) -> None:
        """Move button to next non-eliminated seat."""
        start = (self.button + 1) % self.num_players
        i = start

        while self.eliminated[i]:
            i = (i + 1) % self.num_players

            if i == start:
                break

        self.button = i

    def deal_hole_cards(self) -> None:
        """Deal two hole cards to each non-eliminated player."""
        for i in range(self.num_players):

            if not self.eliminated[i]:
                self.player_hands[i] = self.deck.draw(2)

    def post_blinds(self) -> None:
        """Post small & big blinds (skip eliminated seats)."""
        sb = (self.button + 1) % self.num_players
        bb = (self.button + 2) % self.num_players

        sb = self.next_seat_from(sb, skip_folded=False, include_all_in=True)
        bb = self.next_seat_from(bb, skip_folded=False, include_all_in=True)

        sb_amt = min(self.small_blind, self.stacks[sb])
        self.commit_bet(sb, sb_amt)

        bb_amt = min(self.big_blind, self.stacks[bb])
        self.commit_bet(bb, bb_amt)

        self.rebuild_pots()

    def set_first_to_act_preflop(self) -> None:
        """Set the first to act (left of BB that can act). Also mark who has acted by default."""
        first = self.next_seat_from((self.button + 3) % self.num_players, skip_folded=False, include_all_in=False)
        self.current_player = first

        for i in range(self.num_players):
            self.acted[i] = (self.stacks[i] == 0) or self.eliminated[i] or self.folded[i]

        # blinds should be allowed to act (if they have chips)
        for seat in [(self.button + 1) % self.num_players, (self.button + 2) % self.num_players]:
            if 0 <= seat < self.num_players and not self.eliminated[seat] and not self.folded[seat] and self.stacks[seat] > 0:
                self.acted[seat] = False

    # ---------------------------
    # Internal: apply action
    # ---------------------------

    def apply_action(self, action: int, raise_amount: Optional[int]) -> None:
        """Apply an atomic action for the current player. Invalid actions are ignored."""
        i = self.current_player

        if i is None:
            return
        
        # guard: if seat cannot act, skip forward
        if self.folded[i] or self.eliminated[i] or self.stacks[i] == 0:
            self.current_player = self.next_to_act_from(i)
            return

        amt_to_call = self.amount_to_call(i)

        if action == FOLD:
            self.folded[i] = True
            self.acted[i] = True

            if self.logger:
                self.logger.log_action(self.player_names[i], "fold")

            self.after_action_advance(i)
            return
        
        if action == CHECK:
            if amt_to_call !=0 or self.stacks[i] == 0:
                return
            
            self.acted[i] = True

            if self.logger:
                self.logger.log_action(self.player_names[i], "check")

            self.after_action_advance(i)
            return
        
        if action == CALL:
            if amt_to_call == 0 or self.stacks[i] == 0:
                return
            
            pay = min(amt_to_call, self.stacks[i])
            self.commit_bet(i, pay)
            self.acted[i] = True

            if self.logger:
                self.logger.log_action(self.player_names[i], "call", amount=pay)

            self.rebuild_pots()
            self.after_action_advance(i)
            return

        if action == RAISE:
            can_raise, min_total, max_total = self.raise_window(i, amt_to_call)

            if (not can_raise) or (raise_amount is None):
                return
            
            total_target = max(min_total, min(raise_amount, max_total))
            add = total_target - self.bets[i]

            if add <= 0:
                return
            
            prev_max = max(self.bets)
            self.commit_bet(i, add)
            new_max = max(self.bets)
            self.last_raise_size = max(self.big_blind, new_max - prev_max)
            # re-open action for players not matched

            for p in range(self.num_players):
                self.acted[p] = (self.eliminated[p] or self.folded[p] or self.stacks[p] == 0 or self.bets[p] == new_max)

            if self.logger:
                self.logger.log_action(self.player_names[i], "raise", total_bet=self.bets[i])

            self.rebuild_pots()
            self.after_action_advance(i)
            return
        
        # unknown action -> ignore
        return

    def commit_bet(self, i: int, add: int) -> None:
        """Deduct chips and add to current & total bets."""
        add = max(0, min(add, self.stacks[i]))
        self.stacks[i] -= add
        self.bets[i] += add
        self.total_bets[i] += add

    def after_action_advance(self, acted_by: int) -> None:
        """Advance to the next actor or detect single-player win (hand end)."""
        active = [p for p in range(self.num_players) if not self.folded[p] and not self.eliminated[p]]

        if len(active) == 1:
            self.rebuild_pots()
            total = sum(pot["amount"] for pot in self.pots)
            winner = active[0]
            self.stacks[winner] += total
            self.showdown_results = [{
                "pot": total,
                "winners": [self.player_names[winner]],
                "winning_hands": [[Card.int_to_pretty_str(c) for c in self.player_hands[winner]]],
                "winning_hand_types": ["Uncontested"],
            }]
            self.pots.clear()
            self.end_hand()
            return

        nxt = self.next_to_act_from(acted_by)
        self.current_player = nxt

    def next_to_act_from(self, i: int) -> Optional[int]:
        """Return next player index who has not acted and can act, or None if round is over."""
        if self.betting_round_over():
            return None
        j = (i + 1) % self.num_players
        start = j

        while True:
            if (not self.eliminated[j]) and (not self.folded[j]) and (self.stacks[j] > 0) and (not self.acted[j]):
                return j
            
            j = (j + 1) % self.num_players

            if j == start:
                return None
    
    # ---------------------------
    # Round progression
    # ---------------------------

    def maybe_advance_round_or_showdown(self) -> None:
        """Called after each action: may fast-forward to showdown or advance street when betting round ends."""
        # Fast-forward when all active player are all-in
        if self.all_active_all_in():
            self.deal_remaining_board_to_river()
            self.showdown()
            return

        # If betting round over, advance stage
        if self.betting_round_over():
            if self.stage == "preflop":
                self.advance_stage(3, "flop")
            
            elif self.stage == "flop":
                self.advance_stage(1, "turn")

            elif self.stage == "turn":
                self.advance_stage(1, "river")
            
            elif self.stage == "river":
                self.showdown()

            return
        
        # If current player is None but hand not over, choose next
        if self.current_player is None:
            self.current_player = self.next_to_act_from((self.button + 2) % self.num_players)

    def advance_stage(self, n_cards: int, stage: str) -> None:
        """Deal n_cards to board, reset street bets, set first actor for the new street."""
        self.bets = [0] * self.num_players
        self.acted = [False] * self.num_players
        self.stage = stage

        # draw board cards
        for _ in range(n_cards):
            self.community_cards.extend(self.deck.draw(1))

        # first to act is left of the button who can act
        self.current_player = self.next_seat_from((self.button + 1) % self.num_players, skip_folded=True, include_all_in=False)

        if self.logger:
            self.logger.log_round_start(self)
            self.logger.log_state(self)
    
    def deal_remaining_board_to_river(self) -> None:
        need = 5 - len(self.community_cards)

        if need > 0:
            for _ in range(need):
                self.community_cards.extend(self.deck.draw(1))

    # ---------------------------
    # Pots & Showdown
    # ---------------------------

    def rebuild_pots(self) -> None:
        """Recompute main/side pots from total_bets and folded/eliminated flags."""
        contrib = [(i, self.total_bets[i]) for i in range(self.num_players) if self.total_bets[i] > 0]

        if not contrib:
            self.pots = []
            return
        
        levels = sorted(set(amt for _, amt in contrib))
        pots: List[Dict] = []
        prev = 0

        for lvl in levels:
            contributors = [i for i in range(self.num_players) if self.total_bets[i] >= lvl]
            segment = (lvl - prev) * len(contributors)

            if segment <= 0:
                prev = lvl
                continue

            eligible = {i for i in contributors if not self.folded[i] and not self.eliminated[i]}
            pots.append({"amount": segment, "eligible": eligible})
            prev = lvl

        self.pots = pots

    def showdown(self) -> None:
        """Evaluate hands, allocate pots, and produce showdown_results, then end the hand."""
        self.stage = "showdown"
        self.deal_remaining_board_to_river()

        scores: Dict[int, int] = {}
        hand_types: Dict[int, int] = {}

        for i in range(self.num_players):
            if not self.eliminated[i] and not self.folded[i] and self.player_hands[i]:
                # Evaluator expects (board, hand)
                score = self.evaluator.evaluate(self.community_cards, self.player_hands[i])
                scores[i] = score
                rank = self.evaluator.get_rank_class(score)
                hand_types[i] = self.evaluator.class_to_string(rank)

        results: List[Dict] = []

        for pot in self.pots:
            if not pot["eligible"]:
                continue

            elig_scores = {i: scores[i] for i in pot["eligible"] if i in scores}

            if not elig_scores:
                continue

            best = min(elig_scores.values())
            winners = [i for i, score in elig_scores.items() if score == best]
            share = pot["amount"] // len(winners)
            remainder = pot["amount"] - share * len(winners)

            for w in winners:
                self.stacks[w] += share

            if remainder > 0:
                self.stacks[winners[0]] += remainder

            results.append({
                "pot": pot["amount"],
                "winners": [self.player_names[i] for i in winners],
                "winning_hands": [[Card.int_to_pretty_str(c) for c in self.player_hands[i]] for i in winners],
                "winning_hand_types": [hand_types[i] for i in winners],
                "winning_hand_ranks": [scores[i] for i in winners]
            })

        self.showdown_results = results

        if self.logger:
            self.logger.log_showdown(self)
            self.logger.log_state(self)

        self.pots.clear()
        self.end_hand()

    def end_hand(self) -> None:
        """Mark eliminations, possibly finish game, count hands, and clear per-hand current_player."""
        for i in range(self.num_players):
            if self.stacks[i] == 0:
                self.eliminated[i] = True

        alive = [i for i in range(self.num_players) if not self.eliminated[i]]

        if len(alive) <= 1:
            self.game_over = True
            self.winner = alive[0] if alive else None

        self.hands_played += 1
        self.hand_over = True
        self.current_player = None

    # ---------------------------
    # Seat/turn helpers
    # ---------------------------

    def next_seat_from(self, start: int, *, skip_folded: bool, include_all_in: bool) -> int:
        """Return the next seat index fulfilling constraints: Never returns eliminated seats."""
        i = start % self.num_players
        attempts = 0

        while attempts < self.num_players:
            ok = not self.eliminated[i]

            if skip_folded:
                ok = ok and not self.folded[i]

            if not include_all_in:
                ok = ok and self.stacks[i] > 0

            if ok:
                return i
            
            i = (i + 1) % self.num_players
            attempts += 1

        # fallback to any non-eliminated
        for j in range(self.num_players):
            if not self.eliminated[j]:
                return j
        return 0
    
    # ---------------------------
    # Utility Helpers
    # ---------------------------

    def amount_to_call(self, i: int) -> int:
        return max(self.bets) - self.bets[i]
    
    def raise_window(self, i: int, amount_to_call: int) -> Tuple[bool, int, int]:
        """Return can_raise, min_total_on_street, max_total_on_street."""

        if self.stacks[i] == 0 or self.eliminated[i] or self.folded[i]:
            return (False, 0, 0)
        
        current_total = self.bets[i]
        max_total = current_total + self.stacks[i]

        if amount_to_call == 0 and max(self.bets) == 0:
            min_total = self.big_blind
            return (self.stacks[i] >= min_total, min_total, max_total)
        
        min_total = max(self.bets) + self.last_raise_size
        return (max_total > max(self.bets), min_total, max_total)
    
    def all_active_all_in(self) -> bool:
        """
        Return True if betting cannot continue because all active players are all-in, except one"""
        active = [i for i in range(self.num_players) if not self.folded[i] and not self.eliminated[i]]

        if not active:
            return False
        
        with_chips = [i for i in active if self.stacks[i] > 0]
        all_not_acted = [i for i in active if not self.acted[i]]

        if all_not_acted:
            return False

        if len(with_chips) == 0 or len(with_chips) == 1:
            return True
            
        return False

    def betting_round_over(self) -> bool:
        active = [i for i in range(self.num_players) if not self.folded[i] and not self.eliminated[i]]

        if len(active)  <= 1:
            return True
        
        max_bet = max(self.bets[i] for i in active)

        for i in active:
            if not self.acted[i] and self.stacks[i] > 0:
                return False
            
            if self.bets[i] < max_bet and self.stacks[i] > 0:
                return False
        
        return True
    
    def get_display_pots(self) -> List[Dict]:
        """Return a pots view filtered by active (non-folded/non-eliminated) players for UIs."""
        display_pots = []

        for pot in self.pots:
            active_eligible = {i for i in pot["eligible"] if not self.folded[i] and not self.eliminated[i]}
            display_pots.append({"amount": pot["amount"], "eligible": active_eligible})

        return display_pots