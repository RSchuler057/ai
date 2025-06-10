from treys import Card, Deck, Evaluator
import random

FOLD = 0
CALL = 1
RAISE = 2
CHECK = 3

class Game():
    def __init__(self, player_names=None):
        self.evaluator = Evaluator()

        if player_names is None:
            self.player_names = [f"Player {i+1}" for i in range(6)]

        self.current_player = 0
        self.stacks = [100] * 6
        self.small_blind = 5
        self.big_blind = 10
        self.button = random.randint(0, 5)
        self.pots = [] # List of dicts: [{'amount': int, 'eligible': set(player_indices)}]
        self.eliminated = [False] * 6
    
    def check_chip_consistency(self):
        total_chips = sum(self.stacks) + sum(pot['amount'] for pot in self.pots)
        assert total_chips == 600, f"Chip leakage detected! Total chips: {total_chips}"

    def opponents(self):
        return [(self.current_player + i) % 6 for i in range(1 , 6)]
    
    def update_pots(self):
        bets = self.total_bets.copy()
        pots = []
        eligible = [i for i in range(6)]

        while any(bets[i] > 0 for i in eligible):
            min_bet = min(bets[i] for i in eligible if bets[i] > 0)

            pot_players = [i for i in eligible if bets[i] > 0]

            pot_amount = min_bet * len(pot_players)

            pots.append({
                'amount': pot_amount,
                'eligible': set(pot_players)
            })

            for i in pot_players:
                bets[i] -= min_bet
        self.pots = pots
    
    def get_display_pots(self):
        """Return pots with only currently eligible (not folded, not eliminated) players for display."""
        display_pots = []
        for pot in self.pots:
            active_eligible = {i for i in pot['eligible'] if not self.folded[i] and not self.eliminated[i]}
            display_pots.append({
                'amount': pot['amount'],
                'eligible': active_eligible
            })
            
        return display_pots

    def all_active_all_in(self):
        active = [i for i in range(6) if not self.folded[i] and not self.eliminated[i]]

        if len(active) <= 1:
            return False
        
        return all(self.stacks[i] == 0 for i in active)
    
    def start(self):
        self.deck = Deck()
        self.check_chip_consistency()
        self.player_hands = [self.deck.draw(2) for _ in range(6)]
        self.community_cards = []
        self.pots = [] # Reset side pots at the start of each hand
        
        self.bets = [0] * 6
        self.total_bets = [0] * 6
        self.done = False
        self.stage = 'preflop'
        self.acted = [False] * 6
        self.folded = [False] * 6
        self.eliminated = [stack == 0 for stack in self.stacks]
        self.next_stage = False
        self.showdown_results = None

        sb = (self.button + 1) % 6
        while self.stacks[sb] == 0:
            sb = (sb + 1) % 6

        bb = (self.button + 2) % 6
        while self.stacks[bb] == 0 or bb == sb:
            bb = (bb + 1) % 6

        sb_blind = min(self.small_blind, self.stacks[sb])
        self.stacks[sb] -= sb_blind
        self.bets[sb] = sb_blind
        self.total_bets[sb] = sb_blind

        bb_blind = min(self.big_blind, self.stacks[bb])
        self.stacks[bb] -= bb_blind
        self.bets[bb] = bb_blind
        self.total_bets[bb] = bb_blind

        current = (bb + 1) % 6

        while self.stacks[current] == 0:
            current = (current + 1) % 6

        self.current_player = current

        self.update_pots()
        self.update_eliminated()

        return self.get_state(), self.folded
    
    def flop(self):
        self.bets = [0] * 6
        self.community_cards.extend(self.deck.draw(3))
        self.stage = 'flop'
        self.acted = [False] * 6
        self.next_p = (self.button + 1) % 6

        if self.all_active_all_in():
            return self.showdown()

        while self.folded[self.next_p] or self.eliminated[self.next_p]:
            self.next_p = (self.next_p + 1) % 6

        self.current_player = self.next_p
        return self.get_state(), self.folded

    def turn(self):
        self.bets = [0] * 6
        self.community_cards.extend(self.deck.draw(1))
        self.stage = 'turn'
        self.acted = [False] * 6
        self.next_p = (self.button + 1) % 6

        if self.all_active_all_in():
            return self.showdown()

        while self.folded[self.next_p] or self.eliminated[self.next_p]:
            self.next_p = (self.next_p + 1) % 6

        self.current_player = self.next_p
        return self.get_state(), self.folded

    def river(self):
        self.bets = [0] * 6
        self.community_cards.extend(self.deck.draw(1))
        self.stage = 'river'
        self.acted = [False] * 6
        self.next_p = (self.button + 1) % 6

        if self.all_active_all_in():
            return self.showdown()

        while self.folded[self.next_p] or self.eliminated[self.next_p]:
            self.next_p = (self.next_p + 1) % 6

        self.current_player = self.next_p
        return self.get_state(), self.folded

    def showdown(self):
        self.stage = 'showdown'
        scores = []
        hand_ranks = []
        all_winners = []

        for i in range(6):
            if not self.folded[i]:
                score = self.evaluator.evaluate(self.player_hands[i], self.community_cards)
                rank = self.evaluator.get_rank_class(score)

            else:
                score = float('inf')
                rank = None

            scores.append(score)
            hand_ranks.append(rank)
        
        showdown_results = []

        for pot in self.pots:
            eligible_scores = [scores[i] for i in pot['eligible']]

            if not eligible_scores:
                continue

            min_score = min(eligible_scores)
            winners = [i for i in pot['eligible'] if scores[i] == min_score] 
            all_winners.extend(winners)
            win_amt = pot['amount'] // len(winners)

            for i in winners:
                self.stacks[i] += win_amt

            leftover = pot['amount'] - (win_amt * len(winners))

            if leftover > 0:
                self.stacks[winners[0]] += leftover

            showdown_results.append({
                "pot": pot['amount'],
                "winners": [self.player_names[i] for i in winners],
                "winning_hand_rank": hand_ranks[winners[0]],
            })
        
        self.winners = list(set(all_winners))
        self.scores = scores
        self.hand_ranks = hand_ranks
        self.showdown_results = showdown_results
        self.done = True
        self.pots = []  # Clear the pots after awarding
        self.update_eliminated()
        self.declare_winner()
        return self.get_state(), self.folded
    
    def all_in_check(self):
        if self.stacks[self.current_player] == 0:
            self.acted[self.current_player] = True

    def is_action_valid(self, action, amount_to_call, raise_amount=None):
        if self.folded[self.current_player] or self.eliminated[self.current_player]:
            return False
        
        if self.stacks[self.current_player] == 0:
            return False
        
        if action == FOLD:
            return True
        
        elif action == CHECK:
            return amount_to_call == 0
        
        elif action == CALL:
            return amount_to_call > 0 and self.stacks[self.current_player] > 0
        
        elif action == RAISE:
            min_raise = max(self.bets) - self.bets[self.current_player] + self.big_blind

            if raise_amount is None or raise_amount < min_raise:
                return False
            
            if raise_amount > self.stacks[self.current_player] + self.bets[self.current_player]:
                return False
            
            return True
        
        return False
    
    def action(self, action, amount_to_call, raise_amount=None):
        if not self.is_action_valid(action, amount_to_call, raise_amount):
            raise ValueError("Invalid action")
        
        if action == FOLD:
            return self.fold()

        elif self.folded[self.current_player] is not True and amount_to_call == 0 and action == CHECK:
            return self.check()

        elif self.folded[self.current_player] is not True and amount_to_call > 0 and action == CALL:
            return self.call()

        elif self.folded[self.current_player] is not True and action == RAISE and raise_amount is not None:
            return self.raise_bet(raise_amount)
    
    def fold(self):
        self.folded[self.current_player] = True
        self.acted[self.current_player] = True
        active_players = [i for i in range(6) if not self.folded[i] and not self.eliminated[i]]

        if len(active_players) == 1:
            self.done = True
            self.winner = active_players[0]
            total_pot = sum(pot['amount'] for pot in self.pots)
            self.stacks[self.winner] += total_pot
            self.pots = []
            return self.get_state()
        
        
        self.next_player()
        return self.get_state()

    def check(self):
        self.acted[self.current_player] = True
        self.all_in_check()
        self.next_player()
        return self.get_state()
    
    def call(self):
        amount_to_call = max(self.bets) - self.bets[self.current_player]
        amount_to_call = min(amount_to_call, self.stacks[self.current_player])
        self.stacks[self.current_player] -= amount_to_call
        self.bets[self.current_player] += amount_to_call
        self.total_bets[self.current_player] += amount_to_call
        self.all_in_check()
        self.acted[self.current_player] = True
        self.update_pots()
        self.next_player()
        return self.get_state()
        
    def raise_bet(self, amount):
        min_raise = max(self.bets) - self.bets[self.current_player] + self.big_blind
        max_possible = self.stacks[self.current_player] + self.bets[self.current_player]

        if amount < min_raise and amount < max_possible:
            return
        
        amount = min(amount, max_possible)

        if self.stacks[self.current_player] == 0:
            return

        raise_amount = amount - self.bets[self.current_player]
        self.stacks[self.current_player] -= raise_amount
        self.bets[self.current_player] += raise_amount
        self.total_bets[self.current_player] += raise_amount
        self.all_in_check()
        self.update_pots()
        
        for i in range(6):
            if i == self.current_player or self.folded[i]:
                self.acted[i] = True
            else:
                self.acted[i] = False

        self.next_player()
        return self.get_state()
    
    def next_player(self):
        active = [i for i in range(6) if not self.folded[i] and not self.eliminated[i]]

        if not active or all(self.acted[i] for i in active):
            self.done = True
            return self.get_state()
        
        next_p = (self.current_player + 1) % 6
        start_p = next_p

        while self.acted[next_p] or self.folded[next_p] or self.eliminated[next_p] or self.stacks[next_p] == 0:
            next_p = (next_p + 1) % 6
            if next_p == start_p:
                self.done = True
                return self.get_state()
        
        self.current_player = next_p
        return self.get_state()
    
    def next_hand(self):
        self.button = (self.button + 1) % 6
        self.start()
        return self.get_state()
    
    def betting_round_over(self):
        active = [i for i in range(6) if not self.folded[i] and not self.eliminated[i]]
        all_acted = all(self.acted[i] or self.stacks[i] == 0 for i in active)
        active_bets = [self.bets[i] for i in active]
        bets_equal = len(set(active_bets)) == 1
        return all_acted and bets_equal
    
    def update_eliminated(self):
        self.eliminated = [stack == 0 for stack in self.stacks]
        self.declare_winner()
        if self.eliminated.count(False) == 1:
            self.done = True
            self.game_over = True
        return self.get_state()
        
    def declare_winner(self):
        if self.eliminated.count(False) == 1:
            self.winner = [i for i in range(6) if not self.eliminated[i]][0]
            total_pot = sum(pot['amount'] for pot in self.pots)

            if total_pot > 0:
                self.stacks[self.winner] += total_pot
                self.pots = []

            self.game_over = True
            self.done = True
            
        else:
            self.game_over = False
        
        return self.get_state()
        
    def get_state(self):
        return {
            "player_hands": self.player_hands,
            "community_cards": self.community_cards,
            "stacks": self.stacks,
            "bets": self.bets,
            "current_player": self.current_player,
            "player_names": self.player_names,
            "stage": self.stage,
            "pots": self.pots,
            "display_pots": self.get_display_pots(),
            'folded': self.folded,
            "done": self.done,
            "eliminated": self.eliminated,
            "game_over": getattr(self, "game_over", False),
            "winner": getattr(self, "winner", None),
            "showdown_results": getattr(self, "showdown_results", None),
        }
