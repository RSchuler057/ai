import logging
from treys import Card

class PokerLogger:
    def __init__(self, filename="poker.log", level=logging.INFO):
        self.logger = logging.getLogger("poker_logger")
        self.logger.setLevel(level)

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(filename, mode="w", encoding="utf-8")
            fh.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log_state(self, game):
        """Log player stacks, bets, and pot state."""

        self.logger.info("--- Game State ---")

        for i, name in enumerate(game.player_names):
            self.logger.info(
                f"{name}: stack={game.stacks[i]}, bet={game.bets[i]}, folded={game.folded[i]}, eliminated={game.eliminated[i]}"
            )
        
        for idx, pot in enumerate(game.pots):
            eligibles = [game.player_names[j] for j in pot['eligible']]
            self.logger.info(f"Pot {idx}: {pot['amount']} (eligible: {', '.join(eligibles)})")
        
        self.logger.info("----------------\n")

    def log_action(self, player_name, action, amount=None, total_bet=None):
        if action == "fold":
            self.logger.info(f"{player_name} folds.")
        
        elif action == "check":
            self.logger.info(f"{player_name} checks.")
        
        elif action == "call":
            self.logger.info(f"{player_name} calls {amount}.")

        elif action == "raise":
            self.logger.info(f"{player_name} raises to {total_bet}.")

    def log_round_start(self, game):
        state = game.get_state()
        stage = state["stage"]
        board = state["community_cards"]

        if stage == "flop" and len(board) == 3:
            cards = " ".join(Card.int_to_pretty_str(c) for c in board[:3])
            self.logger.info(f"--- Flop --- {cards}")
        
        elif stage == "turn" and len(board) == 4:
            cards = " ".join(Card.int_to_pretty_str(c) for c in board[:4])
            self.logger.info(f"--- Turn --- {cards}")

        elif stage == "river" and len(board) == 5:
            cards = " ".join(Card.int_to_pretty_str(c) for c in board[:5])
            self.logger.info(f"--- River --- {cards}")

    def log_showdown(self, game):
        state = game.get_state()
        self.logger.info("--- Showdown ---")
        self.logger.info(f"Board: {' '.join(Card.int_to_pretty_str(c) for c in state['community_cards'])}")

        for i, name in enumerate(game.player_names):
            if not game.folded[i] and not game.eliminated[i] and game.player_hands[i]:
                cards = " ".join(Card.int_to_pretty_str(c) for c in game.player_hands[i])
                self.logger.info(f"{name} shows {cards}")

        for res in game.showdown_results:
            if len(res["winners"]) == 1:
                name = res["winners"][0]
                hand = " ".join(res["winning_hands"][0])
                htype = res["winning_hand_types"][0]
                self.logger.info(f"Pot {res['pot']} awarded to {name} with {hand} ({htype})")
            else:
                parts = []
                for w, hand, htype in zip(res["winners"], res["winning_hands"], res["winning_hand_types"]):
                    hand_str = " ".join(hand)
                    parts.append(f"{w} with {hand_str} ({htype})")
                self.logger.info(f"Pot {res['pot']} split between: {', '.join(parts)}")

        self.logger.info("----------------\n")