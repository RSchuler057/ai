import logging
from typing import Any

try:
    from treys import Card
except Exception:
    Card = None  # gracefully handle missing treys

class PokerLogger:
    def __init__(self, filename="poker.log", level=logging.INFO, append=False):
        self.logger = logging.getLogger("poker_logger")
        self.logger.setLevel(level)

        # add a single FileHandler (append or write)
        mode = "a" if append else "w"
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(filename, mode=mode, encoding="utf-8")
            fh.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _pretty_card(self, c: Any) -> str:
        """Return a pretty string for a card value (int or 'As' style)."""
        if Card is None:
            return str(c)
        try:
            # treys Card.int_to_pretty_str expects an int
            if isinstance(c, int):
                return Card.int_to_pretty_str(c)
            # If engine provides 'As' style strings, try to convert
            if isinstance(c, str):
                try:
                    return Card.new(c) and c  # prefer returning original string
                except Exception:
                    return c
        except Exception:
            pass
        return str(c)

    def log_state(self, game):
        """Log player stacks, bets, and pot state."""
        # prefer game.get_state() if available
        state = game.get_state() if hasattr(game, "get_state") else None

        self.logger.info("--- Game State ---")

        # players
        names = getattr(game, "player_names", state.get("player_names") if state else None) or []
        stacks = getattr(game, "stacks", state.get("stacks") if state else None) or []
        bets = getattr(game, "bets", state.get("bets") if state else None) or []
        folded = getattr(game, "folded", state.get("folded") if state else None) or []
        eliminated = getattr(game, "eliminated", state.get("eliminated") if state else None) or []

        for i, name in enumerate(names):
            s = stacks[i] if i < len(stacks) else None
            b = bets[i] if i < len(bets) else None
            f = folded[i] if i < len(folded) else None
            e = eliminated[i] if i < len(eliminated) else None
            self.logger.info(f"{name}: stack={s}, bet={b}, folded={f}, eliminated={e}")

        # pots
        pots = getattr(game, "pots", state.get("pots") if state else None) or []
        for idx, pot in enumerate(pots):
            amount = pot.get("amount", None) if isinstance(pot, dict) else getattr(pot, "amount", None)
            eligible = pot.get("eligible", []) if isinstance(pot, dict) else getattr(pot, "eligible", [])

            # ensure all eligible entries are strings; if they are player indices map to names when possible
            player_names = getattr(game, "player_names", state.get("player_names") if state else None) or []
            def _elig_str(e):
                if isinstance(e, int) and 0 <= e < len(player_names):
                    return str(player_names[e])
                return str(e)
            eligibles = ", ".join(_elig_str(e) for e in eligible) if eligible else ""
            self.logger.info(f"Pot {idx}: {amount} (eligible: {eligibles})")

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
        else:
            self.logger.info(f"{player_name} action={action} amount={amount} total_bet={total_bet}")

    def log_round_start(self, game):
        state = game.get_state() if hasattr(game, "get_state") else {}
        stage = state.get("stage")
        board = state.get("community_cards", [])

        if stage == "flop" and len(board) >= 3:
            cards = " ".join(self._pretty_card(c) for c in board[:3])
            self.logger.info(f"--- Flop --- {cards}")
        elif stage == "turn" and len(board) >= 4:
            cards = " ".join(self._pretty_card(c) for c in board[:4])
            self.logger.info(f"--- Turn --- {cards}")
        elif stage == "river" and len(board) >= 5:
            cards = " ".join(self._pretty_card(c) for c in board[:5])
            self.logger.info(f"--- River --- {cards}")

    def log_showdown(self, game):
        # prefer show results from game object if present
        results = getattr(game, "showdown_results", None)
        state = game.get_state() if hasattr(game, "get_state") else {}

        if not results:
            self.logger.info("--- Showdown (no results) ---")
            return

        # guard empty list
        if len(results) == 0:
            self.logger.info("--- Showdown (no results) ---")
            return

        first = results[0]
        winning_types = first.get("winning_hand_types") if isinstance(first, dict) else None
        if winning_types == ["Uncontested"]:
            winners = first.get("winners", [])
            winner = winners[0] if winners else "Unknown"
            total = first.get("pot", None)
            self.logger.info(f"Everyone folded. {winner} wins {total} chips")
            self.logger.info("---------------")
            return

        self.logger.info("--- Showdown ---")
        board = state.get("community_cards", [])
        self.logger.info(f"Board: {' '.join(self._pretty_card(c) for c in board)}")

        player_hands = getattr(game, "player_hands", state.get("player_hands", []))
        folded = getattr(game, "folded", state.get("folded", []))
        eliminated = getattr(game, "eliminated", state.get("eliminated", []))
        names = getattr(game, "player_names", state.get("player_names", []))

        for i, name in enumerate(names):
            if i < len(folded) and folded[i]:
                continue
            if i < len(eliminated) and eliminated[i]:
                continue
            hand = player_hands[i] if i < len(player_hands) else None
            if hand:
                cards = " ".join(self._pretty_card(c) for c in hand)
                self.logger.info(f"{name} shows {cards}")

        for res in results:
            winners = res.get("winners", []) if isinstance(res, dict) else []
            winning_hands = res.get("winning_hands", []) if isinstance(res, dict) else []
            winning_types = res.get("winning_hand_types", []) if isinstance(res, dict) else []
            pot_amt = res.get("pot", None) if isinstance(res, dict) else None

            if len(winners) == 1:
                name = winners[0]
                hand = winning_hands[0] if winning_hands else []
                hand_str = " ".join(self._pretty_card(c) for c in hand) if hand else ""
                htype = winning_types[0] if winning_types else ""
                self.logger.info(f"Pot {pot_amt} awarded to {name} with {hand_str} ({htype})")
            else:
                parts = []
                for w, hand, htype in zip(winners, winning_hands, winning_types):
                    hand_str = " ".join(self._pretty_card(c) for c in hand) if hand else ""
                    parts.append(f"{w} with {hand_str} ({htype})")
                self.logger.info(f"Pot {pot_amt} split between: {', '.join(parts)}")

        self.logger.info("----------------\n")