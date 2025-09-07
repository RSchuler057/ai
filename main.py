from engine import Game, FOLD, CHECK, CALL, RAISE
from treys import Card
from poker_logger import PokerLogger
import random

def get_manual_action(game):
    """Prompt the user for and action and return it."""
    state = game.get_state()
    current_player = state["current_player"]
    name = state["player_names"][current_player]
    stack = state["stacks"][current_player]
    amt_to_call = state["valid_actions"]["amount_to_call"]

    print("\n--- Player & Pot State ---")
    for i, name in enumerate(state['player_names']):
        print(
            f"{name}: stack={game.stacks[i]}, bet={game.bets[i]}, folded={game.folded[i]}, eliminated={game.eliminated[i]}"
        )
    
    for idx, pot in enumerate(state['pots']):
        eligibles = [state['player_names'][j] for j in pot['eligible']]
        print(f"Pot {idx}: {pot['amount']} (eligible: {', '.join(eligibles)})")
    
    print("--------------------------")

    print(f"\n{name}'s turn. Stack: {stack}, Bet: {state['bets'][current_player]}, Amount to call: {amt_to_call}")
    print(f"Your cards: {[Card.int_to_pretty_str(c) for c in state['player_hands'][current_player]]}")
    print(f"Community Cards: {[Card.int_to_pretty_str(c) for c in state['community_cards']]}")

    available = [a for a, v in state["valid_actions"].items() if v is True]
    print("Valid actions:", available)

    while True:
        choice = input(f"Choose action {tuple(available)}: ").strip().lower()
        print("")
        
        if choice == "fold" and "fold" in available:
            return FOLD, None
        
        elif choice == "call" and "call" in available:
            return CALL, None
        
        elif choice == "check" and "check" in available:
            return CHECK, None
        
        elif choice == "raise" and "raise" in available:
            try:
                amount = int(input(f"Enter raise amount (min {state['valid_actions']['min_total_bet']}, max {state['valid_actions']['max_total_bet']}): "))
                return RAISE, amount
            
            except ValueError:
                print("Invalid raise amount. Try again.")
        
        else:
            print("Invalid choice, try again.")

def get_random_action(game):
    """Randomly pick a valid action for AI/Bots/Testing."""
    actions = game.valid_actions()
    
    possible = [a for a in ["fold", "check", "call", "raise"] if actions[a]]

    act = random.choice(possible)
    amount = None

    if act == "raise":
        amount = random.randint(actions["min_total_bet"], actions["max_total_bet"])

    return act, amount
    
def main():
    logger = PokerLogger()
    game = Game(player_names=[f"Player {i+1}" for i in range(6)], logger=logger, debug=True)

    mode = input("Run manually or random? (m/r): ").strip().lower()
    manual = (mode == "m")

    state = game.start_hand()

    while not state["game_over"]:
        while not state["hand_over"]:
            if manual:
                action, amount = get_manual_action(game)
            else:
                action, amount = get_random_action(game)

            state = game.step(action, amount)

        if not state["game_over"]:
            state = game.start_hand()
    
    print("\nGame Over!")
    print("winner:", state["winner"])


if __name__ == "__main__":
    main()
