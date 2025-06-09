from engine import Game, FOLD, CHECK, CALL, RAISE
from treys import Card

ACTION_NAMES = {0: 'fold', 1: 'call', 2: 'raise', 3: 'check'}

def print_state(game):
    state = game.get_state()
    print("\n--- Game State ---")
    print(f"Stage: {state['stage']}")
    print(f"Community Cards: {pretty_print_hand(state['community_cards'])}")

    for i, stack in enumerate(state['stacks']):
        print(f"{state['player_names'][i]}: {stack} chips | Bet: {state['bets'][i]} | Folded: {state['folded'][i]}")
        
    print(f"Current Player: {state['player_names'][state['current_player']]}")
    print(f"Pot(s): {state['display_pots']}")
    print("-------------------\n")

def get_action(game):
    state = game.get_state()
    player = state['current_player']
    amount_to_call = max(state['bets']) - state['bets'][player]

    print(f"{state['player_names'][player]}'s turn. Stack: {state['stacks'][player]}, Bet: {state['bets'][player]}, Amount to call: {amount_to_call}")
    print(f"Your cards: {pretty_print_hand(state['player_hands'][player])}")
    
    if amount_to_call == 0:
        actions = ['check', 'bet', 'fold']
    else:
        actions = ['call', 'raise', 'fold']

    print(f"Valid actions: {', '.join(actions)}")
    action = input(f"Choose action ({', '.join(actions)}): ").strip().lower()

    if action == 'fold':
        return FOLD, amount_to_call, None
    
    elif action == 'check' and amount_to_call == 0:
        return CHECK, amount_to_call, None
    
    elif action == 'call' and amount_to_call > 0:
        return CALL, amount_to_call, None
    
    elif (action == 'bet' and amount_to_call == 0) or (action == 'raise' and amount_to_call > 0):
        try:
            amt = int(input("Enter raise amount: "))
            return RAISE, amount_to_call, amt
        
        except ValueError:
            print("Invalid amount. Please enter a number.")
            return get_action(game)
        
    else:
        print("Invalid action. Please try again.")
        return get_action(game)
    
def pretty_print_hand(hand):
    return [Card.int_to_pretty_str(card) for card in hand]

def main():
    game = Game()
    game.start()

    while not game.get_state().get("game_over", False):
        print_state(game)
        state = game.get_state()
        player = state['current_player']

        if state['stacks'][player] == 0 or state['folded'][player] or state['eliminated'][player]:
            game.next_player()
            continue

        if game.get_state().get("showdown_results"):
            print("Showdown Results:", game.get_state()["showdown_results"])

        if game.get_state()["done"]:
            print("Hand over. Starting next hand...")
            game.next_hand()
            continue

        action, amount_to_call, raise_amt = get_action(game)
        print(f"{state['player_names'][player]} chooses: {ACTION_NAMES[action]}")

        try:
            game.action(action, amount_to_call, raise_amt)

        except ValueError as e:
            print(f"Error: {e}. Please try again.")
            continue

        if game.betting_round_over() and not game.get_state()["done"]:
            if game.stage == 'preflop':
                    game.flop()

            elif game.stage == 'flop':
                    game.turn()

            elif game.stage == 'turn':
                    game.river()

            elif game.stage == 'river':
                    game.showdown()

    print_state(game)
    print(f"Game over. Winner: {game.player_names[game.winner]}")

if __name__ == "__main__":
    main()
