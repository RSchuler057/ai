from engine import Game, FOLD, CHECK, CALL, RAISE
from treys import Card
from ai_player import AIPlayer

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
    ### Next few lines are for debugging purposes
    print(f"Players Acted: {state['acted']}")
    print(f"Eliminated Players: {state['eliminated']}")
    print(f"Game Over: {state['game_over']}")
    ### End of debugging lines
    print("-------------------\n")

def get_action(game, ai_players=None):
    state = game.get_state()
    player = state['current_player']
    amount_to_call = max(state['bets']) - state['bets'][player]

    if ai_players and player in ai_players:
        valid_actions = []
        if amount_to_call == 0:
            valid_actions = ['check']
            if state['stacks'][player] >= game.big_blind:
                valid_actions.append('bet')
        else:
            valid_actions = ['call', 'fold']
            if state['stacks'][player] > amount_to_call:
                valid_actions.append('raise')
        
        action, raise_amt = ai_players[player].choose_action(
            valid_actions,
            amount_to_call=amount_to_call,
            min_raise=game.big_blind,
            max_raise=state['stacks'][player]
        )

        if action in ['raise', 'bet']:
            print(f"AI {state['player_names'][player]} chooses to {action} with raise amount: {raise_amt}")

        action_map = {'fold': FOLD, 'call': CALL, 'raise': RAISE, 'check': CHECK, 'bet': RAISE}
        return action_map[action], amount_to_call, raise_amt

    print(f"{state['player_names'][player]}'s turn. Stack: {state['stacks'][player]}, Bet: {state['bets'][player]}, Amount to call: {amount_to_call}")
    print(f"Your cards: {pretty_print_hand(state['player_hands'][player])}\n")
    
    if amount_to_call == 0:
        actions = ['check', 'bet']

    else:
        actions = ['call', 'raise', 'fold']

    print(f"Valid actions: {', '.join(actions)}")
    action = input(f"Choose action ({', '.join(actions)}): ").strip().lower()

    if action == 'fold' and amount_to_call > 0:
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
    iteration = 0
    ai_players = {
        0: AIPlayer("AI Player 1"),
        1: AIPlayer("AI Player 2"),
        2: AIPlayer("AI Player 3"),
        3: AIPlayer("AI Player 4"),
        4: AIPlayer("AI Player 5"),
        5: AIPlayer("AI Player 6"),
    }

    while not game.get_state().get("game_over", False):
        iteration += 1
        print_state(game)
        state = game.get_state()
        player = state['current_player']

        if state['stacks'][player] == 0 or state['folded'][player] or state['eliminated'][player]:
            game.next_player()

            if game.get_state()["done"]:
                print("\nHand over. Starting next hand...")
                game.next_hand()
            continue

        if game.get_state()["done"]:
            print("\nHand over. Starting next hand...")
            game.next_hand()
            continue

        action, amount_to_call, raise_amt = get_action(game, ai_players)
        print(f"\n{state['player_names'][player]} chooses: {ACTION_NAMES[action]}")

        try:
            game.action(action, amount_to_call, raise_amt)

        except ValueError as e:
            print(f"Error: {e}. Please try again.")
            continue

        if game.betting_round_over() and not game.get_state()["done"]:
            if game.all_active_all_in():
                print("\nAll active players are all-in. Proceeding to showdown.")
                game.showdown()
                if game.get_state().get("showdown_results"):
                        for result in game.get_state()["showdown_results"]:
                            for winner, hand in zip(result['winners'], result['winning_hand']):
                                print(f"\nPot: {result['pot']}, Winner: {winner}, Winning Hand: {pretty_print_hand(hand)}, Hand Type: {result['winning_hand_type']}")
                continue
                
            if game.stage == 'preflop':
                    game.flop()

            elif game.stage == 'flop':
                    game.turn()

            elif game.stage == 'turn':
                    game.river()

            elif game.stage == 'river':
                    game.showdown()
                    if game.get_state().get("showdown_results"):
                        for result in game.get_state()["showdown_results"]:
                            for winner, hand in zip(result['winners'], result['winning_hand']):
                                print(f"\nPot: {result['pot']}, Winner: {winner}, Winning Hand: {pretty_print_hand(hand)}, Hand Type: {result['winning_hand_type']}")
                    continue

    print_state(game)
    print(f"\nGame over. Winner: {game.player_names[game.winner]}")
    print("Iterations:", iteration)

if __name__ == "__main__":
    main()
