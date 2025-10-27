from engine import Game, FOLD, CHECK, CALL, RAISE
from treys import Card
from poker_logger import PokerLogger
import random
import torch
from torch.distributions import Categorical
from ai import PPOAgent, flatten_state
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Manual player input
# ------------------------------------------------------------
def get_manual_action(game):
    """Prompt the human player for an action."""
    state = game.get_state()
    current_player = state["current_player"]
    name = state["player_names"][current_player]
    stack = state["stacks"][current_player]
    amt_to_call = state["valid_actions"]["amount_to_call"]

    print("\n--- Player & Pot State ---")
    for i, name in enumerate(state['player_names']):
        print(f"{name}: stack={game.stacks[i]}, bet={game.bets[i]}, folded={game.folded[i]}, eliminated={game.eliminated[i]}")
    
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

# ------------------------------------------------------------
# Random action fallback
# ------------------------------------------------------------
def get_random_action(game):
    """Randomly choose a valid action (for missing models)."""
    actions = game.valid_actions()
    possible = [a for a in ["fold", "check", "call", "raise"] if actions[a]]
    act = random.choice(possible)
    amount = None

    if act == "raise":
        amount = random.randint(actions["min_total_bet"], actions["max_total_bet"])
        act = RAISE
    elif act == "fold":
        act = FOLD
    elif act == "call":
        act = CALL
    else:
        act = CHECK

    return act, amount

# ------------------------------------------------------------
# PPO AI decision logic
# ------------------------------------------------------------
def get_ai_action(game, agent):
    """Use a trained PPO model to pick an action."""
    with torch.no_grad():
        state = flatten_state(game.get_state(), device=device)
        logits, _, raise_mean, raise_std = agent(state)
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=1e-6)
        dist = Categorical(probs)
        action_type = dist.sample().item()

        if action_type == RAISE:
            raise_dist = torch.distributions.Normal(raise_mean, raise_std)
            raise_fraction = torch.clamp(raise_dist.sample(), 0.0, 1.0).item()
            valid = game.valid_actions()
            min_bet = valid["min_total_bet"]
            max_bet = valid["max_total_bet"]
            amount = int(min_bet + raise_fraction * (max_bet - min_bet))
        else:
            amount = None

        return action_type, amount

# ------------------------------------------------------------
# Main Gameplay Loop
# ------------------------------------------------------------
def main():
    logger = PokerLogger(filename="logs/poker_vs_ai.log")
    player_names = ["You"] + [f"AI_{i}" for i in range(1, 6)]
    game = Game(player_names=player_names, logger=logger, debug=True)

    # Load PPO models
    num_players = len(player_names)
    state = game.start_hand()
    state_dim = len(flatten_state(state))
    action_dim = 4

    agents = []
    model_dir = "models"
    for i in range(num_players):
        agent = PPOAgent(state_dim, action_dim).to(device)
        path = os.path.join(model_dir, f"poker_ppo_agent_{i}.pth")

        if os.path.exists(path):
            agent.load_state_dict(torch.load(path, map_location=device))
            agent.eval()
            print(f"Loaded model for {player_names[i]}")
        else:
            print(f"No model found for {player_names[i]}. Using random AI.")
            agent = None

        agents.append(agent)

    print("\nGame start! You are Player 0 (You).")
    print("Other seats are AI models.\n")

    # Gameplay loop
    while not state["game_over"]:
        while not state["hand_over"]:
            current_player = state["current_player"]

            if current_player == 0:
                action, amount = get_manual_action(game)
            elif agents[current_player] is not None:
                action, amount = get_ai_action(game, agents[current_player])
            else:
                action, amount = get_random_action(game)

            state = game.step(action, amount)

        if not state["game_over"]:
            state = game.start_hand()

    print("\nGame Over!")
    print("Winner:", state.get("winner", "Unknown"))

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
