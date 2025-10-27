# ------------------------------------------------------------
# PPO + Poker Engine Integration (with your PokerLogger)
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from engine import Game
from poker_logger import PokerLogger
import numpy as np
import os

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
lr = 3e-4
gamma = 0.99
eps_clip = 0.2
K_epochs = 4
hidden_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_eq_weight = 0.2
anneal_episodes = 1000

def current_eq_weight(episode):
    return float(init_eq_weight * max(0.0, 1.0 - episode / max(1, anneal_episodes)))

# ------------------------------------------------------------
# Initialize Poker Logger
# ------------------------------------------------------------
logger = PokerLogger(filename="logs/poker_training.log")

# ------------------------------------------------------------
# Helper: Flatten game state into numeric tensor
# ------------------------------------------------------------
def flatten_state(state_dict, device=device):
    vals = [
        *state_dict["stacks"],
        *state_dict["bets"],
        *state_dict["total_bets"],
        len(state_dict["community_cards"]) / 5.0,
        state_dict.get("current_player", 0),
        float(state_dict["hand_over"]),
        float(state_dict["game_over"]),
        *map(float, state_dict["folded"]),
        *map(float, state_dict["acted"]),
    ]
    return torch.as_tensor(vals, dtype=torch.float32, device=device)

# ------------------------------------------------------------
# PPO Agent
# ------------------------------------------------------------
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden_size, action_dim)
        self.value = nn.Linear(hidden_size, 1)

        self.raise_mean = nn.Linear(hidden_size, 1)
        self.rasie_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.fc(x)
        logits = self.policy(x)
        value = self.value(x)
        raise_mean = torch.sigmoid(self.raise_mean(x))
        raise_std = torch.exp(self.rasie_std)
        return logits, value, raise_mean, raise_std

# ------------------------------------------------------------
# PPO update (per-agent)
# ------------------------------------------------------------
def ppo_update(agent, optimizer, memory):
    if len(memory) < 2:
        return  # no data for this agent

    states = torch.stack([m[0] for m in memory]).to(device)
    actions = torch.tensor([m[1] for m in memory], dtype=torch.long).to(device)
    old_log_probs = torch.stack([m[2].detach() for m in memory]).to(device)
    rewards = [m[3] for m in memory]

    rewards = [r if np.isfinite(r) else 0.0 for r in rewards]
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    for _ in range(K_epochs):
        logits, values, _, _ = agent(states)
        values = values.squeeze(-1)

        # build distribution from logits (more stable) and compute log_probs
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)

        # advantages: returns - values (detach values for advantage)
        advantages = returns - values.detach()
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std.item() > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        # value loss uses un-normalized returns (squared error)
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy = dist.entropy().mean()

        loss = policy_loss + value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

# ------------------------------------------------------------
# Reward (uses player_idx and prev_stacks_all)
# ------------------------------------------------------------
def convert_card_to_treys(card):
    try:
        from treys import Card
    except Exception:
        return None

    if isinstance(card, str):
        ranks = '23456789TJQKA'
        suits = 'cdhs'

        r = ranks[(card % 13)]
        s = suits[(card // 13) % 4]

        try:
            return Card.new(r + s)
        except Exception:
            return None
        
    return None    

def estimate_equity(state, player_idx):
    """
    Monte-Carlo equity estimate for the given player.
    Falls back to simple heuristics if treys isn't installed or hole cards unavailable.
    """
    num_sim = 1000
    hole_cards_all = state.get("hole_cards") or state.get("hands") or None
    if not hole_cards_all:
        return 0.5

    try:
        from treys import Card, Deck, Evaluator
    except Exception:
        try:
            my_hole = hole_cards_all[player_idx]
            # crude heuristic
            if isinstance(my_hole, (list, tuple)) and len(my_hole) == 2:
                a, b = my_hole[0], my_hole[1]
                if isinstance(a, str) and isinstance(b, str):
                    if a[0] == b[0]:
                        return 0.65
                    if 'A' in (a[0], b[0]) or 'K' in (a[0], b[0]):
                        return 0.55
        except Exception:
            pass
        return 0.5

    community = state.get("community_cards", []) or []
    comm_t = [c for c in (convert_card_to_treys(c) for c in community) if c is not None]

    try:
        my_hole_raw = hole_cards_all[player_idx]
        my_hole = [convert_card_to_treys(c) for c in my_hole_raw]
        if None in my_hole or len(my_hole) != 2:
            return 0.5
    except Exception:
        return 0.5

    num_players = len(hole_cards_all)
    folded = state.get("folded")
    active_opponents = [i for i in range(num_players) if i != player_idx]
    if folded and isinstance(folded, (list, tuple)):
        active_opponents = [i for i in range(num_players) if i != player_idx and not folded[i]]
    num_opponents = len(active_opponents)

    evaluator = Evaluator()
    wins = 0
    ties = 0
    sims = max(1, int(num_sim))

    # build base deck and remove known cards
    base_deck = Deck()
    for c in comm_t + my_hole:
        if c in base_deck.cards:
            base_deck.cards.remove(c)

    for _ in range(sims):
        base_deck.shuffle()
        sample_cards = base_deck.cards[:]  # copy
        idx = 0
        opp_hands = []
        for _ in range(num_opponents):
            if idx + 1 >= len(sample_cards):
                break
            c1 = sample_cards[idx]; c2 = sample_cards[idx+1]; idx += 2
            opp_hands.append([c1, c2])
        needed = 5 - len(comm_t)
        board = comm_t[:]
        for j in range(needed):
            if idx >= len(sample_cards):
                break
            board.append(sample_cards[idx]); idx += 1

        hero_score = evaluator.evaluate(board, my_hole)
        opp_scores = [evaluator.evaluate(board, h) for h in opp_hands] if opp_hands else []
        best_opp = min(opp_scores) if opp_scores else 999999
        if hero_score < best_opp:
            wins += 1
        elif hero_score == best_opp:
            ties += 1

    equity = (wins + 0.5 * ties) / sims
    return float(max(0.0, min(1.0, equity)))

def compute_reward(env, prev_stacks_all, new_state, action_type, player_idx,
                   starting_stack=100.0,
                   eq_weight=0.5,
                   risk_aversion=0.005,
                   terminal_weight=1.0,
                   fold_penalty=0.02):
    """
    Reward combines:
      - Primary: terminal (hand-over) normalized net chip change
      - Immediate: small normalized stack change per step
      - EV shaping: (equity - pot_odds) if equity info available (small weight)
      - Risk penalty: proportional to chips committed this step
      - Fold penalty: small negative for folding (tune carefully)
    """
    stacks = new_state.get("stacks", [])
    pot = float(new_state.get("pot", 0.0))

    # previous stack snapshot for this player (keeps existing structure)
    try:
        prev_stack = prev_stacks_all[player_idx][player_idx]
    except Exception:
        prev_stack = stacks[player_idx]

    # basic normalized immediate stack change
    stack_diff = stacks[player_idx] - prev_stack
    immediate = stack_diff / max(1.0, starting_stack)

    # estimate chips risked this action (best-effort)
    chips_risked = max(0.0, prev_stack - stacks[player_idx])

    # EV shaping: if env or estimator can give equity, compare to pot odds
    ev_bonus = 0.0
    equity = None
    try:
        equity = estimate_equity(new_state, player_idx)
        # best effort call_amount: use chips risked as proxy
        call_amount = chips_risked
        if call_amount > 0 and pot + call_amount > 0:
            pot_odds = call_amount / (pot + call_amount)
            ev_bonus = eq_weight * (equity - pot_odds)  # +ve if action is +EV
    except Exception:
        ev_bonus = 0.0

    # small penalty for folding marginally (tunable)
    fold_term = -fold_penalty if action_type == 0 else 0.0

    # terminal bonus when hand ends (dominant signal)
    terminal = 0.0
    if new_state.get("hand_over", False):
        net_gain = stacks[player_idx] - starting_stack
        terminal = terminal_weight * (net_gain / max(1.0, starting_stack))

        # extra bonus for winning at showdown if env exposes winner info
        if new_state.get("winner_idx", None) == player_idx:
            terminal += 0.1  # small extra reward for straightforward win

    # risk penalty (discourage reckless large bets that lose learning signal)
    risk_pen = -risk_aversion * chips_risked / max(1.0, starting_stack)

    # combine terms; keep terminal component dominant
    reward = immediate + ev_bonus + fold_term + risk_pen + terminal

    # safety clamp
    if not np.isfinite(reward):
        reward = 0.0

    return float(reward)

def train_ai(num_episodes=200):
    player_names = [f"AI_{i+1}" for i in range(6)]
    env = Game(player_names=player_names, logger=logger)

    agents = None
    optimizers = None

    for episode in range(num_episodes):
        state_dict = env.start_hand()

        if agents is None:
            state = flatten_state(state_dict)
            state_dim = state.numel()
            action_dim = 4
            agents = [PPOAgent(state_dim, action_dim).to(device) for _ in range(len(player_names))]
            optimizers = [optim.Adam(agent.parameters(), lr=lr) for agent in agents]

        num_players = len(player_names)
        memories = [[] for _ in range(num_players)]
        prev_stacks_all = [state_dict["stacks"].copy() for _ in range(num_players)]
        total_rewards = [0.0 for _ in range(num_players)]

        done = False
        while not done:
            current_player = state_dict["current_player"]
            state = flatten_state(state_dict)
            mask = env.get_valid_actions_mask()

            agent = agents[current_player]
            optimizer_agent = optimizers[current_player]

            logits, _, raise_mean, raise_std = agent(state)

            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
            logits_masked = logits.clone()
            if logits_masked.numel() == mask_tensor.numel():
                logits_masked[~mask_tensor] = -1e9

            dist = Categorical(logits=logits_masked)
            action_type = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_type, device=device)).detach()

            if action_type == 2:
                raise_dist = torch.distributions.Normal(raise_mean, raise_std)
                raise_fraction = torch.clamp(raise_dist.sample(), 0.0, 1.0).item()
            else:
                raise_fraction = 0.0

            try:
                next_state = env.step(action_type, raise_fraction)
            except TypeError:
                next_state = env.step(action_type)

            # use annealed equity weight
            eq_w = current_eq_weight(episode)
            reward = compute_reward(env, prev_stacks_all, next_state, action_type, current_player, eq_weight=eq_w)
            total_rewards[current_player] += reward

            prev_stacks_all[current_player] = next_state["stacks"].copy()

            memories[current_player].append((state.detach(), action_type, log_prob, reward))

            state_dict = next_state
            done = state_dict.get("hand_over", False)

        for i in range(num_players):
            ppo_update(agents[i], optimizers[i], memories[i])

        memories = [[] for _ in range(num_players)]

        print(f"\nEpisode {episode}")
        for i, r in enumerate(total_rewards):
            print(f"  Player {i+1} total reward: {r:.3f}")
        print("-" * 50)

        if state_dict.get("game_over", False):
            print("Game over detected - stopped training.")
            break

    # Save models
    os.makedirs("models", exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"models/poker_ppo_agent_{i+1}.pth")

    print("Training complete. Models saved individually.")
    print("Full gameplay logs available in poker_training.log")

if __name__ == "__main__":
    for i in range(5):
        train_ai()
