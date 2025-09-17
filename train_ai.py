import numpy as np
from poker_env import PokerEnv
from train_ppo import PPOTrainer

N_ACTIONS = 6

def run_training(num_episodes=200, update_every=10):
    env = PokerEnv(player_index=0)
    obs0 = env.reset()
    obs_dim = obs0.shape[0]
    trainer = PPOTrainer(obs_dim, n_actions=N_ACTIONS)

    all_trajectory = []
    episode_count = 0

    while episode_count < num_episodes:
        obs = env.reset()
        done = False
        traj = []
        # Play until end of hand
        while not done:
            action_idx, logp, value = trainer.select_action(obs)
            next_obs, reward, done, _ = env.step(action_idx)
            traj.append({
                "obs": obs,
                "action": action_idx,
                "logp": logp,
                "value": value,
                "reward": reward,
                "done": done
            })
            obs = next_obs
        
        # At end of episode add to buffer
        all_trajectory.extend(traj)
        episode_count += 1

        if episode_count % update_every == 0:
            trainer.update(all_trajectory)
            all_trajectory = []
            print(f"[train] episode {episode_count}: performed update")
    
    print("Training done.")

if __name__ == "__main__":
    run_training(200, update_every=8)