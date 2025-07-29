import random
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import ale_py
import shimmy
import gymnasium as gym # or "import gymnasium as gym"
import stable_baselines3
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import gymnasium as gym

# =================== CONFIG =====================
config = {
    "env_name": "SeaquestNoFrameskip-v4",
    "render_mode": "rgb_array",
    "num_workers": 8,
    "num_steps": 128,
    "batch_size": 256,
    "seed": 0,
    "total_timesteps": 10_000_000,
    "gamma": 0.99,
    "lamda": 0.95,
    "K_epochs" : 4,
    "imagination_horizon" : 128,
    "clip_param" : 0.1,
    "coeff_trans" : 0.3,
    "coeff_reward" : 0.5,
    "coeff_imagine" : 0.1,
    "entropy_coeff"  : 0.01,
    "learning_rate": 2.5e-4,
    "log_dir": 'seaquest/dreamer_final_otherseed',
    "model_save_path": 'seaquest/weights/ppo_{update}_dreamer_final.pth',
    "save_interval": 100,
    "frame_stack": 4,
    
}
if __name__ == '__main__':
    


    class ExperienceReplay():
        def __init__(self, minibatch_size, buffer_size, state_size, num_workers,horizon):
            self.minibatch_size = minibatch_size
            self.buffer_size = buffer_size
            self.state_size = state_size
            self.num_worker = num_workers
            self.horizon = horizon
            self.reset_buffer(horizon, state_size)

        def reset_buffer(self, horizon, state_size):
            transformed_buffer_size = (horizon,) + (self.num_worker,)
            buffer_state_size = transformed_buffer_size + state_size

            self.actions = np.empty(transformed_buffer_size, dtype=np.int32)
            self.rewards = np.empty(transformed_buffer_size, dtype=np.float32)
            self.states = np.empty(buffer_state_size, dtype=np.float32)
            self.next_states = np.empty(buffer_state_size, dtype=np.float32)
            self.dones = np.empty(transformed_buffer_size, dtype=np.int32)
            self.olg_log_probs = np.empty(transformed_buffer_size, dtype=np.float32)
            self.advantages = np.empty(transformed_buffer_size, dtype=np.float32)
            self.values = np.empty(transformed_buffer_size, dtype=np.float32)

            self.head = 0
            self.size = 0

        def add_step(self, state, action, reward, next_state, done, value, olg_log_prob):
            # assert the buffer is not full
            assert self.size < self.buffer_size, "Buffer is full"

            self.states[self.head] = state
            self.actions[self.head] = action
            value = np.squeeze(value)
            self.values[self.head] = value
            self.olg_log_probs[self.head] = olg_log_prob
            self.rewards[self.head] = reward
            self.next_states[self.head] = next_state
            self.dones[self.head] = done
            self.head = (self.head + 1) % self.horizon
            self.size += 1

        def get_minibatch(self):
            assert self.size > self.minibatch_size, "Buffer is empty"
            indices = np.random.randint(0, self.size, size=self.minibatch_size)
            return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], \
            self.dones[indices], self.olg_log_probs[indices], self.values[indices]

        def flatten_buffer(self):
            self.states = self.states.reshape(-1, 4, 84, 84)
            self.actions = self.actions.flatten()
            self.rewards = self.rewards.flatten()
            self.next_states = self.next_states.reshape(-1, 4, 84, 84)
            self.dones = self.dones.flatten()
            self.olg_log_probs = self.olg_log_probs.flatten()
            self.values = self.values.flatten()
            self.advantages = self.advantages.flatten()

        def clean_buffer(self):
            self.reset_buffer(self.horizon, self.state_size)

        def __len__(self):
            return self.size


    class Agent(nn.Module):

        def __init__(self, action_size,num_updates, num_workers=8, num_steps=128, batch_size=256):
            super(Agent, self).__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
            )

            self.transition_model = nn.Linear(512+action_size, 512) # one hot encoded action is concat

            self.actor = nn.Sequential(
                nn.Linear(512, action_size),
                nn.Softmax(dim=-1)

            )
            self.critic = nn.Sequential(
                nn.Linear(512, 1)
            )
            self.reward_model = nn.Sequential(
                nn.Linear(512 + action_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            
            self.number_epochs = 0
           
            self.num_workers = num_workers
            self.num_steps = num_steps
            self.batch_size = batch_size
            self.action_size = action_size
            self.ortogonal_initialization()
            self.num_updates = num_updates
          

        def ortogonal_initialization(self):

            for m in self.reward_model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.transition_model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.critic.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.cnn.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

      
        def forward(self, x, y=None, get_transi=False, get_reward=False):
            x = x / 255.0

            z = self.cnn(x)  # latent representation

            next_z_i = None
            reward_pred = None

            if get_transi or get_reward:
                assert y is not None, "Actions needed to compute transition or reward"

                y_onehot = F.one_hot(y.long(), num_classes=self.action_size).float()
                if get_transi:
                    next_z_i = self.transition_model(torch.cat([z, y_onehot], dim=-1))
                if get_reward:
                    reward_pred = self.reward_model(torch.cat([z, y_onehot], dim=-1)).squeeze(-1)

            logits = self.actor(z)
            value = self.critic(z)
            dist = Categorical(logits)

            return dist, value, next_z_i, reward_pred

        def get_action(self, obs, deterministic=False):
            with torch.no_grad():
                dist, value, _ ,_  = self.forward(obs)
                if deterministic:
                    action = torch.argmax(dist.probs).unsqueeze(0)

                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

            return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

        def decay_learning_rate(self, optimizer,num_updates,cur_update, decay_rate=0.99):
            print("Decaying learning rate")
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], self.number_epochs)
            frac = 1.0 - ((cur_update - 1.0) / (num_updates))
            for param_group in optimizer.param_groups:
                param_group['lr'] *= frac
            if 1e-5 > param_group['lr']:
                param_group['lr'] = 1e-5

        def save_model(self, path='ppo.pth'):
            torch.save(self.state_dict(), path)

        def load_model(self, path='ppo.pth'):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def compute_advantages(experience_replay: ExperienceReplay, agent: Agent, gamma=0.99, lamda=0.95):

        for worker in range(experience_replay.num_worker):
            values = experience_replay.values[:, worker]

            advantages = np.zeros(agent.num_steps, dtype=np.float32)
            last_advantage = 0
            last_value = values[-1]
            for i in reversed(range(agent.num_steps)):
                mask = 1 - experience_replay.dones[i, worker]
                last_value = last_value * mask
                last_advantage = last_advantage * mask
                delta = experience_replay.rewards[i, worker] + gamma * last_value - values[i]
                last_advantage = delta + gamma * lamda * last_advantage
                advantages[i] = last_advantage
                last_value = values[i]

            experience_replay.advantages[:, worker] = advantages
        
        experience_replay.flatten_buffer()
        advantages = experience_replay.advantages
        return advantages

    def rollout_episode(env, agent, experience_replay, render=False):
        time_step = 0
        state = env.reset()
        state = np.array(state)
        state = np.transpose(state, (0, 3, 1, 2))
        total_reward = np.zeros(env.num_envs)
        total_time_step = 0
        step = 0
        step_counter = np.zeros(env.num_envs)
        warmup_steps = 1000
        
        for update in range(1,agent.num_updates + 1 ):
          
            frac = 1.0 - (update - 1.0) / agent.num_updates
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow
            writer.add_scalar('lr', lrnow, update)
        
            for horizon in range(agent.num_steps):
              
                step += 1
                time_step += agent.num_workers
                total_time_step += 1
             

                action, log_prob, value = agent.get_action(torch.from_numpy(state).to(device))
             

                next_state, reward, done_list, info = env.step(action)
                next_state = np.array(next_state)
                next_state = np.transpose(next_state, (0, 3, 1, 2))
              
                done_to_add = [1 if done else 0 for done in done_list]
                experience_replay.add_step(state, action, reward, next_state, done_to_add, value, log_prob)
               
                total_reward += reward

             
                step_counter += 1

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={time_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], time_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], time_step)
                        break

             
                state = next_state
              

            print(f"-" * 50)
            print(f"updating the agent...")
            print(f"-" * 50)
            warmup_factor = min(1.0, update / warmup_steps)
            writer.add_scalar("charts/warmup_factor",  warmup_factor,update)

            train_agent(agent=agent, experience_replay=experience_replay,warmup_factor=warmup_factor,K_epochs=config["K_epochs"],clip_param=config["clip_param"],entropy_coeff=config["entropy_coeff"],
                        imagination_horizon=config["imagination_horizon"],coeff_trans=config["coeff_trans"],coeff_imagine=config["coeff_imagine"],coeff_reward=config["coeff_reward"])
            if update % 100 == 0:
                agent.save_model(f'seaquest/weights/ppo_{update}_critic_rollout.pth')

           
   
    def imagine_rollout(agent, z0, horizon=15, gamma=0.99):
        """
        Performs imagined rollout starting from z0 latent.
        Returns tensors of rewards and values along the imagined trajectory.
        """
        zs = [z0]
        rewards = []
        values = []

        z = z0
        for _ in range(horizon):
            with torch.no_grad():
                logits = agent.actor(z)
                dist = Categorical(logits)
                a = dist.sample()
                a_onehot = F.one_hot(a, num_classes=agent.action_size).float()

                z = agent.transition_model(torch.cat([z, a_onehot], dim=-1))
                zs.append(z)

                r = agent.reward_model(torch.cat([z, a_onehot], dim=-1)).squeeze(-1)
                rewards.append(r)

                v = agent.critic(z).squeeze(-1)
                values.append(v)

        rewards = torch.stack(rewards, dim=0)  
        values = torch.stack(values, dim=0)    

        returns = []
        R = values[-1]
        for t in reversed(range(horizon)):
            R = rewards[t] + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns, dim=0)

        return returns, values
    def train_agent(agent: Agent, experience_replay: ExperienceReplay,warmup_factor: float, K_epochs: int , clip_param: float, entropy_coeff:float, imagination_horizon:int,
                    coeff_trans:float, coeff_reward:float, coeff_imagine:float ):

        advantages = compute_advantages(experience_replay, agent, gamma=0.99, lamda=0.95)
        states = torch.from_numpy(experience_replay.states).to(device)
        actions = torch.from_numpy(experience_replay.actions).to(device)
        old_log_probs = torch.from_numpy(experience_replay.olg_log_probs).to(device).detach()
        next_states = torch.from_numpy(experience_replay.next_states).to(device)
        rewards = torch.from_numpy(experience_replay.rewards).to(device)

        advantages = torch.from_numpy(advantages).to(device)
        values = torch.from_numpy(experience_replay.values).to(device)

        returns = advantages + values

        numer_of_samples = agent.num_steps * experience_replay.num_worker

        number_mini_batch = numer_of_samples // experience_replay.minibatch_size
        assert number_mini_batch > 0, "batch size is too small"
        assert numer_of_samples % experience_replay.minibatch_size == 0, "batch size is not a multiple of the number of samples"
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(numer_of_samples)
        np.random.shuffle(indices)
        for _ in range(K_epochs):
            for batch_index in range(number_mini_batch):
                start = batch_index * experience_replay.minibatch_size
                end = (batch_index + 1) * experience_replay.minibatch_size
                indice_batch = indices[start:end]
                advantages_batch = advantages[indice_batch]
                normalized_advantages = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                rewards_batch = rewards[indice_batch]
                agent.number_epochs += 1
                new_dist, new_values, next_latent, reward_pred = agent(states[indice_batch],get_transi=True,y=actions[indice_batch],get_reward=True)
                
                log_pi = new_dist.log_prob(actions[indice_batch])
                next_gt_latent = agent.cnn(next_states[indice_batch])
                
                ratio = torch.exp(log_pi - old_log_probs[indice_batch].detach())
                surr1 = ratio * normalized_advantages
                surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * normalized_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values.squeeze(), returns[indice_batch])
                transiton_loss = F.mse_loss(next_latent,next_gt_latent)
                reward_loss = F.mse_loss(reward_pred,rewards_batch)

                lambda_trans = coeff_trans * warmup_factor
                lambda_reward = coeff_reward  * warmup_factor


                entropy_loss = new_dist.entropy().mean()
                writer.add_scalar('entropy', entropy_loss, agent.number_epochs)
                writer.add_scalar('transiton_loss',transiton_loss*lambda_trans, agent.number_epochs)
                writer.add_scalar('reward_loss',reward_loss*lambda_reward, agent.number_epochs)

                loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy_loss + lambda_trans * transiton_loss + lambda_reward * reward_loss
                

                if warmup_factor > 0.5:
                    with torch.no_grad():
                        z0 = agent.cnn(states[indice_batch]).detach()

                    imagined_returns, imagined_values = imagine_rollout(agent, z0, horizon=imagination_horizon, gamma=0.99)

                    imagined_returns = imagined_returns.view(-1)
                    imagined_values = imagined_values.view(-1)

                    imagined_critic_loss = F.mse_loss(imagined_values, imagined_returns)
                    lambda_imagine = coeff_imagine * warmup_factor
                    writer.add_scalar('imagined_critic_loss',imagined_critic_loss*lambda_imagine, agent.number_epochs)

                    loss = loss + lambda_imagine * imagined_critic_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

                optimizer.step()
                
                                
                
            experience_replay.clean_buffer()

        

    
    writer = SummaryWriter(log_dir=config["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    #### get number of log files in log directory

    lr = config["learning_rate"]
    env_name = config["env_name"]
    env = gym.make(env_name, render_mode="rgb_array")
    num_workers = config["num_workers"]
    num_steps = config["num_steps"]
    batch_size = config["batch_size"]
    seed = config["seed"]
    total_timesteps = config["total_timesteps"]
    frame_stack = config["frame_stack"]
    env = stable_baselines3.common.env_util.make_atari_env(env_name, n_envs=num_workers, seed=seed,env_kwargs={'render_mode': 'rgb_array'})
    env = stable_baselines3.common.vec_env.vec_frame_stack.VecFrameStack(env, n_stack=frame_stack)
   
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    num_updates = total_timesteps // (int(num_steps * num_workers))
    state_size = (4, 84, 84)
    action_size = env.action_space.n
    experience_replay = ExperienceReplay(batch_size, num_steps * num_workers, state_size=state_size,
                                        num_workers=num_workers,horizon=num_steps)
    agent = Agent(action_size, num_workers=num_workers, num_steps=num_steps, batch_size=batch_size,num_updates=num_updates)
    agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    rollout_episode(env=env,agent=agent,experience_replay=experience_replay)
   