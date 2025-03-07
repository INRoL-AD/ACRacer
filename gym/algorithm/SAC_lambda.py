import time
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import tqdm
import pickle
from copy import deepcopy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from algorithm import core

## Hyperparameters for the discriminator (not necessary, but useful for stable training)
LOG_DISC_MAX = -1
LOG_DISC_MIN = 1


class ExtendedReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, track_name, obs_dim, act_dim, size, frame_dim=None, frame_exist=False, is_demo_permanent=False):
        self.track_name = track_name
        if track_name == "monza":
            self.track_section_cand = [0, 1, 3, 5, 6, 8, 9]
            self.laptime_threshold = 142460     # Used for the prior calculation (In our work, we used the average laptime of the demonstrations)
        elif track_name == "silverstone":
            self.track_section_cand = [0, 2, 3]
            self.laptime_threshold = 113850
        
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.section_buf = np.zeros(size, dtype=np.float32)
        self.laptime_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.demo_size, self.max_size = 0, 0, 0, size
        self.is_demo_permanent = is_demo_permanent      # If True, the demo data is not overwritten by the new data.
        self.ptr_temp = []
        
        self.frame_exist = frame_exist
        if self.frame_exist:
            self.frame_buf = np.zeros(core.combined_shape(size, frame_dim), dtype=np.uint8)
            self.frame2_buf = np.zeros(core.combined_shape(size, frame_dim), dtype=np.uint8)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def store(self, obs, act, rew, next_obs, done, tp, laptime=None):
        """
        Store the transitions into the buffer.
        Episode replays are temporarily holded back and labels are assigned based on lap times.
        The labeled data are then evaluated against predefined thresholds to quantitatively determine its positiveness.
        Additionally, uniform sampling is applied w.r.t track sections.
        """
        if self.frame_exist:
            self.obs_buf[self.ptr] = obs[0]
            self.obs2_buf[self.ptr] = next_obs[0]
            self.frame_buf[self.ptr] = obs[1]
            self.frame2_buf[self.ptr] = next_obs[1]
        else:
            self.obs_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        ##### Calc. track section #####
        ## 0: Straight, 1: R Sweeper, 2: L Sweeper, 3: R Corner, 4: L Corner,
        ## 5: R-L Chicane, 6: L-R Chicane, 7: R-L-R Esses, 8: L-R-L Esses,
        ## 9: R U-Turn, 10: L U-Turn, 11: R Hairpin, 12: L Hairpin.
        if self.track_name == "monza":
            if (tp >= 0 and tp < 0.11) or (tp >= 0.38 and tp < 0.41) or (tp >= 0.51 and tp < 0.64) or (tp >=0.73 and tp < 0.85) or (tp >= 0.97 and tp <= 1.00):
                self.section_buf[self.ptr] = 0
            elif (tp >= 0.19 and tp < 0.33):
                self.section_buf[self.ptr] = 1
            elif (tp >= 0.41 and tp < 0.47) or (tp >= 0.47 and tp < 0.51):
                self.section_buf[self.ptr] = 3
            elif (tp >= 0.11 and tp < 0.19):
                self.section_buf[self.ptr] = 5
            elif (tp >= 0.33 and tp < 0.38):
                self.section_buf[self.ptr] = 6
            elif (tp >= 0.64 and tp < 0.73):
                self.section_buf[self.ptr] = 8
            elif (tp >= 0.85 and tp < 0.97):
                self.section_buf[self.ptr] = 9
        
        elif self.track_name == "silverstone":
            if (tp >= 0 and tp < 0.06) or (tp >= 0.13 and tp < 0.19) or (tp >= 0.40 and tp < 0.50) or (tp >= 0.58 and tp < 0.64) or (tp >= 0.72 and tp < 0.77) or (tp >= 0.83 and tp < 0.91):
                self.section_buf[self.ptr] = 0
            elif (tp >= 0.19 and tp < 0.25) or (tp >= 0.33 and tp < 0.40) or (tp >= 0.77 and tp < 0.83):
                self.section_buf[self.ptr] = 2
            elif (tp >= 0.06 and tp < 0.13) or (tp >= 0.25 and tp < 0.33) or (tp >= 0.50 and tp < 0.58) or (tp >= 0.64 and tp < 0.72) or (tp >= 0.91 and tp <= 1.00):
                self.section_buf[self.ptr] = 3
        ###############################
        
        ## Propagate the laptime label
        ## -1 if a lap is not completed and laptime[ms] otherwise
        self.laptime_buf[self.ptr] = -1
        self.ptr_temp.append(self.ptr)
        if done == 1 and laptime:
            for i in self.ptr_temp:
                self.laptime_buf[i] = laptime
            # print("Laptime {} has propagated idx from{} to {}.".format(laptime, self.ptr_temp[0], self.ptr_temp[-1]))
            self.ptr_temp = []
        elif done == 1:
            # print("Done... but laptime is {}. Neglect idx from {} to {}.".format(laptime, self.ptr_temp[0], self.ptr_temp[-1]))
            self.ptr_temp = []

        if self.is_demo_permanent:
            self.ptr = max(self.demo_size, (self.ptr+1) % (self.max_size))
        else:
            self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)       


    def sample_batch(self, batch_size=32, demo_portion=None, is_uniform=False):
        """
        Sample a batch of data from the buffer.
        """
        if is_uniform:
            section_indices = {section: np.where(self.section_buf[:self.size] == section)[0]
                               for section in self.track_section_cand}
            
            ## Select the section where the agent has visited at least once
            available_sections = {section: idxs for section, idxs in section_indices.items() if len(idxs) > 0}
            num_available_sections = len(available_sections)
            
            ## Uniformly sample among the available sections
            sampled_idxs = []
            samples_per_section = batch_size // num_available_sections
            for section, idxs in available_sections.items():
                # Restrict the number of samples if collected data is less than the required samples
                if len(idxs) >= samples_per_section:
                    sampled_idxs.extend(np.random.choice(idxs, samples_per_section, replace=False))
                else:
                    sampled_idxs.extend(idxs)
            
            ## Additional sampling if the number of samples is not enough
            remaining_samples = batch_size - len(sampled_idxs)
            if remaining_samples > 0:
                remaining_sections = np.concatenate(list(available_sections.values()))
                sampled_idxs.extend(np.random.choice(remaining_sections, remaining_samples, replace=False))
            idxs = np.array(sampled_idxs)
        
        else:
            if not demo_portion:
                idxs = np.random.randint(0, self.size, size=batch_size)
            else:
                demo_size = int(batch_size * demo_portion)
                idxs = np.random.randint(0, self.demo_size, size=demo_size)
                if batch_size - demo_size > 0:
                    idxs = np.append(idxs, np.random.randint(self.demo_size, self.size, size=batch_size-demo_size))
        
        if self.frame_exist:
            batch = dict(obs=(self.obs_buf[idxs], self.frame_buf[idxs]),
                        obs2=(self.obs2_buf[idxs], self.frame2_buf[idxs]),
                        act=self.act_buf[idxs],
                        rew=self.rew_buf[idxs],
                        done=self.done_buf[idxs])
        else:
            batch = dict(obs=self.obs_buf[idxs],
                        obs2=self.obs2_buf[idxs],
                        act=self.act_buf[idxs],
                        rew=self.rew_buf[idxs],
                        done=self.done_buf[idxs])
        
        for k, v in batch.items():
            if isinstance(v, tuple):
                batch[k] = [torch.as_tensor(v[i], dtype=torch.float32).to(self.device) for i in range(len(v))]
            else:
                batch[k] = torch.as_tensor(v, dtype=torch.float32).to(self.device)
        
        return batch


    def calc_prior(self, demo_size=0):
        """
        Calculate the prior of the data based on the laptime.
        """
        laptimes_subset = self.laptime_buf[:self.size]
        negative_laptimes = laptimes_subset[laptimes_subset < 0]
        unlabeled_laptimes = laptimes_subset[laptimes_subset > 0]
        positive_laptimes = unlabeled_laptimes[unlabeled_laptimes < self.laptime_threshold]
        prior = (len(positive_laptimes) + demo_size) / (self.size + demo_size + 1e-6)

        return prior


class Algo:
    def __init__(self, env, base=core.MLPActorCritic, frame_exist=False, track_exist=False,
                 seed=0, steps_per_epoch=20000, epochs=400, replay_size=int(1e6),
                 batch_size=1024, batch_size_demo=1024, is_uniform_sampling=False, is_demo_permanent=False,
                 demo_sample_portion=None, decay_rate_demo_portion=0.9,
                 lr_pi=1e-3, lr_q=1e-3, lr_disc=1e-3, gamma=0.99, polyak=0.995, alpha=0.2, lamb=0.0,
                 hidden_sizes=[], hidden_sizes_disc=[], lr_decay_pi=1, lr_decay_q=1, lr_decay_disc=1,
                 decay_every_pi=40000, decay_every_q=40000, decay_every_disc=40000,
                 start_steps=200000, use_buffer_after=0, update_after=20000, update_every=1000,
                 num_update=1000, num_test_episodes=10, max_ep_len=20000, save_freq=1, log_dir="./",
                 pretrain_kwargs=None, **kwargs):

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.frame_exist = frame_exist
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.batch_size_demo = batch_size_demo
        self.is_uniform_sampling = is_uniform_sampling
        self.is_demo_permanent = is_demo_permanent
        self.demo_sample_portion = demo_sample_portion
        self.decay_rate_demo_portion = decay_rate_demo_portion
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lamb = lamb
        self.start_steps = start_steps
        self.use_buffer_after = use_buffer_after
        self.update_after = update_after
        self.update_every = update_every
        self.num_update = num_update
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.random_respawn = kwargs['is_random_spawn']
        self.random_respawn_portion = kwargs['random_spawn_portion']
        
        self.initial_epoch = 0
        self.when_best_lap_time_updated = {}    # Record the steps when best lap time is achieved. {training_steps: best_lap_time}
        self.env = env(frame_exist, track_exist, **kwargs)
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape
        if frame_exist:
            frame_dim = self.env.frame_dim
        else:
            frame_dim = None
            
        ## Initialize the agent's buffer and demonstrations
        self.replay_buffer = ExtendedReplayBuffer(track_name=self.env.track_name, obs_dim=obs_dim, act_dim=act_dim,size=replay_size,
                                                  frame_dim=frame_dim, frame_exist=frame_exist, is_demo_permanent=is_demo_permanent)
        pretrain_buffer_path = pretrain_kwargs["demo_ckpt_name"]
        with open(pretrain_buffer_path, 'rb') as f:
            self.demo = pickle.load(f)

        ## Load pre-trained model if necessary
        if pretrain_kwargs["pretrain_dir"] is not None:
            pretrain_ckpt_path = pretrain_kwargs["pretrain_dir"] + "ckpt/" + pretrain_kwargs["pretrain_ckpt_name"] + ".pt"
            checkpoint = torch.load(pretrain_ckpt_path)
            self.initial_epoch = checkpoint['step'] // steps_per_epoch
            self.ac.load_state_dict(checkpoint['ac'])
            self.ac_targ.load_state_dict(checkpoint['ac_targ'])
            self.disc.load_state_dict(checkpoint['disc'])
            self.pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])

        ## Initialize the buffer with demonstrations if necessary
        if pretrain_kwargs["init_buffer_with_demo"]:
            demo_size = self.demo.size
            self.replay_buffer.obs_buf[0:demo_size] = self.demo.obs_buf
            self.replay_buffer.obs2_buf[0:demo_size] = self.demo.obs2_buf
            self.replay_buffer.act_buf[0:demo_size] = self.demo.act_buf
            self.replay_buffer.rew_buf[0:demo_size] = self.demo.rew_buf
            self.replay_buffer.done_buf[0:demo_size] = self.demo.done_buf
            self.replay_buffer.section_buf[0:demo_size] = self.demo.section_buf
            self.replay_buffer.laptime_buf[0:demo_size] = self.demo.laptime_buf
            self.replay_buffer.size = demo_size
            self.replay_buffer.demo_size = demo_size
            self.replay_buffer.ptr = demo_size
            self.frame_exist = frame_exist
            if self.frame_exist:
                self.replay_buffer.frame_buf[0:demo_size] = self.demo.frame_buf
                self.replay_buffer.frame2_buf[0:demo_size] = self.demo.frame2_buf
            print(" DEMO loaded successfully: {} demos".format(demo_size))
        
        ## Load pre-trained buffer if necessary
        elif pretrain_kwargs["with_pretrain_buffers"]:
            pretrain_buffer_path = pretrain_kwargs["pretrain_dir"] + "ckpt/replay_buffer.pkl"
            with open(pretrain_buffer_path, 'rb') as f:
                self.replay_buffer = pickle.load(f)
                self.replay_buffer.ptr_temp = []
                print(" BUFFER loaded successfully")
        
        ## Create actor-critic module, target network, and discriminator
        if frame_exist:
            self.ac = base(self.env.observation_space, frame_dim, self.env.action_space, hidden_sizes)
        else:
            self.ac = base(self.env.observation_space, self.env.action_space, hidden_sizes)
        self.ac_targ = deepcopy(self.ac)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.disc = core.Disc(self.env.observation_space, self.env.action_space, hidden_sizes_disc)
        self.eta = min(1, self.replay_buffer.calc_prior(demo_size=self.demo.size))
        
        for p in self.ac_targ.parameters():     ## Freeze target networks with respect to optimizers (only update via polyak averaging)
            p.requires_grad = False   
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2, self.disc])   ## Count variables
        wandb.config.update({'Number of parameters': {'pi': var_counts[0], 'q1': var_counts[1], 'q2': var_counts[2], 'disc': var_counts[3]}})
        wandb.log({"Alpha": self.alpha}, commit=False)
        wandb.log({"Eta": self.eta}, commit=False)

        ## Set up optimizers for policy, Q-functions, and discriminator
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr_pi)
        self.q_optimizer = Adam(self.q_params, lr=lr_q)
        self.disc_optimizer = Adam(self.disc.parameters(), lr=lr_disc)
        self.pi_scheduler = StepLR(self.pi_optimizer, step_size=decay_every_pi, gamma=lr_decay_pi)
        self.q_scheduler = StepLR(self.q_optimizer, step_size=decay_every_q, gamma=lr_decay_q)
        self.disc_scheduler = StepLR(self.disc_optimizer, step_size=decay_every_disc, gamma=lr_decay_disc)


    def compute_loss_disc(self, data, demo):
        """
        Compute a loss for the discriminator.
        """
        o, a = data['obs'], data['act']
        o_demo, a_demo = demo['obs'], demo['act']
        
        _, _, logp_pi_original = self.ac.pi(o, with_logprob_original=True)
        _, _, logp_pi_demo_original = self.ac.pi(o_demo, with_logprob_original=True)
        logits_agent = self.disc(o, a, logp_pi_original)
        logits_expert = self.disc(o_demo, a_demo, logp_pi_demo_original)
        
        ## PU learning objective
        loss_expert = self.eta * -F.logsigmoid(logits_expert).mean()
        loss_agent = -F.logsigmoid(-logits_agent).mean()
        loss_extra = self.eta * F.logsigmoid(-logits_expert).mean()
        loss_disc = loss_expert + loss_agent + loss_extra
        
        ## Useful info for logging
        disc_info = dict(DiscAgent=self.disc.d(o, a, logp_pi_original).cpu().detach().numpy(),
                        DiscExpert=self.disc.d(o_demo, a_demo, logp_pi_demo_original).cpu().detach().numpy())

        return loss_disc, disc_info


    def compute_loss_q(self, data):
        """
        Compute a loss for the Q-networks.
        """
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        ## Bellman backup for Q functions
        with torch.no_grad():
            ## Target actions come from current policy
            a2, logp_a2, logp_a2_original = self.ac.pi(o2, with_logprob_original=True)

            ## Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            extra = torch.clamp(torch.log(1 - self.disc.d(o2, a2, logp_a2_original)), min=LOG_DISC_MIN, max=LOG_DISC_MAX)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.lamb * extra - self.alpha * logp_a2)

        ## MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = (loss_q1 + loss_q2)/2

        ## Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                    Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info
    

    def compute_loss_pi(self, data):
        """
        Compute a loss for the policy.
        """
        o = data['obs']
        pi, logp_pi, logp_pi_original = self.ac.pi(o, with_logprob_original=True)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        extra = torch.clamp(torch.log(1 - self.disc.d(o, pi, logp_pi_original)), min=LOG_DISC_MIN, max=LOG_DISC_MAX)

        ## Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi + self.lamb * extra).mean()

        ## Useful info for logging
        if logp_pi.is_cuda:
            logp_pi = logp_pi.cpu()
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info
    

    def update(self, data, demo):
        """
        Update the SAC_lambda agent.
        """
        ## 1) Perform a gradient descent step for the discriminator
        self.disc_optimizer.zero_grad()
        loss_disc, disc_info = self.compute_loss_disc(data, demo)
        loss_disc.backward()
        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=1.0)
        self.disc_optimizer.step()
        self.disc_scheduler.step()
        
        wandb.log({"lr_disc": self.disc_scheduler.get_last_lr()[0]}, commit=False)
        wandb.log({"LossDisc": loss_disc.item()}, commit=False)

        ## 2) Perform a gradient descent step for the Q-functions
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        self.q_scheduler.step()

        wandb.log({"lr_q": self.q_scheduler.get_last_lr()[0]}, commit=False)
        wandb.log({"LossQ": loss_q.item()}, commit=False)

        ## 3) Perform a gradient descent step for the policy.
        for p in self.q_params:
            p.requires_grad = False
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        self.pi_scheduler.step()
        for p in self.q_params:
            p.requires_grad = True

        wandb.log({"lr_pi": self.pi_scheduler.get_last_lr()[0]}, commit=False)
        wandb.log({"LossPi": loss_pi.item()}, commit=False)

        ## Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def get_action(self, o, deterministic=False):
        """
        Get an action from the policy considering deterministic or stochastic.
        """
        if self.frame_exist:
            obs = []
            for i in range(len(o)):
                obs.append(torch.as_tensor(o[i], dtype=torch.float32).to(self.device))   
            return self.ac.act(obs, deterministic)
        else:
            return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), deterministic)
        
        
    def test_agent(self, best_test_ep_ret):
        """
        Test the agent and log the results.
        """
        ep_ret_list = []
        ep_len_list = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                ## Take deterministic actions at test time 
                o, r, d, _, info = self.env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
            ep_len_list.append(ep_len)
        o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0

        if np.mean(ep_ret_list) > best_test_ep_ret:
            best_test_ep_ret = np.mean(ep_ret_list)
            self.save_model(self.log_dir, "best_model")
            
        if len(ep_ret_list) > 0:
            wandb.log({"TestEpRetAvg": np.mean(ep_ret_list), "TestEpLenAvg": np.mean(ep_len_list)}, commit=False)
            wandb.log({"TestEpRetStd": np.std(ep_ret_list), "TestEpLenStd": np.std(ep_len_list)}, commit=False)
            wandb.log({"TestEpRetMax": np.max(ep_ret_list), "TestEpLenMax": np.max(ep_len_list)}, commit=False)
            wandb.log({"TestEpRetMin": np.min(ep_ret_list), "TestEpLenMin": np.min(ep_len_list)}, commit=False)

        return best_test_ep_ret
    
    
    def save_model(self, log_dir, ckpt_name):
        """
        Save the model and buffer.
        """
        checkpoint = {
            'step': self.step,
            'pi_optimizer': self.pi_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'disc': self.disc.state_dict(),
            'ac': self.ac.state_dict(),
            'ac_targ': self.ac_targ.state_dict(),
        }
        save_problem = True
        while save_problem:
            torch.save(checkpoint, log_dir + "/ckpt/" + ckpt_name + ".pt")
            try:
                torch.load(log_dir + "/ckpt/" + ckpt_name + ".pt")
                save_problem = False
            except:
                print("Model save failed. Try again.")

        save_problem = True
        while save_problem:
            with open(log_dir + "/ckpt/" + "replay_buffer.pkl", 'wb') as f:
                pickle.dump(self.replay_buffer, f)
            try:
                with open(log_dir + "/ckpt/" + "replay_buffer.pkl", 'rb') as f:
                    replay_buffer = pickle.load(f)
                save_problem = False
            except:
                print("Buffer save failed. Try again.")


    def train(self):
        """
        Train the SAC_lambda agent.
        """
        update_time = 0
        o, ep_ret, ep_len = self.env.reset(self.random_respawn, self.random_respawn_portion), 0, 0
        ep_ret_list = []
        ep_len_list = []
        best_test_ep_ret = -np.inf

        for epoch in tqdm.trange(self.initial_epoch, self.epochs, initial=self.initial_epoch, total=self.epochs, desc='Epoch', unit='epoch'):
            start_time = time.time()
            self.epoch = epoch
            for t in tqdm.trange(self.steps_per_epoch, desc='Training', unit='step'):
                self.step = epoch * self.steps_per_epoch + t
                if self.step > (self.initial_epoch * self.steps_per_epoch + self.start_steps):
                    a = self.get_action(o)
                else:
                    a = self.env.action_space.sample()

                o2, r, d, _, info, obs_dict = self.env.step(a, return_dict=True)
                update_time = 0
                ep_ret += r
                ep_len += 1

                tp = obs_dict['ego']['track_progress']
                if d and (obs_dict['ego']['lap_count'] == 2):
                    self.replay_buffer.store(o, a, r, o2, d, tp, obs_dict['ego']["last_lap_time"])
                else:
                    self.replay_buffer.store(o, a, r, o2, d, tp)
                wandb.log({"BufferPtr": self.replay_buffer.ptr}, commit=False)
                o = o2

                if info["best_lap_time_updated"]:
                    self.env.pause()
                    s_t = time.time()
                    self.when_best_lap_time_updated[self.step] = obs_dict['ego']["best_lap_time"]
                    wandb.config.update({"when_best_lap_time_updated":self.when_best_lap_time_updated}, allow_val_change=True)
                    self.save_model(self.log_dir, "best_lap_time_model")
                    self.env.resume()

                if d or (ep_len == self.max_ep_len):
                    ep_ret_list.append(ep_ret)
                    ep_len_list.append(ep_len)
                    wandb.log({"EpRet": ep_ret, "EpLen": ep_len}, commit=False)
                    o, ep_ret, ep_len = self.env.reset(self.random_respawn, self.random_respawn_portion), 0, 0

                if (self.step >= (self.initial_epoch * self.steps_per_epoch + self.update_after)) and (self.step % self.update_every == 0) and (self.step != 0) and (self.epoch != self.initial_epoch):
                    self.env.pause()
                    s_t = time.time()
                    for j in tqdm.trange(self.num_update, desc='Updating', unit='update', leave=False):
                        batch = self.replay_buffer.sample_batch(self.batch_size, is_uniform=self.is_uniform_sampling)
                        demo = self.demo.sample_batch(self.batch_size_demo, is_uniform=self.is_uniform_sampling)
                        self.update(data=batch, demo=demo)
                    update_time = time.time()-s_t
                    wandb.log({"update_time": update_time}, commit=False)
                    self.env.resume()

                if (self.step+1) % self.steps_per_epoch == 0:
                    best_test_ep_ret = self.test_agent(best_test_ep_ret)
                    wandb.log({"Epoch": epoch}, commit=False)
                    if len(ep_ret_list) > 0:
                        wandb.log({"EpRetAvg": np.mean(ep_ret_list), "EpLenAvg": np.mean(ep_len_list)}, commit=False)
                        wandb.log({"EpRetStd": np.std(ep_ret_list), "EpLenStd": np.std(ep_len_list)}, commit=False)
                        wandb.log({"EpRetMax": np.max(ep_ret_list), "EpLenMax": np.max(ep_len_list)}, commit=False)
                        wandb.log({"EpRetMin": np.min(ep_ret_list)}, commit=False)
                    ep_ret_list, ep_len_list = [], []
                    
                    wandb.log({'EpochInterval': time.time()-start_time}, commit=False)
                    if ((epoch+1) % self.save_freq == 0) or ((epoch+1) == self.epochs):
                        self.save_model(self.log_dir, "model_epoch_" + str(epoch))

                    self.eta = min(1, self.replay_buffer.calc_prior(self.demo.size))
                    if self.step > self.use_buffer_after:
                        self.demo_sample_portion = self.demo_sample_portion * self.decay_rate_demo_portion
                    wandb.log({"Alpha": self.alpha}, commit=False)
                    wandb.log({"Eta": self.eta}, commit=False)
                    wandb.log({"DemoPortion": self.demo_sample_portion}, commit=False)
                    
                    o, ep_ret, ep_len = self.env.reset(self.random_respawn, self.random_respawn_portion), 0, 0

                wandb.log({"Step_train": self.step}, commit=True)


    def test(self):
        """
        Test the SAC_lambda agent.
        """
        o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
        while not(d or (ep_len == self.max_ep_len)):
            o, r, d, _, info = self.env.step(self.get_action(o, True))
            ep_ret += r
            ep_len += 1
    
