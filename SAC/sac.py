import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from SAC.utils import soft_update, hard_update
from SAC.model import GaussianPolicy, QNetwork, DeterministicPolicy, ThreatModel, HeuThreatModel, DiscretePolicy


class SAC(object):
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, discrete_action, gamma=0.95, tau=0.01, critic_lr=0.01, policy_lr=0.01, threat_lr=0.01, 
                    entropy_lr=0.01, hidden_dim=64, device="cuda", constrained=True, policy_type="sac", automatic_entropy_tuning=True, bound=None, 
                        seq_len_in=None, seq_len_out=None):

        self.gamma = gamma
        self.tau = tau
        self.alpha = 0.2

        self.policy_type = policy_type
        self.target_update_interval = 1
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = device

        self.niter = 0

        # INFO: set the critic, target critic and the optimization
        self.critic = QNetwork(num_in_critic, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_target = QNetwork(num_in_critic, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "sac":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.Tensor([3]).to(self.device).item() # 2
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=entropy_lr)
            # INFO: set the policy and its optimization
            self.policy = GaussianPolicy(num_in_pol, num_out_pol, hidden_dim, bound).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)
        elif self.policy_type == "sac_discrete":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.Tensor([-0.22]).to(self.device).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=entropy_lr)
            # INFO: set the policy and its optimization
            self.policy = DiscretePolicy(num_in_pol, num_out_pol, hidden_dim, bound).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_in_pol, num_out_pol, hidden_dim, bound).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=policy_lr)

        # INFO: set the threat model
        # self.threat = ThreatModel(num_inputs=12*seq_len_in, hidden_dim=hidden_dim)
        # self.threat_optim = Adam(self.threat.parameters(), lr=threat_lr)
        self.threat = HeuThreatModel(seq_len_in=seq_len_in, seq_len_out=seq_len_out)
        

    def select_action(self, state, augment_obs=None):
        state = state[0].unsqueeze(0)
        if augment_obs is not None:
            state = torch.concat((state, self.threat(augment_obs.view(1,-1)).detach()), dim=-1)
        # INFO: reparameter trick in training
        action, _, _ = self.policy.sample(state)
        return [action.squeeze()]

    def get_agent_action(self, obs, low_maddpgs, train_low=False):
        if obs.ndim == 1:
            batch_size = 1
        else:
            batch_size = obs.shape[0]
        candidate_actions = []
        for sub_i, subpolicy in enumerate(low_maddpgs):
            if sub_i < 3:
                torch_agent_actions = subpolicy.agents[0].policy(obs)
                # torch_agent_actions = subpolicy.agents[0].policy(obs[...,:16])
            elif sub_i < 4:
                torch_agent_actions = subpolicy.agents[0].policy(obs)
            else:
                raise NotImplementedError
            # step([obs[0][...,:16]], explore=False)
            candidate_actions.append(torch_agent_actions)
        torch_candidate_actions = torch.cat(candidate_actions, dim=-1)
        if train_low == False:
            torch_candidate_actions = torch_candidate_actions.detach()
        # INFO: Augment the high observation with low proposed actions
        high_observation = torch.cat((obs, torch_candidate_actions), dim=-1).view(batch_size, -1)
        subpolicy_coeffs, log_pi, _ = self.policy.sample(high_observation)
        subpolicy_coeffs = subpolicy_coeffs.view(batch_size, -1, 1)
        action = torch.sum((subpolicy_coeffs * torch_candidate_actions.view(batch_size,-1,2)), dim=1)
        return high_observation.detach(), action.clamp(-1, 1), log_pi

    def update(self, sample, agent_i, low_maddpgs, train_option="regular", logger=None, train_low = False):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, dones_batch = sample
        state_batch, action_batch, next_state_batch = state_batch[0], action_batch[0], next_state_batch[0]
        reward_batch = reward_batch[0].unsqueeze(1)
        mask_batch = (1 - dones_batch[0]).unsqueeze(1)
        augmented_obs, pi, log_pi = self.get_agent_action(state_batch, low_maddpgs, train_low)

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)


        with torch.no_grad():
            augmented_nextObs, next_state_action, next_state_log_pi = self.get_agent_action(next_state_batch, low_maddpgs, train_low)
            # next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # INFO: double Q network
            qf1_next_target, qf2_next_target = self.critic_target(augmented_nextObs, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(augmented_obs, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # INFO: gradent can pass through policy!
        
        # pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(augmented_obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.niter % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            
        if train_low:
            for subpolicy in low_maddpgs:
                torch.nn.utils.clip_grad_norm(subpolicy.agents[0].policy.parameters(), 0.5)
                subpolicy.agents[0].policy_optimizer.step()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'qf1_loss': qf1_loss.item(),
                                'qf2_loss': qf2_loss.item(),
                                'min_qf_pi': min_qf_pi.mean().item(),
                                'alpha': self.alpha.item(),
                                'policy_loss': policy_loss.item(),
                                'log_pi': log_pi.mean(),
                                'alpha_loss': alpha_loss.item(),
                                'alpha_tlogs': alpha_tlogs.item(),
                                },
                               self.niter)
            
        self.niter = self.niter + 1
            
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def update_bl(self, sample, agent_i, train_option="regular", logger=None):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, dones_batch = sample
        state_batch, action_batch, next_state_batch = state_batch[0], action_batch[0], next_state_batch[0]
        reward_batch = reward_batch[0].unsqueeze(1)
        mask_batch = (1 - dones_batch[0]).unsqueeze(1)
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # INFO: double Q network
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # INFO: gradent can pass through policy!
        
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.niter % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'qf1_loss': qf1_loss.item(),
                                'qf2_loss': qf2_loss.item(),
                                'min_qf_pi': min_qf_pi.mean().item(),
                                'alpha': self.alpha.item(),
                                'policy_loss': policy_loss.item(),
                                'log_pi': log_pi.mean(),
                                'alpha_loss': alpha_loss.item(),
                                'alpha_tlogs': alpha_tlogs.item(),
                                },
                               self.niter)
            
        self.niter = self.niter + 1
            
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def update_threat_rl(self, sample, agent_i, train_option="regular", logger=None):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, dones_batch, detection_in_batch, next_detection_in_batch = sample
        state_batch, action_batch, next_state_batch, detection_in_batch, next_detection_in_batch = state_batch[0], action_batch[0], next_state_batch[0], detection_in_batch[0], next_detection_in_batch[0]
        batch_size = state_batch.shape[0]
        state_batch = torch.concat((state_batch, self.threat(detection_in_batch.view(batch_size,-1)).detach()), dim=-1)
        next_state_batch = torch.concat((next_state_batch, self.threat(next_detection_in_batch.view(batch_size,-1)).detach()), dim=-1)
        reward_batch = reward_batch[0].unsqueeze(1)
        mask_batch = (1 - dones_batch[0]).unsqueeze(1)
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            # INFO: double Q network
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # INFO: gradent can pass through policy!
        
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.niter % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'qf1_loss': qf1_loss.item(),
                                'qf2_loss': qf2_loss.item(),
                                'min_qf_pi': min_qf_pi.mean().item(),
                                'alpha': self.alpha.item(),
                                'policy_loss': policy_loss.item(),
                                'log_pi': log_pi.mean(),
                                'alpha_loss': alpha_loss.item(),
                                'alpha_tlogs': alpha_tlogs.item(),
                                },
                               self.niter)
            
        self.niter = self.niter + 1
            
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def update_threat(self, detection_in_sample, detection_out_sample, dones_sample, logger=None):
        loss = self.threat.loss(detection_in_sample, detection_out_sample, dones_sample)
        self.threat_optim.zero_grad()
        loss.backward()
        self.threat_optim.step()
        if logger is not None:
            logger.add_scalars('threat/losses',
                               {'threat_loss': loss.item(),},
                               self.niter)            
        return

    # Save model parameters
    def save(self, ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                    # 'threat_model_dict': self.threat_optim.state_dict()
                    },
                    ckpt_path)

    # Load model parameters
    def init_from_save(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()