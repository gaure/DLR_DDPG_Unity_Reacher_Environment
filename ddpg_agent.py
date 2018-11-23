from actor import Actor
from critic import Critic
from utils import OUNoise
from utils import ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentDDPG:

    def __init__(self, state_size, action_size, seed):
        """

        :state_size: size of the state vector
        :action_size: size of the action vector
        """

        self.state_size = state_size
        self.action_size = action_size
        self.t_step = 0
        self.score = 0.0
        self.best = 0.0
        self.seed = seed
        self.total_reward = 0.0
        self.count = 0
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001
        self.batch_size = 64
        self.update_every = 10

        # Instances of the policy function or actor and the value function or critic
        # Actor critic with Advantage

        # Actor local and target network definitions
        self.actor_local = Actor(self.state_size,
                                 self.action_size,
                                 self.seed).to(device)

        self.actor_target = Actor(self.state_size,
                                  self.action_size,
                                  self.seed).to(device)

        # Critic local and target
        self.critic_local = Critic(self.state_size,
                                   self.action_size,
                                   self.seed).to(device)

        self.critic_target = Critic(self.state_size,
                                    self.action_size,
                                    self.seed).to(device)
        # Actor Optimizer
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)

        # Critic Optimizer
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic)

        # Make sure local and target start with the same weights
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Initialize the Gaussin Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2 
        self.noise = OUNoise(self.action_size,
                             self.exploration_mu,
                             self.exploration_theta,
                             self.exploration_sigma)

        # Initialize the Replay Memory
        self.buffer_size = 100000
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Parameters for the Algorithm
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Soft update for target parameters Actor Critic with Advantage

    # Actor interact with the environment through the step
    def step(self, state, action, reward, next_state, done):
        # Add to the total reward the reward of this time step
        self.total_reward += reward
        # Increase your count based on the number of rewards
        # received in the episode
        self.count += 1
        # Stored experience tuple in the replay buffer
        self.memory.add(state,
                        action,
                        reward,
                        next_state,
                        done)

        # Learn every update_times time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            # Check to see if you have enough to produce a batch
            # and learn from it

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                # Train the networks using the experiences
                self.learn(experiences)

        # Roll over last state action (not needed)
        # self.last_state = next_state

    # Actor determines what to do based on the policy
    def act(self, state):
        # Given a state return the action recommended by the policy
        # Reshape the state to fit the torch tensor input
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Pass the state to the actor local model to get an action
        # recommend for the policy in a state
        # set the actor_local model to predict not to train
        self.actor_local.eval()
        # set the model so this operation is not counted in the
        # gradiant calculation.
        with torch.no_grad():
            actions = self.actor_local(state)
        # set the model back to training mode
        self.actor_local.train()

        # Because we are exploring we add some noise to the
        # action vector
        return list(actions.detach().numpy().reshape(4,) + self.noise.sample())

    # This is the Actor learning logic called when the agent
    # take a step to learn
    def learn(self, experiences):
        """
        Learning means that the networks parameters needs to be updated
        Using the experineces batch.
        Network learns from experiences not form interaction with the
        environment
        """
            
        # Reshape the experience tuples in separate arrays of states, actions
        # rewards, next_state, done
        # Your are converting every member of the tuple in a column or vector
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Now reshape the numpy arrays for states, actions and next_states to torch tensors
        # rewards and dones does not need to be tensors.
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)
        actions = torch.from_numpy(actions).float().unsqueeze(0).to(device)
        next_states = torch.from_numpy(next_states).float().unsqueeze(0).to(device)

        # Firs we pass a batch of next states to the actor so it tell us what actions
        # to execute, we use the actor target network instead of the actor local network
        # because of the advantage principle

        # set the target network to predict because this is not part of the training, this model
        # weights are alter by a soft update not by an optimizer
        self.actor_target.eval()
        with torch.no_grad():
            next_state_actions = self.actor_target(next_states).detach()
        self.actor_target.train()

        # The critic evaluates the actions taking by the actor in the next state and generates the
        # Q(a,s) value of the next state taking those actions. These action, next_state tuple comes from the
        # ReplayBuffer not from interacting with the environment.
        # Remember the Critic or q_value function inputs is states, actions
        # We calculate the q_targets of the next state. We will use this to calculate the current
        # state q_value using the bellman equation.

        # set the target network to predict because this is not part of the training, this model
        # weights are alter by a soft update not by an optimizer
        self.critic_target.eval()
        with torch.no_grad():
            q_targets_next_state_action_values = self.critic_target(next_states, next_state_actions).detach()
        self.actor_target.train()

        # With the next state q_value that is a vector of action values Q(s,a) of a random selected
        # next_states from the replay buffer. We calculate the CURRENT state target Q(s,a).
        # using the TD one-step Sarsa equations and the q_target_next value we got from the critic_target net
        # We make terminal states target Q(s,a) 0 and Non terminal the Q_targtes value
        # This is done to train the critic_local model in a supervise learning fashion, this is the target values.
        q_targets = torch.from_numpy(rewards + self.gamma * q_targets_next_state_action_values.numpy() *
                                     (1 - dones)).float()

        # Using the critic_local network and the current state, I calculated the expected q_value or
        # which is the same the q_value of the current state. The different between the q_expected and
        # the q_target calculated with the bellman equation is our error signal.
        # state q_value using the bellman equation.

        # --- Optimize the local Critic Model ----#

        # Here we start the supervise training process of the critic_local network
        # we pass a bunch of states actions samples it produces the expected output
        # q_value of each action we passed. We also need the gradient of the expected
        # q_value with respect to the actions of this model to used it in the calculation
        # of the local_critic optimization as part of the loss signal; this why we set the
        # actions vector to requires_grad
        actions.requires_grad = True
        q_expected = self.critic_local(states, actions)

        # Clear grad buffer values in preparation.
        self.critic_optimizer.zero_grad()

        # loss function for the critic_local model mean square of the difference
        # between the q_expected value and the q_target value.
        critic_loss = F.smooth_l1_loss(q_expected, q_targets)
        critic_loss.backward(retain_graph=True)

        # clip the critic parameters before optimize them
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        # optimize the critic_local model using the optimizer defined for the critic
        # In the init function of this class
        self.critic_optimizer.step()

        # collect the gradients of the critic actions w.r.t the q_expected value
        # this tells how much the actions magnitud is with respect to the q_value
        critic_action_gradients = torch.autograd.grad(q_expected, actions, grad_outputs=torch.ones(q_expected.shape),
                                                      allow_unused=True)

        # --- Optimize the local Actor Model ---#

        # This is as per David Silver Deterministic Policy Gradient Algorithms paper
        # The policy improvement may be decomposed into the "gradient of the action-
        # value with respect to actions(critic_action_gradient), and the
        # "gradient of the policy with respect to the policy parameters".

        # Set the model gradients to zero in preparation
        self.actor_optimizer.zero_grad()

        # Create a dictionary with the actor local parameter
        # indexed by layer
        parameter_to_layer = defaultdict(list)
        for name, w in self.actor_local.named_parameters():
            parameter_to_layer[name] = w

        # Get the actor actions using the experience buffer states
        actor_actions = self.actor_local(states)

        # Calculate the merging gradient of the actor_local model w.r.t the model parameters and
        # add the negative of the critic actions gradient w.r.t the q_expected value; do this one layer
        # at a time.
        for layer in parameter_to_layer.keys():
            merged_gradients = torch.autograd.grad(actor_actions, parameter_to_layer[layer],
                                                   grad_outputs=- critic_action_gradients[0],
                                                   retain_graph=True)[0]
            for l, w in self.actor_local.named_parameters():
                if l == layer:
                    w.grad = merged_gradients

        # optimize the actor_local model using the optimizer defined for the actor
        # In the init function of this class
        self.actor_optimizer.step()

        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_episode_score(self):
        """
        Calculate the episode scores
        :return: None
        """
        # Update score and best score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best:
            self.best = self.score

    def save_model_weights(self, actor_local):
        torch.save(actor_local.state_dict(), './checkpoints.pkl')

