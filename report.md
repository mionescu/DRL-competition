
# Continuous Control: Report

### 1. Introduction

This file provides a description of the implementation of 2 competitive agents that can play tennis. The report describes the learning algorithm, the choice of hyperparameters, the neural networks architecure and shows the performance of the jointly trained agents as the evolution of the rewards it obtaines over multiple episodic runs.

### 2. Implementation

#### 2.1 Learning Algorithm
The learning algorihtm is based on the Multi-Agent Deep Deterministic Policy Gradient (DDPG) approach, where the hyperparameters have been adapted specifically to this problem.

Each agent has its own set of four networks, according to the DDPG approach: a local and a target network for the actor and a local and a target network for the critic. The actor network is based on the policy-gradient method so it maps a state into an action that during training should converge towards the optimum action for that state. The critic network is based on the Q-learning appoach and it computes the action-value function that should converge towards the optimum action-value function. In this implementation, the networks' architecture is described in the Network Architecture subsection below. 

Each agent receives its own local observation and adds its experience to a replay buffer that is shared between them. Each agent's respective actor and critic networks are updated after each time-step using BATCH_SIZE different samples from the replay buffer of capacity BUFFER_SIZE. 

Since the target policy of the actor is deterministic, in order to make the agents try more options, noise is added to each action. This introduces an element of exploration in the action space (which is continuous, not discrete). The noise is generated as a Ornstein–Uhlenbeck process (which is time-correlated). 

*Local critic network update*
The target actor network predicts the optimal action, which is subsequently used by the target critic network to predict the state-action value function. The critic uses all the information available from both agents (i.e. states and actions). Therefore the collaborative aspect of the implementation is that each agent learns from the experience of the other agent.
An update on the local critic network is achieved through the computation of the mean-squared Bellman error (MSBE) between the two respective Q-functions (local and target) of the critic. The optimizer of choice is Adam with a learning rate of LR_CRITIC.

Update policy and value parameters using given batch of experience tuples: (state, action, reward, next_state, done)
Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) * (1 - done)
where:
  actor_target(state) -> action
  critic_target(state, action) -> Q-value
  
The Mean-squared Bellman error is defined by the Expectation[ (Q_locals - Q_targets)^2 ]


*Local actor network update*
The actor network learns in the basis of the local critic network through gradient ascent with a learning rate of LR_ACTOR. The local actor network predicts a best action *a* to take for a given state *s*, for the states of both agents. The predicted actions are then used as an input to compute the total expected rewards estimated using the local critic Q-value function. The negative of the total reward constitute the actor "loss" (hence performing gradient acent on the total reward).

* Target networks update*
The update of the target networks is performed via a parameter TAU such that: target_params = TAU*train_params + (1-TAU)*target_params.


#### 2.2 Network Architecture

The neural network (NN) models were design follows:

a) The actor networks have an input size of 24 (corresponding to the dimension of the state space) and output size of 2 (corresponding to the dimension of the action space). The activation functions between layers are all ReLU. The output layer of the actor network is activated by *tanh* as it outputs values between -1 and 1 corresponding to the action space boundaries. 

b) The critic networks have input size corresponding to the state space and action space of both agents, so 52 dimensions.
The output layer of the critic network is unidimentional and linear (no activation function requried) as it outputs the total predicted reward for a given state and action.

#### 2.3 Hyperparameters Optimization

The list of hyperparameters of the current commit is as follows:

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay


### 3. Plot of Rewards

The following plot shows the joint training evoluiton of the 2 agents, over multiple episodes, recording the winning score each time. The agents learn by episode 3476 how to obtain an average reward (over 100 episodes) of at least +0.5. 

![image](https://github.com/mionescu/udacity-competition/blob/master/rewards_plot_v2.png)
*Rewards evolution over multiple episodes*


### 5. Ideas for Future Work

To improve the agents' performance, some ideas for future work are to try using other network architectures (i.e. layers and activation functions). Montre Carlo Tree Search could be an interesting for a future implementation! 

