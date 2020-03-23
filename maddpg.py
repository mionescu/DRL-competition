from ddpg_agent import DDPGAgent
from buffer import ReplayBuffer
import numpy as np

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size



class MADDPG:
    def __init__(self,num_agents=2, random_seed=1): #np.random.randint(1000)
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(24, 16, 8, 2, 52, 42, 24, random_seed), 
                             DDPGAgent(24, 16, 8, 2, 52, 42, 24, random_seed)]
        
        self.num_agents = num_agents


        # Replay memory
        action_size=2
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def act(self, obs_all_agents, noise_ampl=1):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise_ampl) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions
    
    def add_memory(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.num_agents = self.num_agents
        self.memory.add(state, action, reward, next_state, done)
      
    def step(self):
        """Save experience in replay memory, and use random sample from buffer to learn."""
       
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            
            for n in range(0,self.num_agents):
            
                experiences = self.memory.sample()
   
                self.maddpg_agent[n].step(experiences)
                
                
    def reset(self):
        for n in range(0,self.num_agents):
            self.maddpg_agent[n].reset()
        
        
        


