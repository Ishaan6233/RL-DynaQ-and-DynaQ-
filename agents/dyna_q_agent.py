import numpy as np
from utils.base_agent import BaseAgent

class DynaQAgent(BaseAgent):
    def agent_init(self, agent_info):
        self.num_states = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.epsilon = agent_info["epsilon"]
        self.step_size = agent_info["step_size"]
        self.gamma = agent_info["discount"]
        self.planning_steps = agent_info["planning_steps"]
        self.rand_generator = np.random.RandomState(agent_info.get("random_seed", 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get("planning_random_seed", 42))
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.model = {}
        self.actions = list(range(self.num_actions))
        self.past_state = None
        self.past_action = None

    def update_model(self, past_state, past_action, state, reward):
        if past_state not in self.model:
            self.model[past_state] = {}
        self.model[past_state][past_action] = (state, reward)

    def planning_step(self):
        for _ in range(self.planning_steps):
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))
            s_prime, r = self.model[s][a]
            if s_prime == -1:
                target = r
            else:
                target = r + self.gamma * np.max(self.q_values[s_prime])
            self.q_values[s, a] += self.step_size * (target - self.q_values[s, a])

    def argmax(self, q_values):
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = [i]
            elif q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state):
        if self.rand_generator.rand() < self.epsilon:
            return self.rand_generator.choice(self.actions)
        else:
            return self.argmax(self.q_values[state])

    def agent_start(self, state):
        action = self.choose_action_egreedy(state)
        self.past_state = state
        self.past_action = action
        return action

    def agent_step(self, reward, state):
        target = reward + self.gamma * np.max(self.q_values[state])
        self.q_values[self.past_state, self.past_action] += self.step_size * (target - self.q_values[self.past_state, self.past_action])
        self.update_model(self.past_state, self.past_action, state, reward)
        self.planning_step()
        action = self.choose_action_egreedy(state)
        self.past_state = state
        self.past_action = action
        return action

    def agent_end(self, reward):
        self.q_values[self.past_state, self.past_action] += self.step_size * (reward - self.q_values[self.past_state, self.past_action])
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.planning_step()
