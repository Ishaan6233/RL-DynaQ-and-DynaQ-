import numpy as np
from utils.base_agent import BaseAgent

class DynaQPlusAgent(BaseAgent):

    def agent_init(self, agent_info):
        self.num_states = agent_info["num_states"]
        self.num_actions = agent_info["num_actions"]
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)
        self.kappa = agent_info.get("kappa", 0.001)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.tau = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {}

    def update_model(self, past_state, past_action, state, reward):
        if past_state not in self.model:
            self.model[past_state] = {past_action: (state, reward)}
            for action in self.actions:
                if action != past_action:
                    self.model[past_state][action] = (past_state, 0)
        else:
            self.model[past_state][past_action] = (state, reward)

    def planning_step(self):
        for _ in range(self.planning_steps):
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))
            next_state, reward = self.model[s][a]
            bonus = self.kappa * np.sqrt(self.tau[s][a])
            reward += bonus
            if next_state == -1:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_values[next_state])
            self.q_values[s][a] += self.step_size * (target - self.q_values[s][a])

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
        self.tau += 1
        self.tau[self.past_state][self.past_action] = 0

        target = reward + self.gamma * np.max(self.q_values[state])
        self.q_values[self.past_state][self.past_action] += self.step_size * (target - self.q_values[self.past_state][self.past_action])

        self.update_model(self.past_state, self.past_action, state, reward)
        self.planning_step()

        action = self.choose_action_egreedy(state)
        self.past_state = state
        self.past_action = action
        return action

    def agent_end(self, reward):
        self.tau += 1
        self.tau[self.past_state][self.past_action] = 0

        self.q_values[self.past_state][self.past_action] += self.step_size * (reward - self.q_values[self.past_state][self.past_action])

        self.update_model(self.past_state, self.past_action, -1, reward)
        self.planning_step()
