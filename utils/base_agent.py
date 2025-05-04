class BaseAgent:
    def agent_init(self, agent_info):
        raise NotImplementedError

    def agent_start(self, state):
        raise NotImplementedError

    def agent_step(self, reward, state):
        raise NotImplementedError

    def agent_end(self, reward):
        raise NotImplementedError
