import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.total_reward = 0.0
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.1
        self.Q = {}
        self.default_Q_value = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward = 0.0

    def random_action(self):
        from random import choice
        return choice(self.env.valid_actions)
        
    def eval_state(self, next_waypoint, inputs):
        legal_right = inputs['light'] == 'green' or inputs['left'] != 'forward'
        legal_forward = inputs['light'] == 'green'
        legal_left = inputs['light'] == 'green' and inputs['oncoming'] != 'right' and inputs['oncoming'] != 'forward'
        return {'next_waypoint': next_waypoint, 'legal_right': legal_right, 'legal_forward': legal_forward, 'legal_left': legal_left}
        
    def Q_action(self, state):
        actions = []
        best_score = self.Q.get((str(state), None), self.default_Q_value)
        for action in self.env.valid_actions:
            val = self.Q.get((str(state), action), self.default_Q_value)
            if val < best_score - 1e-9:
                continue
            elif val > best_score + 1e-9:
                actions = [action]
                best_score = val
            else:
                actions.append(action)
                
        from random import choice
        return choice(actions)
        
    def explore_exploite_action(self):
        from random import random
        if random() < self.epsilon:
            # Exploration
            print "LearningAgent.explore_exploite_action(): Exploration" # [debug]
            return self.random_action()
        else:
            # Exploitation
            print "LearningAgent.explore_exploite_action(): Exploitation" # [debug]
            return self.Q_action(self.state)
            
    def bellman(self, state, action, reward):
        new_next_waypoint = self.planner.next_waypoint()
        new_inputs = self.env.sense(self)
        new_state = self.eval_state(new_next_waypoint, new_inputs)
        new_action = self.Q_action(new_state)
        
        old_val = self.Q.get((str(state), action), self.default_Q_value)
        update_val = reward + self.gamma * self.Q.get((str(new_state), new_action), self.default_Q_value)
        
        return (1 - self.alpha) * old_val + self.alpha * update_val

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.eval_state(self.next_waypoint, inputs)
        
        # TODO: Select action according to your policy
        action = self.explore_exploite_action()

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        # TODO: Learn policy based on state, action, reward
        self.Q[(str(self.state), action)] = self.bellman(self.state, action, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, total_reward = {}".format(deadline, inputs, action, reward, self.total_reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
