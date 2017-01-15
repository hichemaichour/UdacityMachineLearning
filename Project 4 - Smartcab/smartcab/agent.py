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
        return {'next_waypoint': next_waypoint, 'light': inputs['light'], 'right': inputs['right'], 'oncoming': inputs['oncoming'], 'left': inputs['left']}
        
    def Q_action(self, state):
        actions = []
        best_score = self.Q.get((str(state), None), self.default_Q_value)
        for action in self.env.valid_actions:
            val = self.Q.get((str(state), action), self.default_Q_value)
            if val < best_score - 1e-9:
                # Better action was found before
                continue
            elif val > best_score + 1e-9:
                # This is a better action than the one(s) found before
                actions = [action]
                best_score = val
            else:
                # This action produces the same best_score so far
                actions.append(action)
                
        from random import choice
        return choice(actions)
        
    def explore_exploite_action(self):
        from random import random
        if random() < self.epsilon:
            # Exploration
            # print "LearningAgent.explore_exploite_action(): Exploration" # [debug]
            return self.random_action()
        else:
            # Exploitation
            # print "LearningAgent.explore_exploite_action(): Exploitation" # [debug]
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

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, total_reward = {}".format(deadline, inputs, action, reward, self.total_reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    alpha_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    gamma_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    average_rewards = []
    success_rates = []
    
    for alpha in alpha_values:
        average_rewards_row = []
        success_rates_row = []
        
        for gamma in gamma_values:
            # Set up environment and agent
            e = Environment()  # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
            
            a.alpha = alpha
            a.gamma = gamma

            # Now simulate it
            sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            
            print "LearningAgent.run(): Average Simulation Reward = {}".format(sim.average_reward) # debug
            print "LearningAgent.run(): Simulation Success Rate = {}".format(sim.success_rate) # debug
            
            average_rewards_row.append(sim.average_reward)
            success_rates_row.append(sim.success_rate)
            
            if (alpha == 0.2 and gamma == 0.4):
                Q_optimal = a.Q
            
        average_rewards.append(average_rewards_row)
        success_rates.append(success_rates_row)
        
    import numpy as np
    average_rewards = np.array(average_rewards)
    success_rates = np.array(success_rates)
    np.set_printoptions(precision=3)
    print "run(): Average Rewards:"
    print repr(average_rewards)
    print "run(): Success Rate:"
    print repr(success_rates)
    
    # Print Q table
    for k, v in sorted(Q_optimal.iteritems()):
       print k, "{0:.3f}".format(v)


if __name__ == '__main__':
    run()
