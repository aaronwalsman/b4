import pickle

class SolvedAgent:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def play(self, state):
        policy = self.data['p'][state]
        return policy
