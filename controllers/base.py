class Controller:
    def policy(self, observation, reward, done, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass


class Action:
    def __init__(self, basal=0, bolus=0):
        self.basal = basal
        self.bolus = bolus
