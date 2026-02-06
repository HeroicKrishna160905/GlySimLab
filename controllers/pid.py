from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P=1, I=0, D=0, target=140):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time')

        # BG from CGM
        bg = observation.CGM

        control_input = (
            self.P * (bg - self.target)
            + self.I * self.integrated_state
            + self.D * (bg - self.prev_state) / sample_time
        )

        # update states
        self.prev_state = bg
        self.integrated_state += (bg - self.target) * sample_time

        # return insulin action
        action = Action(basal=control_input, bolus=0)
        return action

    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0
