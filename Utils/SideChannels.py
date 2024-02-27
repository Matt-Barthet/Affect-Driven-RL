import re
from abc import ABC
from uuid import UUID

from mlagents_envs.side_channel import SideChannel, IncomingMessage


class MySideChannel(SideChannel, ABC):

    def __init__(self):
        super().__init__(UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self.score = 0
        self.low_res_state = {}
        self.direction = []
        self.levelEnd = False
        self.arousal_vector = []
        self.race_ended = False
        self.collision = False

    def on_message_received(self, msg: IncomingMessage):
        test = msg.read_string()
        self.levelEnd = False
        if '[Score]' in test:
            self.score = int(re.search(r'\d+', test).group())
        elif '[Low-Resolution State]' in test:
            test = test.removeprefix("[Low-Resolution State]:").split(",")
            self.low_res_state.clear()
            for item in test:
                self.low_res_state.update({item.split(":")[0]: item.split(":")[1]})
        elif '[Direction]' in test:
            vector = test.removeprefix("[Direction]:").split(",")
            self.direction = [float(vector[counter]) for counter in range(3)]
        elif test == 'Level Ended' or test == "Collision":
            self.levelEnd = True
        elif '[Vector]' in test:
            test = test.removeprefix("[Vector]:")
            self.arousal_vector = [float(value) for value in test.split(",")[:-1]]
            # print(self.arousal_vector)

        if 'Race Ended' == test:
            self.race_ended = True
        if 'Collision' in test:
            self.collision = True
        

