class Config:
    def __init__(self):
        self.conv1_train = False
        self.conv2_train = False
        self.conv3_train = False
        self.conv4_train = True
        self.lstm_train = True
        self.critic_linear_train = True
        self.actor_linear_train = True

        self.conv1_reset = False
        self.conv2_reset = False
        self.conv3_reset = False
        self.conv4_reset = False
        self.lstm_reset = False
        self.critic_linear_reset = False
        self.actor_linear_reset = False