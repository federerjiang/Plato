class Args:
    def __init__(self):
        self.lr = 0.0001  # learning rate (default: 0.0001)
        self.gamma = 0.99  # discount factor for rewards (default: 0.99)
        self.tau = 1.00  # parameter for GAE (default: 1.00)
        self.entropy_coef = 0.01  # entropy term coefficient (default: 0.01)
        self.value_loss_coef = 0.5  # value loss coefficient (default: 0.5)
        self.max_grad_norm = 50  #
        self.num_process = 4  # how many training processes to use (default: 4)
        self.num_steps = 20  # number of forward steps in A3C (default: 20)
