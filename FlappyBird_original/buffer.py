import numpy as np
class ReplayBuffer(object): 
    def __init__(self, max_size,alpha, beta, num_iter):
        super().__init__()
        #self._replay_memory = []
        self._replay_memory = np.empty(max_size, dtype=np.object)
        self._priority = np.zeros(max_size)
        self.alpha = alpha
        self.beta = beta
        self.pointer = 0 
        self.max_p = 1
        self.size = 0
        self.initial_beta = beta
        self.num_iter = num_iter

    def add(self, sample):
        self._priority[self.pointer] = abs(self.max_p)**self.alpha
        '''
        if len(self._replay_memory) == len(self._priority):
            self._replay_memory[self.pointer] = sample
        else:
            self._replay_memory.append(sample)
        '''
        self._replay_memory[self.pointer] = sample
        self.pointer += 1
        self.pointer = self.pointer%len(self._priority)
        self.size += 1
    def update(self, indices, priority):
        self._priority[indices] = abs(priority)**self.alpha
        self.max_p = max(self.max_p, max(priority))

    def sample(self, batch_size):
        size = min(self.size, len(self._replay_memory))
        if size > batch_size:
            p = (self._priority / self._priority.sum())[:size]
            indices = np.random.choice(size, batch_size, p=p,replace=False)
            weight = abs((1/p[indices])*(1/size))**self.beta
            weight = weight/max(weight)
        else:
            indices = [size-1]
            weight = np.array([1])
        replay = self._replay_memory[indices]

        '''
        replay = []
        for i in indices:
            replay.append(self._replay_memory[i])
        '''
        #print(sorted(indices))
        return replay, indices, weight

    def update_beta(self, iter):
        self.beta = 1 + ((self.num_iter - iter) * (self.initial_beta - 1) / self.num_iter)