import numpy as np

from ars.policy import Policy

class LinearPolicy(Policy):
  def __init__(self, input_size: int, output_size: int, discrete_output=False):
    self.discrete_output = discrete_output
    self.input_size = input_size
    self.output_size = output_size
    self.shape = (input_size, output_size)

    self.param_size = input_size * output_size + output_size

    self.c_W = input_size * output_size

    super().__init__(self.param_size)

  def call(self, params, input):
    W = params[:self.c_W].reshape(self.shape)
    b = params[self.c_W:]
    r = np.matmul(input, W) + b
    if self.discrete_output:
      r = 1 if r[0] > 0.5 else 0
    return r
  
  def init_params(self):
    # return np.array([-0.00191783, 0.08849352, 0.20041636, 0.27868464, 0.48853733], dtype=np.float64)
    return np.zeros(self.param_size, dtype=np.float64)