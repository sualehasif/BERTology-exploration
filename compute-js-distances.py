
"""Computes the average Jenson-Shannon Divergence between attention heads."""

import argparse
import numpy as np
import torch
import time

def logged_loop(iterable, n=None, **kwargs):
  if n is None:
    n = len(iterable)
  ll = LoopLogger(n, **kwargs)
  for i, elem in enumerate(iterable):
    ll.update(i + 1)
    yield elem


class LoopLogger(object):
  """Class for printing out progress/ETA for a loop."""

  def __init__(self, max_value=None, step_size=1, n_steps=25, print_time=True):
    self.max_value = max_value
    if n_steps is not None:
      self.step_size = max(1, max_value // n_steps)
    else:
      self.step_size = step_size
    self.print_time = print_time
    self.n = 0
    self.start_time = time.time()

  def step(self, values=None):
    self.update(self.n + 1, values)

  def update(self, i, values=None):
    self.n = i
    if self.n % self.step_size == 0 or self.n == self.max_value:
      if self.max_value is None:
        msg = 'On item ' + str(self.n)
      else:
        msg = '{:}/{:} = {:.1f}%'.format(self.n, self.max_value,
                                         100.0 * self.n / self.max_value)
        if self.print_time:
          time_elapsed = time.time() - self.start_time
          time_per_step = time_elapsed / self.n
          msg += ', ELAPSED: {:.1f}s'.format(time_elapsed)
          msg += ', ETA: {:.1f}s'.format((self.max_value - self.n)
                                         * time_per_step)
      if values is not None:
        for k, v in values:
          msg += ' - ' + str(k) + ': ' + ('{:.4f}'.format(v)
                                          if isinstance(v, float) else str(v))
      print(msg)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--attn-data-file", required=True,
    help="Pickle file containing extracted attention maps.")
  parser.add_argument("--outfile", required=True,
                      help="Where to write out the distances between heads.")
  args = parser.parse_args()

  print("Loading attention data")
  data = torch.load(args.attn_data_file)

  print("Computing head distances")
  js_distances = np.zeros([144, 144])
  for doc in logged_loop(data, n_steps=None):
    if "attn" not in doc:
      continue
    tokens, attns = doc["tokens"], doc["attn"].detach().numpy()

    attns_flat = attns.reshape([144, attns.shape[2], attns.shape[3]])
    for head in range(144):
      head_attns = np.expand_dims(attns_flat[head], 0)
      head_attns_smoothed = (0.001 / head_attns.shape[1]) + (head_attns * 0.999)
      attns_flat_smoothed = (0.001 / attns_flat.shape[1]) + (attns_flat * 0.999)
      m = (head_attns_smoothed + attns_flat_smoothed) / 2
      js = -head_attns_smoothed * np.log(m / head_attns_smoothed)
      js += -attns_flat_smoothed * np.log(m / attns_flat_smoothed)
      js /= 2
      js = js.sum(-1).sum(-1)
      js_distances[head] += js

    torch.save(js_distances, args.outfile)


if __name__ == "__main__":
  main()