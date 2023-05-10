import numpy as np
import torch
import random
import os
from torch import Tensor
from snntorch import utils

from typing import List, Tuple


def and_generator(size: int) -> List[Tuple[Tensor, Tensor]]:
  x = Tensor(np.random.choice([0, 1], (size, 2)))
  y = Tensor([1 if i[0] and i[1] else 0 for i in x]).reshape(size, 1)

  return list(zip(x, y))

def or_generator(size: int) -> List[Tuple[Tensor, Tensor]]:
  x = Tensor(np.random.choice([0, 1], (size, 2)))
  y = Tensor([1 if i[0] or i[1] else 0 for i in x]).reshape(size, 1)

  return list(zip(x, y))

def xor_generator(size: int) -> List[Tuple[Tensor, Tensor]]:
  x = np.random.choice([0, 1], (size, 2))
  y = Tensor([1 if i[0] ^ i[1] else 0 for i in x]).reshape(size, 1)
  x = Tensor(x)
  return list(zip(x, y))

def continous_and_generator(size: int) -> List[Tuple[Tensor, Tensor]]:
    x = torch.from_numpy(np.random.random(size=(size, 2)).round(2))
    y = torch.from_numpy(np.apply_along_axis(lambda t: t[0] > 0.5 and t[1] > 0.5, 1, x).astype(int)).to(torch.float16)
    return list(zip(x, y))

def continous_or_generator(size: int) -> List[Tuple[Tensor, Tensor]]:
    x = torch.from_numpy(np.random.random(size=(size, 2)).round(2)).to(torch.float32)
    y = torch.from_numpy((~np.apply_along_axis(lambda t: t[0] < 0.5 and t[1] < 0.5, 1, x)).astype(int)).to(torch.float16)
    return list(zip(x, y))

def continous_xor_generator(size: int) -> List[Tuple[Tensor, Tensor]]:
    x = torch.from_numpy(np.random.random(size=(size, 2)).round(2)).to(torch.float32)
    y = torch.from_numpy((~np.apply_along_axis(lambda t: (t[0] < 0.5 and t[1] < 0.5) or t[0] > 0.5 and t[1] > 0.5, 1, x)).astype(int)).to(torch.float16)
    return list(zip(x, y))

def set_seed(seed: int = 42) -> None:
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def forward_pass(net, data, num_steps: int) -> Tuple[Tensor, np.ndarray]:
  spk_rec = []
  mem_hist = []
  utils.reset(net)

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      mem_hist.append(mem_out.cpu().detach().numpy())
      spk_rec.append(spk_out)

  return torch.stack(spk_rec), np.stack(mem_hist)
