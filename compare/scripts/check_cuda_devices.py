#!/usr/bin/env python3
"""Perform a minimal allocation on every CUDA device visible to PyTorch."""

import torch


def main() -> None:
    count = torch.cuda.device_count()
    print(f"visible_devices={count}")
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        value = torch.zeros(1, device=device).cpu().item()
        print(f"cuda:{index}\t{torch.cuda.get_device_name(index)}\tallocation={value}")


if __name__ == "__main__":
    main()
