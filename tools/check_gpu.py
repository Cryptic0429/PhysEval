#!/usr/bin/env python3
"""Perform a minimal allocation on every CUDA device visible to PyTorch."""

def main(argv: list[str] | None = None) -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    import torch
    count = torch.cuda.device_count()
    print(f"visible_devices={count}")
    for index in range(count):
        device = torch.device(f"cuda:{index}")
        value = torch.zeros(1, device=device).cpu().item()
        print(f"cuda:{index}\t{torch.cuda.get_device_name(index)}\tallocation={value}")


if __name__ == "__main__":
    main()
