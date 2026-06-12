import torch

import minipu_backend


def main() -> None:
    minipu_backend.register()

    print(f"backend: {minipu_backend.BACKEND_NAME}")
    print(f"extension registered: {minipu_backend.is_registered()}")
    print(f"torch.minipu.is_available(): {torch.minipu.is_available()}")

    x = torch.tensor([-2.0, 1.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])

    added = torch.ops.minipu.add(x, y)

    print(f"add: {added.tolist()}")


if __name__ == "__main__":
    main()
