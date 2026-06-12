import torch

import minipu_backend


def main() -> None:
    minipu_backend.register()

    print(f"backend: {minipu_backend.BACKEND_NAME}")
    print(f"extension registered: {minipu_backend.is_registered()}")
    print(f"torch.minipu.is_available(): {torch.minipu.is_available()}")

    try:
        x = torch.ones(4).minipu()
        y = x + 1
    except Exception as exc:
        print("tensor smoke skipped:")
        print(f"  {type(exc).__name__}: {exc}")
    else:
        print(f"tensor device: {y.device}")
        print(f"tensor value: {y.cpu().tolist()}")


if __name__ == "__main__":
    main()
