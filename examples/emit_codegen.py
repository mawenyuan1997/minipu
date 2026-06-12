import minipu_backend


def main() -> None:
    out_dir = minipu_backend.emit_demo_codegen()
    print(f"generated code written to: {out_dir}")


if __name__ == "__main__":
    main()
