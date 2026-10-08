import argparse

from vere.core.llm import available_models, get_llm

def main():
    parser = argparse.ArgumentParser(
        prog="vere", description="Terminal research + coding assistant"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help=f"one of: {', '.join(available_models())}",
    )
    sub = parser.add_subparsers(dest="cmd")
    ask = sub.add_parser("ask", help="one-shot question")
    ask.add_argument("question", nargs="+")
    args = parser.parse_args()

    if args.cmd == "ask":
        llm = get_llm(args.model)
        for chunk in llm.stream(" ".join(args.question)):
            print(chunk.content, end="", flush=True)
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
