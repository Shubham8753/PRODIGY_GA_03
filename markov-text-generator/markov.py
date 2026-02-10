"""Simple Markov chain text generator (word- and char-level).

Usage examples:
  python markov.py --file sample.txt --mode word --order 2 --length 50
  python markov.py --demo

This script provides a `MarkovGenerator` class and a small CLI.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable


class MarkovGenerator:
    def __init__(self, order: int = 1, mode: str = "word"):
        self.order = max(1, int(order))
        if mode not in ("word", "char"):
            raise ValueError("mode must be 'word' or 'char'")
        self.mode = mode
        # model: key(tuple) -> Dict[next_token, count]
        self.model: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _tokenize(self, text: str) -> List[str]:
        if self.mode == "word":
            return text.split()
        # char mode: keep every character
        return list(text)

    def train(self, text: str) -> None:
        tokens = self._tokenize(text)
        if len(tokens) <= self.order:
            return
        for i in range(len(tokens) - self.order):
            key = tuple(tokens[i : i + self.order])
            next_tok = tokens[i + self.order]
            self.model[key][next_tok] += 1

    def train_from_file(self, path: str, encoding: str = "utf-8") -> None:
        with open(path, "r", encoding=encoding) as f:
            self.train(f.read())

    def _choose_next(self, counts: Dict[str, int]) -> str:
        choices = list(counts.keys())
        weights = list(counts.values())
        return random.choices(choices, weights=weights, k=1)[0]

    def generate(self, length: int = 100, start: Iterable[str] | None = None) -> str:
        if not self.model:
            raise RuntimeError("Model is empty. Train the generator before generating.")

        if start is not None:
            start_tokens = list(start)
            if len(start_tokens) < self.order:
                raise ValueError("start must provide at least `order` tokens")
            key = tuple(start_tokens[: self.order])
            if key not in self.model:
                # fallback to a random key
                key = random.choice(list(self.model.keys()))
        else:
            key = random.choice(list(self.model.keys()))

        output = list(key)

        for _ in range(max(0, length - self.order)):
            key_tuple = tuple(output[-self.order :])
            if key_tuple not in self.model:
                break
            next_tok = self._choose_next(self.model[key_tuple])
            output.append(next_tok)

        if self.mode == "word":
            return " ".join(output)
        return "".join(output)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple Markov chain text generator")
    p.add_argument("--file", help="text file to train from", default=None)
    p.add_argument("--text", help="text to train from (alternative to --file)", default=None)
    p.add_argument("--mode", choices=("word", "char"), default="word")
    p.add_argument("--order", type=int, default=1, help="Markov order (n-gram size)")
    p.add_argument("--length", type=int, default=50, help="length of generated output (tokens or chars)")
    p.add_argument("--start", help="starting token(s) (space-separated for word mode)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--demo", action="store_true", help="run a short demo using built-in sample text")
    return p


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # If the user provided no input options, default to the built-in demo
    if not (args.file or args.text or args.demo):
        print("No input provided — running built-in demo.")
        args.demo = True

    if args.seed is not None:
        random.seed(args.seed)

    gen = MarkovGenerator(order=args.order, mode=args.mode)

    if args.demo:
        sample = (
            "Mary had a little lamb\nIts fleece was white as snow.\n"
            "And everywhere that Mary went, the lamb was sure to go."
        )
        gen.train(sample)
        print("--- Demo (trained on tiny nursery rhyme) ---")
        print(gen.generate(length=args.length))
        return

    if args.file:
        gen.train_from_file(args.file)
    elif args.text:
        gen.train(args.text)
    else:
        parser.error("Either --file, --text or --demo must be provided")

    start_tokens = None
    if args.start:
        if args.mode == "word":
            start_tokens = args.start.split()
        else:
            start_tokens = list(args.start)

    out = gen.generate(length=args.length, start=start_tokens)
    print(out)


if __name__ == "__main__":
    main()
