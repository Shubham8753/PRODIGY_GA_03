# Markov Text Generator

Simple, single-file Markov chain text generator (word- and char-level).

Requirements
- Python 3.8+

Quick start

Train on a file and generate text (word mode, 2nd-order):

```bash
python markov.py --file sample.txt --mode word --order 2 --length 50 --seed 1
```

Run the built-in demo:

```bash
python markov.py --demo
```

On Windows you can run the bundled `run_demo.bat`.

Files
- `markov.py` — main script
- `sample.txt` — small sample training text
- `run_demo.bat` — Windows demo runner
