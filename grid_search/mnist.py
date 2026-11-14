"""MNIST grid search; see README.md for parameters and device options."""
from pathlib import Path
from search import main

if __name__ == "__main__":
    main("MNIST", Path(__file__).parent / "configs" / "mnist.json")
