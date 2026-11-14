"""FashionMNIST grid search; see README.md for parameters and device options."""
from pathlib import Path
from search import main

if __name__ == "__main__":
    main("FashionMNIST", Path(__file__).parent / "configs" / "fashion_mnist.json")
