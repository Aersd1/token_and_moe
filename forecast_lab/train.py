from tslab.config import parse_config
from tslab.engine import train


if __name__ == "__main__":
    train(parse_config())
