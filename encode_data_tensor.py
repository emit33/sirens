from utils.encode import encode_data_tensor
from utils.config import Config

if __name__ == "__main__":
    # Obtain config
    config = Config.from_yaml(
        "/home/tempus/projects/siren_analysis/siren_experiments/02_mnist/config_mnist.yaml"
    )

    # Encode tensor
    encode_data_tensor(config.model, config.training, config.paths)

    print(f"Data tensor successfully encoded at {config.paths.results_dir}")
