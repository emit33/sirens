from utils.encode_dir import encode_dir
from utils.config import Config

if __name__ == "__main__":
    # Obtain config
    config = Config.from_yaml("/home/tempus/projects/sirens/config_testing.yaml")

    # Encode directory
    encode_dir(config.model, config.training, config.paths)

    print(f"Directory successfully encoded at {config.paths.results_dir}")
