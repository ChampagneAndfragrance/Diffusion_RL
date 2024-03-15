import yaml

def config_loader(path=None):
    with open(path) as f:
        config = yaml.full_load(f)
    return config