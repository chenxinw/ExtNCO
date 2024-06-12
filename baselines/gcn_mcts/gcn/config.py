import json


class Settings(dict):

    def __init__(self, config_dict):
        super().__init__()
        for key in config_dict:
            self[key] = config_dict[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    __delattr__ = dict.__delitem__


def get_default_config():  # Returns default settings object
    return Settings(json.load(open('baselines/gcn_mcts/gcn/configs/default.json')))


def get_config(filepath):  # Returns settings from json file
    config = get_default_config()
    config.update(Settings(json.load(open(filepath))))
    return config
