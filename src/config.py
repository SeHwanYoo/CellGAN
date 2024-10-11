import yaml


class Config(dict):
    def __init__(self, filename, mode):
        super().__init__()
        config_file = "./configs/{:s}.yaml".format(filename)
        with open(config_file, 'r') as f:
            # self._yaml = f.read()
            # self._dict = yaml.safe_load(self._yaml)
            self._dict = yaml.safe_load(f)
            self._dict['MODE'] = mode

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        else:
            return None
        
    def update_config(self, updates):
        self._dict.update(updates)

    # def save_config(self):
    #     with open(self.config_file, 'w') as f:
    #         yaml.safe_dump(self._dict, f)

    def print_info(self):
        
        print('Model configurations:')
        print('---------------------------------')
        # print(self._yaml)
        print(yaml.dump(self._dict))
        print('')
        print('---------------------------------\n')
