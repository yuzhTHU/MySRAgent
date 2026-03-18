# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import os
import yaml

__all__ = ["AttrDict"]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            if isinstance(value, list):
                self[key] = [
                    AttrDict(item) if isinstance(item, dict) else item for item in value
                ]
        self.__dict__ = self

    @classmethod
    def load_yaml(cls, __filepath):
        assert os.path.exists(__filepath), f"File {__filepath} not found!"
        yaml_dict = yaml.load(open(__filepath, "r"), Loader=yaml.FullLoader)
        return cls(yaml_dict)

    @classmethod
    def load_yaml_str(cls, str):
        yaml_dict = yaml.load(str, Loader=yaml.FullLoader)
        return cls(yaml_dict)

    def __or__(self, __other):
        return AttrDict({**self, **__other})

    def __repr__(self) -> str:
        return "AttrDict(" + super().__repr__() + ")"

    def __str__(self) -> str:
        return (
            "{\n"
            + "\n".join(
                [f"  {k}: " + str(v).replace("\n", "\n  ") for k, v in self.items()]
            )
            + "\n}"
        )

    def __getitem__(self, __key):
        if isinstance(__key, int):
            return list(self.values())[__key]
        return super().__getitem__(__key)
