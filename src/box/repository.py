import os
from itertools import groupby
from typing import Any

import yaml


class Serializer:
    def serialize(self, entity) -> Any:
        pass

    def deserialize(self, serialized_entity):
        pass


class YamlSerializer(Serializer):
    def serialize(self, entity):
        return yaml.dump(entity)

    def deserialize(self, serialized_entity):
        return yaml.load(serialized_entity)


class DataStore:
    def put(self, entity):
        pass

    def put_all(self, entities):
        pass

    def get(self, id):
        pass

    def get_all(self):
        pass


class BatchRepository:
    def __init__(self, filename, serializer: Serializer = YamlSerializer()):
        self.filename = filename
        self.serializer = serializer
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as handle:
                self.data = self.serializer.deserialize(handle.read())
        else:
            self.data = []

    def save(self, data):
        self.data = data
        with open(self.filename, 'w') as handle:
            handle.write(self.serializer.serialize(self.data))


class Repository:
    def __init__(self, filename, serializer: Serializer = YamlSerializer()):
        self.filename = filename
        self.serializer = serializer
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as handle:
                self.data = self.serializer.deserialize(handle.read())
        else:
            self.data = {}

    def save(self):
        with open(self.filename, 'w') as handle:
            handle.write(yaml.dump(self.data))

    def put(self, id, coupon):
        self.data[id] = coupon
        self.save()

    def get(self, id):
        return self.data.get(id)



if __name__ == '__main__':
    def prints(k, v):
        print(k)
        print(list(v))


    print({prints(key, value) for key, value in groupbyUnsorted([1, 1, 2, 2, 3, 2, 1], lambda x: x)})
