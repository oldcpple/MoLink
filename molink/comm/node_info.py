import json

class NodeInfo:
    def __init__(self, ip, uuid, dht_port, grpc_port, model_name, start_layer, end_layer):
        self.ip = ip
        self.uuid = uuid
        self.dht_port = dht_port
        self.grpc_port = grpc_port
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.info_dict = json.dumps({
            "ip": self.ip,
            "uuid": self.uuid,
            "dht_port": self.dht_port,
            "grpc_port": grpc_port,
            "model_name": self.model_name,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
        })
