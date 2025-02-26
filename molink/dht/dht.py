from kademlia.network import Server
import asyncio
import json
import socket
import uuid
from vllm.logger import init_logger
from .node_info import NodeInfo

logger = init_logger(__name__)

class DHTNode:

    def __init__(self, initial_peer, model_name, start_layer, end_layer):
        # 50051 is the default port of gRPC server
        # but for testing, multiple gRPC servers might be
        # set on the same node
        grpc_port = find_unbind_port(50051)
        dht_port = find_unbind_port(8468)
        self.ip = extract_ip()

        grpc_info = f'{self.ip}:{grpc_port}'
        dht_info = f'{self.ip}:{dht_port}'
        logger.info("GRPC INFO: MoLink server gRPC works at %s", grpc_info)
        logger.info("DHT INFO: MoLink server DHT works at %s", dht_info)
        logger.info("If this is the first node of the swarm, you can copy the DHT INFO as initial peer of the following nodes")

        self.uuid = str(uuid.uuid4())
        self.node_info = NodeInfo(self.ip, self.uuid, dht_port, grpc_port, model_name, start_layer, end_layer)
        self.node = register_node(initial_peer)
        asyncio.create_task(self.refresh_registration())

    async def store_primary_kv(self):
        primary_kv = self.node.get('node_info')
        if primary_kv is None:
            primary_kv = [self.uuid]
            await self.node.set('node_info', primary_kv)
        elif self.uuid not in primary_kv:
            primary_kv.append(self.uuid)
            await self.node.set('node_info', primary_kv)

    async def store_sub_kv(self):
        await self.node.set(self.uuid, self.node_info)

    async def refresh_registration(self):
        while True:
            await self.store_primary_kv()
            await self.store_sub_kv()
            await asyncio.sleep(300)
            
    

async def register_node(initial_peer, port):
    node = Server()
    await node.listen(port)
    # judge
    if initial_peer is None or initial_peer == '':
        peer = []
    else:
        peer_ip, peer_port = initial_peer.split(':')
        peer = (peer_ip, peer_port)
    await node.bootstrap(peer)
    return node


import socket

def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:       
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    
    return IP

def find_unbind_port(start_port):
    IP = extract_ip()
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
                test_sock.bind((IP, port))
                test_sock.close()
            break
        except OSError:
            port += 1
    
    return port
