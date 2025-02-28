from kademlia.network import Server
import asyncio
import json
import socket
import uuid
from .node_info import NodeInfo


class DHTNode:

    def __init__(self, initial_peer, model_name, start_layer, end_layer):
        # 50051 is the default port of gRPC server
        # but for testing, multiple gRPC servers might be
        # set on the same node
        grpc_port = find_unbind_port(50051, 'tcp')
        dht_port = find_unbind_port(8468, 'udp')
        self.ip = extract_ip()

        self.uuid = str(uuid.uuid4())
        self.node_info = NodeInfo(self.ip, self.uuid, dht_port, grpc_port, model_name, start_layer, end_layer)
        asyncio.create_task(self.register_node(initial_peer, dht_port))
        asyncio.create_task(self.refresh_registration())

    async def store_primary_kv(self):
        primary_kv = await self.node.get('node_info')
        if primary_kv is None:
            # list object cannot be stored by kademlia
            primary_kv = json.dumps([self.uuid]).encode('utf-8')
            await self.node.set('node_info', primary_kv)
        else:
            primary_kv = json.loads(primary_kv.decode('utf-8'))

            if self.uuid not in primary_kv:
                primary_kv.append(self.uuid)
                primary_kv = json.dumps(primary_kv).encode('utf-8')
                await self.node.set('node_info', primary_kv)

    async def store_sub_kv(self):
        await self.node.set(self.uuid, json.dumps(self.node_info.info_dict).encode('utf-8'))

    async def refresh_registration(self):
        await asyncio.sleep(5)
        while True:
            await self.store_primary_kv()
            await self.store_sub_kv()
            await asyncio.sleep(3)
            
    
    async def register_node(self, initial_peer, port):
        self.node = Server()
        await self.node.listen(port)
        # judge
        if initial_peer is None or initial_peer == '':
            peer = []
        else:
            peer_ip, peer_port = initial_peer.split(':')
            peer = [(peer_ip, int(peer_port))]
        await self.node.bootstrap(peer)


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

def find_unbind_port(start_port, protocol):
    """Find an available port for TCP/UDP on all interfaces."""
    ip = '0.0.0.0'
    port = start_port
    while True:
        try:
            if protocol == 'tcp':
                sock_type = socket.SOCK_STREAM
            elif protocol == 'udp':
                sock_type = socket.SOCK_DGRAM
            else:
                raise ValueError("Protocol must be 'tcp' or 'udp'")

            with socket.socket(socket.AF_INET, sock_type) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((ip, port))
            return port
        except OSError as e:
            print(f"Port {port} ({protocol}) is occupied: {e}")
            port += 1
