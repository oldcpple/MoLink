from typing import List, Optional
import asyncio
import json
from .dht import DHTNode

class PipelineManager():

    def __init__(self, dht: DHTNode):
        self.dht = dht
        self.pipeline_info = {}
        asyncio.create_task(self.run_in_background())
    
    async def manage_pipeline(self):
        await asyncio.sleep(5) # make sure the dht node has finished initialization
        dht_node_list = await self.dht.node.get('node_info')
        if dht_node_list is None:
            return {}
        dht_node_list = json.loads(dht_node_list.decode('utf-8'))
        node_info_dict = {}
        for node_id in dht_node_list:
            node_info = await self.dht.node.get(node_id)
            if node_info is not None:
                node_info = json.loads(node_info.decode('utf-8'))
                node_info = json.loads(node_info)
                ip = node_info.get('ip')
                grpc_port = node_info.get('grpc_port')
                ip = f'{ip}:{grpc_port}'
                start_layer = node_info.get('start_layer')
                node_info_dict.update({ip : start_layer})

        
        sorted_ips = [ip for ip, _ in sorted(node_info_dict.items(), key=lambda item: item[1])]

        pipeline_info = {}
        pipeline_info.update({'head' : f'{self.dht.ip}:{self.dht.node_info.grpc_port}'})
        pipeline_info.update({'server_list' : sorted_ips})
        return pipeline_info
    
    async def run_in_background(self):
        while True:
            self.pipeline_info = await self.manage_pipeline()
            if len(self.pipeline_info) > 0 and len(self.pipeline_info['server_list']) > 1:
                print('Multiple nodes has connected, swarm info: {}'.format(self.pipeline_info))
            await asyncio.sleep(3)