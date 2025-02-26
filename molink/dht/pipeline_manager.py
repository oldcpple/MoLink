from typing import List, Optional
import asyncio
from .dht import DHTNode

class PipelineManager():

    def __init__(self, dht: DHTNode):
        self.dht = dht
        self.pipeline_info = {}
        asyncio.create_task(self.run_in_background())
    
    async def manage_pipeline(self):
        dht_node_list = await self.dht.node.get('node_info')
        #remote_sequence_info = remote_sequence_info.value
        node_info_dict = {}
        for node_id in dht_node_list:
            node_info = await self.dht.node.get(node_id)
            ip = node_info.get('ip')
            grpc_port = node_info.get('grpc_port')
            ip = f'{ip}:{grpc_port}'
            start_layer = node_info.get('start_layer')
            node_info_dict.update({ip : start_layer})

        
        sorted_ips = [ip for ip, _ in sorted(node_info_dict.items(), key=lambda item: item[1])]

        pipeline_info = {}
        pipeline_info.update({'head' : self.dht.ip})
        pipeline_info.update({'server_list' : sorted_ips})
        return pipeline_info
    
    async def run_in_background(self):
        while True:
            self.pipeline_info = await self.manage_pipeline()
            await asyncio.sleep(3)