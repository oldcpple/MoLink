import multiprocessing as mp

class RequestTracker:
    def __init__(self, pipeline_parallel_size):
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_data = [{
            'intermediate_tensors_cpu': [],
            'execute_model_req': [],
            'grpc_metadata': [],
            'virtual_engine': []
        }] * pipeline_parallel_size
        self.stack = []
        self.MAX_STACK_SIZE = self.pipeline_parallel_size * 10

    def update(self, intermediate_tensors_cpu, execute_model_req, grpc_metadata, virtual_engine):
        """
        update pipeline data
        """

        self.pipeline_data[virtual_engine]['intermediate_tensors_cpu'] = intermediate_tensors_cpu
        self.pipeline_data[virtual_engine]['grpc_metadata'] = grpc_metadata
        self.pipeline_data[virtual_engine]['execute_model_req'] = execute_model_req
        self.pipeline_data[virtual_engine]['virtual_engine'] = virtual_engine

        self.stack.append(virtual_engine)
        print(f"count: microbatch: {virtual_engine}, request: {self.count_requests(execute_model_req)}")

        if len(self.stack) > self.MAX_STACK_SIZE:
            self.stack = self.stack[-self.MAX_STACK_SIZE:]


    def count_requests(self, execute_model_req):
        """
        count requests in execute_model_req
        """
        return str(execute_model_req).count('SequenceGroupMetadata')
    
    def get_current_vm(self):
        unique_virtual_engines = set()
        return_arr = []
        for i in range(self.pipeline_parallel_size - 1, -1, -1):
            if i not in unique_virtual_engines:
                unique_virtual_engines.add(i)
                return_arr.append(i)
            else:
                break
        return return_arr


    def get_data(self) -> list:
        unique_vm = self.get_current_vm()
        # print(unique_vm)
        unique_pipelines = []
        for vm in unique_vm:
            unique_pipelines.append(self.pipeline_data[vm])

        return unique_pipelines
    

class ServerFailureError(Exception):
    """Custom exception for server failure."""
    def __init__(self, *args):
        super().__init__(*args)
        # print('error init!', flush=True)
        # print('server failure error init!!\n\n', flush=True)


class MoLinkEvent:
    def __init__(self):
        self.event = mp.Event()
        self.message_queue = mp.Queue()

    def set_message(self, message):
        self.message_queue.put(message)

    def get_message(self):
        if not self.message_queue.empty():
            return self.message_queue.get()
        return None

    def set(self, message=None):
        if message:
            self.set_message(message)
        self.event.set()

    def clear(self):
        self.event.clear()

    def wait(self):
        self.event.wait()

    def is_set(self):
        return self.event.is_set()