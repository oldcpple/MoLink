from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.utils import (enable_trace_function_call_for_thread,
                        resolve_obj_by_qualname, update_environment_variables)
from molink.worker.worker import MolinkWorker

class MolinkWorkerWrapperBase(WorkerWrapperBase):

    def init_worker(self, all_kwargs: List[Dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        enable_trace_function_call_for_thread(self.vllm_config)

        from vllm import configure_as_vllm_process
        configure_as_vllm_process()

        from vllm.plugins import load_general_plugins
        load_general_plugins()

        '''
        if isinstance(self.vllm_config.parallel_config.worker_cls, str):
            worker_class = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_cls)
        else:
            assert isinstance(self.vllm_config.parallel_config.worker_cls,
                              bytes)
            worker_class = cloudpickle.loads(
                self.vllm_config.parallel_config.worker_cls)
        '''
        self.worker = MolinkWorker(**kwargs)
        assert self.worker is not None