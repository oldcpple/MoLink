from vllm.distributed import GroupCoordinator, init_model_parallel_group, get_world_group, \
                            get_tensor_model_parallel_world_size, model_parallel_is_initialized, \
                            get_pp_group
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)
import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup
import vllm.distributed.parallel_state as P


_ENABLE_CUSTOM_ALL_REDUCE = True
USE_DHT = False
NODE_PORT = ''
IN_AUTODL = False
AUTODL_WORKER_NUM = 0
AUTODL_SERVER_IP_MAP = []

#_TP: Optional[GroupCoordinator] = None
#_PP: Optional[GroupCoordinator] = None

class MolinkGroupCoordinator(GroupCoordinator):
    
    def __init__(self, 
        _is_first_rank: bool, 
        _is_last_rank: bool,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,):

        super().__init__(
                    group_ranks,
                    local_rank,
                    torch_distributed_backend,
                    use_device_communicator,
                    use_message_queue_broadcaster,
                    group_name,)

        self._is_first_rank = _is_first_rank
        self._is_last_rank = _is_last_rank

    @property
    def is_first_rank(self):
        return self._is_first_rank

    @property
    def is_last_rank(self):
        return self._is_last_rank

def init_model_parallel_group_PP(
    _is_first_rank: bool,
    _is_last_rank: bool,
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
) -> MolinkGroupCoordinator:
    return MolinkGroupCoordinator(
        _is_first_rank = _is_first_rank,
        _is_last_rank = _is_last_rank,
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_device_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
    )

def initialize_model_parallel(
    _is_first_rank: bool,
    _is_last_rank: bool,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = tensor_model_parallel_size
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    if (world_size !=
            tensor_model_parallel_size):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = (world_size //
                                             tensor_model_parallel_size)
    #global _TP
    assert P._TP is None, ("tensor model parallel group is already initialized")
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    P._TP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    use_message_queue_broadcaster=True,
                                    group_name="tp")

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = tensor_model_parallel_size
    #global _PP
    assert P._PP is None, (
        "pipeline model parallel group is already initialized")
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    P._PP = init_model_parallel_group_PP(_is_first_rank,
                                    _is_last_rank,
                                    group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="pp")

    assert P._DP is None, ("data parallel group is already initialized")

    P._DP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="dp")


def ensure_model_parallel_initialized(
    _is_first_rank: bool,
    _is_last_rank: bool,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(_is_first_rank,
                                  _is_last_rank,
                                  tensor_model_parallel_size,
                                  pipeline_model_parallel_size, backend)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    pp_world_size = get_pp_group().world_size
    assert (pp_world_size == pipeline_model_parallel_size), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}")
    