from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorEntry(_message.Message):
    __slots__ = ("key", "tensor_data")
    KEY_FIELD_NUMBER: _ClassVar[int]
    TENSOR_DATA_FIELD_NUMBER: _ClassVar[int]
    key: str
    tensor_data: bytes
    def __init__(self, key: _Optional[str] = ..., tensor_data: _Optional[bytes] = ...) -> None: ...

class IntermediateTensors(_message.Message):
    __slots__ = ("tensors",)
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedCompositeFieldContainer[TensorEntry]
    def __init__(self, tensors: _Optional[_Iterable[_Union[TensorEntry, _Mapping]]] = ...) -> None: ...

class GrpcRequestData(_message.Message):
    __slots__ = ("scheduler_output", "intermediate_tensors", "grpc_metadata", "virtual_engine")
    SCHEDULER_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    INTERMEDIATE_TENSORS_FIELD_NUMBER: _ClassVar[int]
    GRPC_METADATA_FIELD_NUMBER: _ClassVar[int]
    VIRTUAL_ENGINE_FIELD_NUMBER: _ClassVar[int]
    scheduler_output: bytes
    intermediate_tensors: IntermediateTensors
    grpc_metadata: bytes
    virtual_engine: int
    def __init__(self, scheduler_output: _Optional[bytes] = ..., intermediate_tensors: _Optional[_Union[IntermediateTensors, _Mapping]] = ..., grpc_metadata: _Optional[bytes] = ..., virtual_engine: _Optional[int] = ...) -> None: ...

class GrpcResponseData(_message.Message):
    __slots__ = ("res", "error_message")
    RES_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    res: int
    error_message: str
    def __init__(self, res: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class GrpcTriggerRequest(_message.Message):
    __slots__ = ("virtual_engine", "scheduler_output")
    VIRTUAL_ENGINE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    virtual_engine: int
    scheduler_output: bytes
    def __init__(self, virtual_engine: _Optional[int] = ..., scheduler_output: _Optional[bytes] = ...) -> None: ...

class SamplerOutput(_message.Message):
    __slots__ = ("output_data", "virtual_engine")
    OUTPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    VIRTUAL_ENGINE_FIELD_NUMBER: _ClassVar[int]
    output_data: bytes
    virtual_engine: int
    def __init__(self, output_data: _Optional[bytes] = ..., virtual_engine: _Optional[int] = ...) -> None: ...

class NodeInfo(_message.Message):
    __slots__ = ("ip", "start_layer", "end_layer", "pp_rank", "tp_size")
    IP_FIELD_NUMBER: _ClassVar[int]
    START_LAYER_FIELD_NUMBER: _ClassVar[int]
    END_LAYER_FIELD_NUMBER: _ClassVar[int]
    PP_RANK_FIELD_NUMBER: _ClassVar[int]
    TP_SIZE_FIELD_NUMBER: _ClassVar[int]
    ip: str
    start_layer: int
    end_layer: int
    pp_rank: int
    tp_size: int
    def __init__(self, ip: _Optional[str] = ..., start_layer: _Optional[int] = ..., end_layer: _Optional[int] = ..., pp_rank: _Optional[int] = ..., tp_size: _Optional[int] = ...) -> None: ...

class PipelineTopology(_message.Message):
    __slots__ = ("nodes", "total_layers")
    NODES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LAYERS_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[NodeInfo]
    total_layers: int
    def __init__(self, nodes: _Optional[_Iterable[_Union[NodeInfo, _Mapping]]] = ..., total_layers: _Optional[int] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("healthy", "status")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status: str
    def __init__(self, healthy: bool = ..., status: _Optional[str] = ...) -> None: ...

class KVCacheConfigData(_message.Message):
    __slots__ = ("num_gpu_blocks", "num_cpu_blocks", "kv_cache_config")
    NUM_GPU_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    NUM_CPU_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    KV_CACHE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    num_gpu_blocks: int
    num_cpu_blocks: int
    kv_cache_config: bytes
    def __init__(self, num_gpu_blocks: _Optional[int] = ..., num_cpu_blocks: _Optional[int] = ..., kv_cache_config: _Optional[bytes] = ...) -> None: ...

class ModelConfigData(_message.Message):
    __slots__ = ("vllm_config", "pp_rank", "pp_size")
    VLLM_CONFIG_FIELD_NUMBER: _ClassVar[int]
    PP_RANK_FIELD_NUMBER: _ClassVar[int]
    PP_SIZE_FIELD_NUMBER: _ClassVar[int]
    vllm_config: bytes
    pp_rank: int
    pp_size: int
    def __init__(self, vllm_config: _Optional[bytes] = ..., pp_rank: _Optional[int] = ..., pp_size: _Optional[int] = ...) -> None: ...
