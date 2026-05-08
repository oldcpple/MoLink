import signal
import queue
from typing import Any, Callable

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine.core import (
    DPEngineCoreProc,
    EngineCoreProc,
    EngineCoreRequestType,
    EngineShutdownState,
    SignalCallback,
)

logger = init_logger(__name__)


class MolinkEngineCoreProc(EngineCoreProc):

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch MoLink EngineCore busy loop in background process."""

        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = (
                parallel_config.data_parallel_size > 1 or dp_rank > 0
            )
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            decorate_logs()

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = MolinkEngineCoreProc(
                    *args, engine_index=dp_rank, **kwargs
                )

            assert engine_core is not None

            def wakeup_engine():
                engine_core.input_queue.put_nowait(
                    (EngineCoreRequestType.WAKEUP, None)
                )

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()
