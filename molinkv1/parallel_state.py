"""
MoLink parallel state utilities for cross-node pipeline parallelism.

This module provides utilities to override the default pipeline parallel
layer distribution for cross-node PP using MoLink. When MoLink is enabled,
each physical node runs with pp_size=1 locally, but the model layers
are explicitly distributed across nodes based on the MoLink configuration.
"""

from typing import Optional, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global MoLink parallel state
_MOLINK_ENABLED: bool = False
_MOLINK_START_LAYER: int = 0
_MOLINK_END_LAYER: int = -1  # -1 means use default (all layers)
_MOLINK_NUM_HIDDEN_LAYERS: Optional[int] = None
_MOLINK_IS_FIRST_STAGE: bool = True
_MOLINK_IS_LAST_STAGE: bool = True
_MOLINK_PP_RANK: int = 0
_MOLINK_PP_SIZE: int = 1


def init_molink_parallel_state(
    enabled: bool,
    start_layer: int,
    end_layer: int,
    num_hidden_layers: Optional[int] = None,
    pp_rank: int = 0,
    pp_size: int = 1,
) -> None:
    """Initialize MoLink parallel state.

    This function sets up the global state for MoLink cross-node PP.
    When enabled, it overrides the default layer distribution.

    Args:
        enabled: Whether MoLink is enabled.
        start_layer: The first layer this node handles (inclusive).
        end_layer: The last layer this node handles (exclusive).
                   -1 means use all remaining layers.
        num_hidden_layers: Total number of hidden layers in the model.
        pp_rank: The pipeline parallel rank of this node in the cluster.
        pp_size: The total pipeline parallel size (number of nodes).
    """
    global _MOLINK_ENABLED, _MOLINK_START_LAYER, _MOLINK_END_LAYER
    global _MOLINK_NUM_HIDDEN_LAYERS, _MOLINK_IS_FIRST_STAGE, _MOLINK_IS_LAST_STAGE
    global _MOLINK_PP_RANK, _MOLINK_PP_SIZE

    _MOLINK_ENABLED = enabled
    _MOLINK_START_LAYER = start_layer
    _MOLINK_END_LAYER = end_layer
    _MOLINK_NUM_HIDDEN_LAYERS = num_hidden_layers
    _MOLINK_PP_RANK = pp_rank
    _MOLINK_PP_SIZE = pp_size

    if enabled:
        # Determine if this is first/last stage
        _MOLINK_IS_FIRST_STAGE = start_layer == 0

        if end_layer == -1:
            _MOLINK_IS_LAST_STAGE = True
        elif num_hidden_layers is not None:
            _MOLINK_IS_LAST_STAGE = end_layer >= num_hidden_layers
        else:
            # Conservative: assume last stage if end_layer == -1
            _MOLINK_IS_LAST_STAGE = end_layer == -1

        logger.info(
            f"MoLink parallel state initialized: "
            f"layers {start_layer}-{end_layer}, "
            f"is_first_stage={_MOLINK_IS_FIRST_STAGE}, "
            f"is_last_stage={_MOLINK_IS_LAST_STAGE}, "
            f"pp_rank={pp_rank}, pp_size={pp_size}"
        )


def is_molink_enabled() -> bool:
    """Check if MoLink is enabled."""
    return _MOLINK_ENABLED


def get_molink_layer_range() -> Tuple[int, int]:
    """Get the layer range for this node.

    Returns:
        Tuple of (start_layer, end_layer).
    """
    return _MOLINK_START_LAYER, _MOLINK_END_LAYER


def get_molink_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Tuple[int, int]:
    """Get the layer indices for this node when MoLink is enabled.

    This function overrides the default get_pp_indices behavior when
    MoLink is enabled. Instead of using the pp_rank and pp_size from
    torch distributed, it uses the explicit layer range from MoLink config.

    Args:
        num_hidden_layers: Total number of hidden layers.
        pp_rank: The pipeline parallel rank (ignored when MoLink is enabled).
        pp_size: The pipeline parallel size (ignored when MoLink is enabled).

    Returns:
        Tuple of (start_layer, end_layer).
    """
    if not _MOLINK_ENABLED:
        # Fall back to default behavior
        from vllm.distributed.utils import get_pp_indices as _get_pp_indices

        return _get_pp_indices(num_hidden_layers, pp_rank, pp_size)

    start_layer = _MOLINK_START_LAYER
    end_layer = _MOLINK_END_LAYER

    # Handle special case where end_layer == -1 (use all remaining layers)
    if end_layer == -1:
        end_layer = num_hidden_layers

    # Validate range
    if start_layer < 0:
        start_layer = 0
    if end_layer > num_hidden_layers:
        end_layer = num_hidden_layers
    if start_layer >= end_layer:
        logger.warning(
            f"Invalid layer range: start_layer={start_layer}, "
            f"end_layer={end_layer}. Using default range."
        )
        start_layer = 0
        end_layer = num_hidden_layers

    return start_layer, end_layer


def is_molink_first_stage() -> bool:
    """Check if this node is the first stage in the pipeline."""
    return _MOLINK_IS_FIRST_STAGE


def is_molink_last_stage() -> bool:
    """Check if this node is the last stage in the pipeline."""
    return _MOLINK_IS_LAST_STAGE


def get_molink_pp_rank() -> int:
    """Get the MoLink pipeline parallel rank."""
    return _MOLINK_PP_RANK


def get_molink_pp_size() -> int:
    """Get the MoLink pipeline parallel size."""
    return _MOLINK_PP_SIZE


def update_molink_stage_info(num_hidden_layers: int) -> None:
    """Update MoLink stage information after model config is loaded.

    This should be called once the model's num_hidden_layers is known.

    Args:
        num_hidden_layers: Total number of hidden layers in the model.
    """
    global _MOLINK_NUM_HIDDEN_LAYERS, _MOLINK_IS_LAST_STAGE

    if not _MOLINK_ENABLED:
        return

    _MOLINK_NUM_HIDDEN_LAYERS = num_hidden_layers

    end_layer = _MOLINK_END_LAYER
    if end_layer == -1:
        end_layer = num_hidden_layers

    _MOLINK_IS_LAST_STAGE = end_layer >= num_hidden_layers

    logger.info(
        f"MoLink stage info updated: "
        f"num_hidden_layers={num_hidden_layers}, "
        f"is_last_stage={_MOLINK_IS_LAST_STAGE}"
    )


def destroy_molink_parallel_state() -> None:
    """Destroy MoLink parallel state."""
    global _MOLINK_ENABLED, _MOLINK_START_LAYER, _MOLINK_END_LAYER
    global _MOLINK_NUM_HIDDEN_LAYERS, _MOLINK_IS_FIRST_STAGE, _MOLINK_IS_LAST_STAGE
    global _MOLINK_PP_RANK, _MOLINK_PP_SIZE

    _MOLINK_ENABLED = False
    _MOLINK_START_LAYER = 0
    _MOLINK_END_LAYER = -1
    _MOLINK_NUM_HIDDEN_LAYERS = None
    _MOLINK_IS_FIRST_STAGE = True
    _MOLINK_IS_LAST_STAGE = True
    _MOLINK_PP_RANK = 0
    _MOLINK_PP_SIZE = 1


# ============================================================================
# Patching utilities for integrating with vLLM's distributed module
# ============================================================================

_ORIGINAL_GET_PP_INDICES = None
_PP_GROUP_PATCHED = False


def patch_get_pp_indices() -> None:
    """Patch vllm.distributed.utils.get_pp_indices for MoLink.

    This function patches get_pp_indices to use MoLink's explicit
    layer range when MoLink is enabled. The original function is
    preserved and called when MoLink is not enabled.
    """
    global _ORIGINAL_GET_PP_INDICES

    if _ORIGINAL_GET_PP_INDICES is not None:
        # Already patched
        return

    from vllm.distributed import utils as dist_utils

    _ORIGINAL_GET_PP_INDICES = dist_utils.get_pp_indices

    def patched_get_pp_indices(
        num_hidden_layers: int, pp_rank: int, pp_size: int
    ) -> Tuple[int, int]:
        if is_molink_enabled():
            return get_molink_pp_indices(num_hidden_layers, pp_rank, pp_size)
        return _ORIGINAL_GET_PP_INDICES(num_hidden_layers, pp_rank, pp_size)

    dist_utils.get_pp_indices = patched_get_pp_indices
    logger.info("Patched get_pp_indices for MoLink layer distribution")


def patch_pp_group() -> None:
    """Patch the PP group for MoLink stage info.

    This patches is_first_rank and is_last_rank properties on the
    PP GroupCoordinator to use MoLink's stage information.
    """
    global _PP_GROUP_PATCHED

    if _PP_GROUP_PATCHED:
        return

    from vllm.distributed.parallel_state import _PP, GroupCoordinator

    if _PP is None:
        logger.warning("PP group not initialized, cannot patch for MoLink")
        return

    # Store original property getters
    original_is_first_rank = GroupCoordinator.is_first_rank.fget
    original_is_last_rank = GroupCoordinator.is_last_rank.fget

    def patched_is_first_rank(self) -> bool:
        if is_molink_enabled():
            return is_molink_first_stage()
        return original_is_first_rank(self)

    def patched_is_last_rank(self) -> bool:
        if is_molink_enabled():
            return is_molink_last_stage()
        return original_is_last_rank(self)

    # Replace the properties
    GroupCoordinator.is_first_rank = property(patched_is_first_rank)
    GroupCoordinator.is_last_rank = property(patched_is_last_rank)

    _PP_GROUP_PATCHED = True
    logger.info("Patched PP group is_first_rank/is_last_rank for MoLink")


def apply_molink_patches() -> None:
    """Apply all MoLink patches to vLLM distributed module.

    This should be called after distributed initialization but before
    model loading.
    """
    if not is_molink_enabled():
        return

    patch_get_pp_indices()
    patch_pp_group()

    logger.info("All MoLink patches applied")
