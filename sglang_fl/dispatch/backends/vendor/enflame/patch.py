import logging

from .patches.supported_devices import patch_supported_devices
from .patches.device_communicators import patch_communicator_hooks
from .patches.flashattention_backend import patch_flashattention_backend
from .patches.vision import patch_vision_flashattention_backend

logger = logging.getLogger(__name__)
_patches_applied = False

def apply_gcu_patches():
    """Apply all Template-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_supported_devices()
    patch_communicator_hooks()
    patch_flashattention_backend()
    patch_vision_flashattention_backend()

apply_gcu_patches()
