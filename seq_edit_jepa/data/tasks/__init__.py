from seq_edit_jepa.data.tasks.registry import available_tasks, build_task, register_task

# Import modules for registration.
from seq_edit_jepa.data.tasks import fineweb, igsm, lano, lm1b, official_igsm, openthoughts  # noqa: F401

__all__ = ["available_tasks", "build_task", "register_task"]
