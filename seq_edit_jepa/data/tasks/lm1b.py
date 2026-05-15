from __future__ import annotations

from seq_edit_jepa.data.tasks.hf_text import HFTextTask
from seq_edit_jepa.data.tasks.registry import register_task


@register_task("lm1b")
class LM1BTask(HFTextTask):
    default_dataset_name = "dvruette/lm1b"
    default_text_field = "text"
