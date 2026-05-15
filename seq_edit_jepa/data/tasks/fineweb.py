from __future__ import annotations

from seq_edit_jepa.data.tasks.hf_text import HFTextTask
from seq_edit_jepa.data.tasks.registry import register_task


@register_task("fineweb")
class FineWebTask(HFTextTask):
    default_dataset_name = "HuggingFaceFW/fineweb"
    default_dataset_config = "sample-10BT"
    default_text_field = "text"
