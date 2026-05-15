from __future__ import annotations

from seq_edit_jepa.data.tasks.hf_text import HFTextTask
from seq_edit_jepa.data.tasks.registry import register_task


@register_task("openthoughts")
class OpenThoughtsTask(HFTextTask):
    default_dataset_name = "open-thoughts/OpenThoughts3-1.2M"
    default_text_field = "conversations"
