from __future__ import annotations

import torch


def generate_from_prompt(model, clean, tokenizer, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = clean.input_ids.shape[0]
    output = torch.full_like(clean.input_ids, int(tokenizer.pad_token_id))
    for row in range(batch_size):
        editable_positions = torch.where(clean.editable_mask[row] & clean.attention_mask[row].bool())[0]
        if editable_positions.numel() == 0:
            first_edit = int(clean.attention_mask[row].sum().item()) - 1
        else:
            first_edit = int(editable_positions[0].item())
        prompt_len = max(1, first_edit)
        prompt = clean.input_ids[row : row + 1, :prompt_len]
        prompt_mask = clean.attention_mask[row : row + 1, :prompt_len]
        generated = model.greedy_generate(
            prompt,
            prompt_mask,
            max_new_tokens=max(1, int(seq_len) - prompt_len),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        copy_len = min(int(seq_len), generated.shape[1])
        output[row, :copy_len] = generated[0, :copy_len]
    output_mask = (output != int(tokenizer.pad_token_id)).long()
    return output, output_mask
