"""Print reproducible CPU layout diagnostics; this script is not a benchmark."""

import json
import platform

import torch

from layout_answers import pack_logical, storage_index
from torch_answers import merge_heads, split_heads


def describe(tensor: torch.Tensor, base: torch.Tensor) -> dict[str, object]:
    """Describe simultaneously live, nonempty CPU tensors used in this lab."""
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "offset_elements": tensor.storage_offset(),
        "default_contiguous": tensor.is_contiguous(),
        "same_live_storage_as_base": (
            tensor.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        ),
        "logical_flatten": tensor.reshape(-1).tolist(),
    }


def main() -> None:
    result: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "device_used": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "purpose": "semantics only; no latency or throughput claims",
        }
    }
    x = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    cases = {
        "base": x,
        "transpose": x.t(),
        "transpose_contiguous": x.t().contiguous(),
        "step_slice": x[:, ::2],
        "crop": x[:, 1:3],
        "offset_contiguous": x[1:],
        "same_shape_other_layout": x.t().contiguous().t(),
        "transpose_clone": x.t().clone(),
    }
    result["layouts"] = {name: describe(t, x) for name, t in cases.items()}
    view_results = {}
    for name, tensor in cases.items():
        try:
            flat = tensor.view(-1)
        except RuntimeError as error:
            view_results[name] = {"success": False, "error": str(error)}
        else:
            view_results[name] = {"success": True, "stride": list(flat.stride())}
    result["view_flatten"] = view_results
    result["exercise_1"] = {
        "transpose_index_2_1": storage_index((4, 3), (1, 4), (2, 1)),
        "packed_transpose": pack_logical(list(range(12)), (4, 3), (1, 4)),
    }
    tokens = torch.arange(24).reshape(2, 3, 4)
    heads_view = split_heads(tokens, 2)
    heads_packed = heads_view.contiguous()
    merged_view = merge_heads(heads_view)
    merged_packed = merge_heads(heads_packed)
    assert torch.equal(merged_view, tokens)
    assert torch.equal(merged_packed, tokens)
    result["head_layouts"] = {
        "split_view": describe(heads_view, tokens),
        "packed_bhsd": describe(heads_packed, tokens),
        "merge_split_view": describe(merged_view, heads_view),
        "merge_packed_bhsd": describe(merged_packed, heads_packed),
    }
    image = torch.arange(120).reshape(2, 3, 4, 5)
    channels_last = image.contiguous(memory_format=torch.channels_last)
    result["channels_last"] = {
        "shape": list(channels_last.shape),
        "stride": list(channels_last.stride()),
        "default_contiguous": channels_last.is_contiguous(),
        "channels_last_contiguous": channels_last.is_contiguous(
            memory_format=torch.channels_last
        ),
        "same_values": torch.equal(image, channels_last),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
