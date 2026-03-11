#!/usr/bin/env python3
"""
Convert a PyTorch checkpoint/model (.pt/.pth/.ckpt) into
Axiom's canonical JSON descriptor format:

    format = "axiom.pytorch.sequential.v1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


FORMAT_NAME = "axiom.pytorch.sequential.v1"


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _to_list_f32(tensor: Any) -> Any:
    return tensor.detach().cpu().to(dtype=torch.float32).tolist()


def _pair(value: Any) -> List[int]:
    if isinstance(value, int):
        return [int(value), int(value)]
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    _fail(f"Expected int or pair, got {value!r}")


def _layer_from_module(module: Any, strict: bool) -> Dict[str, Any] | None:
    if isinstance(module, torch.nn.Linear):
        layer: Dict[str, Any] = {
            "type": "Linear",
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "weight": _to_list_f32(module.weight.data),
        }
        layer["bias"] = _to_list_f32(module.bias.data) if module.bias is not None else None
        return layer

    if isinstance(module, torch.nn.Conv2d):
        layer = {
            "type": "Conv2d",
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
            "kernel_size": _pair(module.kernel_size),
            "stride": _pair(module.stride),
            "padding": _pair(module.padding),
            "dilation": _pair(module.dilation),
            "groups": int(module.groups),
            "weight": _to_list_f32(module.weight.data),
            "bias": _to_list_f32(module.bias.data) if module.bias is not None else None,
        }
        return layer

    if isinstance(module, torch.nn.BatchNorm2d):
        layer = {
            "type": "BatchNorm",
            "num_features": int(module.num_features),
            "eps": float(module.eps),
            "momentum": float(module.momentum if module.momentum is not None else 0.1),
            "affine": bool(module.affine),
            "track_running_stats": bool(module.track_running_stats),
            "running_mean": _to_list_f32(module.running_mean.data),
            "running_var": _to_list_f32(module.running_var.data),
        }
        if module.affine:
            layer["weight"] = _to_list_f32(module.weight.data)
            layer["bias"] = _to_list_f32(module.bias.data)
        else:
            layer["weight"] = None
            layer["bias"] = None
        return layer

    if isinstance(module, torch.nn.LayerNorm):
        normalized_shape = module.normalized_shape
        if isinstance(normalized_shape, int):
            normalized_shape = [int(normalized_shape)]
        else:
            normalized_shape = [int(x) for x in normalized_shape]
        layer = {
            "type": "LayerNorm",
            "normalized_shape": normalized_shape,
            "eps": float(module.eps),
            "elementwise_affine": bool(module.elementwise_affine),
            "weight": _to_list_f32(module.weight.data) if module.elementwise_affine else None,
            "bias": _to_list_f32(module.bias.data) if module.elementwise_affine else None,
        }
        return layer

    if isinstance(module, torch.nn.MaxPool2d):
        return {
            "type": "MaxPool2d",
            "kernel_size": _pair(module.kernel_size),
            "stride": _pair(module.stride if module.stride is not None else module.kernel_size),
            "padding": _pair(module.padding),
        }

    if isinstance(module, torch.nn.AvgPool2d):
        return {
            "type": "AvgPool2d",
            "kernel_size": _pair(module.kernel_size),
            "stride": _pair(module.stride if module.stride is not None else module.kernel_size),
            "padding": _pair(module.padding),
            "count_include_pad": bool(module.count_include_pad),
        }

    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        output_size = module.output_size
        if isinstance(output_size, int):
            output_size = [output_size, output_size]
        else:
            output_size = [int(output_size[0]), int(output_size[1])]
        return {
            "type": "AdaptiveAvgPool2d",
            "output_size": output_size,
        }

    if isinstance(module, torch.nn.Flatten):
        return {
            "type": "Flatten",
            "start_dim": int(module.start_dim),
            "end_dim": int(module.end_dim),
        }

    if isinstance(module, torch.nn.ReLU):
        return {"type": "ReLU"}
    if isinstance(module, torch.nn.Sigmoid):
        return {"type": "Sigmoid"}
    if isinstance(module, torch.nn.Tanh):
        return {"type": "Tanh"}
    if isinstance(module, torch.nn.Softmax):
        return {"type": "Softmax", "dim": int(module.dim if module.dim is not None else -1)}
    if isinstance(module, torch.nn.LeakyReLU):
        return {"type": "LeakyReLU", "negative_slope": float(module.negative_slope)}
    if isinstance(module, torch.nn.Identity):
        return {"type": "Identity"}

    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        return {"type": "Dropout"}

    if strict:
        _fail(f"Unsupported module type: {module.__class__.__name__}")
    return None


def _collect_layers(module: Any, strict: bool, out: List[Dict[str, Any]]) -> None:
    children = list(module.children())
    if children:
        for child in children:
            _collect_layers(child, strict, out)
        return
    layer = _layer_from_module(module, strict)
    if layer is not None:
        out.append(layer)


def _from_module(module: Any, strict: bool) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    _collect_layers(module, strict, layers)
    if not layers:
        _fail("No supported layers found in module")
    return {
        "format": FORMAT_NAME,
        "layers": layers,
    }


def _from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Fallback heuristic for bare state_dict: infer a linear stack from *.weight/*.bias pairs.
    linear_keys = sorted(
        k for k, v in state_dict.items()
        if k.endswith(".weight") and hasattr(v, "ndim") and int(v.ndim) == 2
    )
    if not linear_keys:
        _fail("Could not infer layers from state_dict (no 2D weight matrices found)")

    layers: List[Dict[str, Any]] = []
    for weight_key in linear_keys:
        prefix = weight_key[:-7]
        weight = state_dict[weight_key]
        out_features, in_features = [int(x) for x in weight.shape]
        bias_key = f"{prefix}.bias"
        layer = {
            "type": "Linear",
            "in_features": in_features,
            "out_features": out_features,
            "weight": _to_list_f32(weight),
            "bias": _to_list_f32(state_dict[bias_key]) if bias_key in state_dict else None,
            "source_key": prefix,
        }
        layers.append(layer)

    return {
        "format": FORMAT_NAME,
        "layers": layers,
        "warning": "Inferred from state_dict; non-linear ops cannot be reconstructed from weights alone.",
    }


def _load_checkpoint(path: Path, strict: bool) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, torch.nn.Module):
        return _from_module(obj, strict=strict)

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return _from_state_dict(obj["state_dict"])
        if all(isinstance(k, str) for k in obj.keys()):
            return _from_state_dict(obj)

    _fail(f"Unsupported checkpoint payload type: {type(obj)!r}")
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .pt/.pth/.ckpt file")
    parser.add_argument("--output", required=True, help="Output JSON descriptor path")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on unsupported module types (default: true)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        import torch  # pylint: disable=import-error
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(f"Failed to import torch: {exc}", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    try:
        descriptor = _load_checkpoint(input_path, strict=args.strict)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - surfaced in Julia wrapper/tests
        print(str(exc), file=sys.stderr)
        sys.exit(1)
