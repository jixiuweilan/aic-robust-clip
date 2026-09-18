"""Opt-in execution settings; absent settings preserve legacy identities."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceConfig:
    cache_batch_size: int
    eval_batch_size: int
    head_batch_size: int
    num_workers: int = 0
    eval_num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False

    @classmethod
    def from_config(cls, config):
        value = config.get("performance", {})
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) - fields:
            raise ValueError("invalid performance configuration fields")
        micro = config.get("batch_size", 1)
        effective = config.get("effective_batch_size", 128)
        for name, number in (("batch_size", micro), ("effective_batch_size", effective)):
            if type(number) is not int or number <= 0:
                raise ValueError(f"{name} must be a positive integer")
        result = cls(**{**dict.fromkeys(("cache_batch_size", "eval_batch_size", "head_batch_size"), micro),
                       "eval_num_workers": value.get("num_workers", 0), **value})
        for name in ("cache_batch_size", "eval_batch_size", "head_batch_size", "prefetch_factor"):
            number = getattr(result, name)
            if type(number) is not int or number <= 0:
                raise ValueError(f"performance.{name} must be a positive integer")
        for name in ("num_workers", "eval_num_workers"):
            if type(getattr(result, name)) is not int or not 0 <= getattr(result, name) <= 16:
                raise ValueError(f"performance.{name} must be an integer between 0 and 16")
        if result.prefetch_factor > 4 or type(result.pin_memory) is not bool:
            raise ValueError("prefetch_factor must be at most 4 and pin_memory must be boolean")
        if effective % micro or effective % result.head_batch_size:
            raise ValueError("effective batch size must be divisible by training/head batch sizes")
        parameters = config.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be an object")
        accumulation = parameters.get("accumulation_steps", effective // micro)
        if config.get("execution_mode", "smoke") == "formal" and (
                type(accumulation) is not int or accumulation != effective // micro):
            raise ValueError("accumulation_steps conflicts with the declared effective batch size")
        if config.get("execution_mode", "smoke") == "smoke" and (
                micro != 1 or any(getattr(result, name) != 1 for name in (
                    "cache_batch_size", "eval_batch_size", "head_batch_size"))
                or result.num_workers or result.eval_num_workers or result.pin_memory):
            raise ValueError("smoke requires batch sizes 1, zero workers and no pinned prefetch")
        return result


def transfer_tensor(tensor, device, **kwargs):
    """Only pinned CUDA input transfers opt into asynchronous submission."""
    return tensor.to(device=device, non_blocking=str(device).startswith("cuda") and tensor.is_pinned(), **kwargs)
