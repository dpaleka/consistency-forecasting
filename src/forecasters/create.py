from .forecaster import Forecaster, LoadForecaster, CrowdForecaster
from .basic_forecaster import (
    BasicForecaster,
    BasicForecasterWithExamples,
    BasicForecasterTextBeforeParsing,
)
from .cot_forecaster import (
    CoT_Forecaster,
    CoT_ForecasterWithExamples,
    CoT_ForecasterTextBeforeParsing,
)
from .advanced_forecaster import AdvancedForecaster
from .consistent_forecaster import ConsistentForecaster
from .PromptedToCons_Forecaster import PromptedToCons_Forecaster
from static_checks import choose_checkers
import importlib
from pathlib import Path
from typing import Any


def make_custom_forecaster(
    custom_path: str, class_name: str | None, forecaster_config: dict[str, Any] | None
) -> Forecaster:
    if not custom_path:
        raise ValueError("custom_path must be provided for Custom forecaster.")

    custom_path = Path(custom_path)
    if not custom_path.exists():
        raise FileNotFoundError(f"Custom forecaster file not found: {custom_path}")

    spec = importlib.util.spec_from_file_location("custom_module", custom_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {custom_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find subclass of Forecaster
    custom_classes = [
        cls
        for cls in module.__dict__.values()
        if isinstance(cls, type)
        and issubclass(cls, Forecaster)
        and cls is not Forecaster
    ]

    print(f"Custom classes found in file: {custom_classes}")
    if not custom_classes:
        raise ValueError("No subclass of Forecaster found in the custom module.")
    if class_name:
        custom_classes = [
            cls for cls in custom_classes if cls.__qualname__ == class_name
        ]
    if len(custom_classes) != 1:
        raise ValueError("Expected exactly one Forecaster subclass.")

    return custom_classes[0](**forecaster_config)  # Instantiate the custom forecaster


def make_predefined_forecaster(
    forecaster_class: str, forecaster_config: dict[str, Any] | None
) -> Forecaster:
    match forecaster_class:
        case "BasicForecaster":
            return BasicForecaster(**forecaster_config)
        case "BasicForecasterWithExamples":
            return BasicForecasterWithExamples(**forecaster_config)
        case "BasicForecasterTextBeforeParsing":
            return BasicForecasterTextBeforeParsing(**forecaster_config)
        case "CoT_Forecaster":
            return CoT_Forecaster(**forecaster_config)
        case "CoT_ForecasterWithExamples":
            return CoT_ForecasterWithExamples(**forecaster_config)
        case "CoT_ForecasterTextBeforeParsing":
            return CoT_ForecasterTextBeforeParsing(**forecaster_config)
        case "PromptedToCons_Forecaster":
            return PromptedToCons_Forecaster(**forecaster_config)
        case "AdvancedForecaster":
            return AdvancedForecaster(**forecaster_config)
        case "LoadForecaster":
            return LoadForecaster(**forecaster_config)
        case "CrowdForecaster":
            return CrowdForecaster(**forecaster_config)
        case _:
            raise ValueError(f"Invalid forecaster class: {forecaster_class}")


def make_consistent_forecaster(
    forecaster_config: dict[str, Any] | None,
    checks: list[str],
    depth: int,
) -> ConsistentForecaster:
    checks = choose_checkers(checks).values()
    return ConsistentForecaster.recursive(
        depth=depth,
        hypocrite=make_predefined_forecaster("BasicForecaster", forecaster_config),
        checks=checks,
        instantiation_kwargs={"model": forecaster_config["model"]},
        bq_func_kwargs={"model": forecaster_config["model"]},
        **forecaster_config,
    )


def make_forecaster(
    forecaster_class: str | None,
    custom_path: str | None,
    forecaster_config: dict[str, Any] | None,
    checks: list[str] | None,
    depth: int | None,
) -> Forecaster:
    """Kwargs are already parsed before this"""
    if custom_path is not None:
        assert (
            forecaster_class is None
        ), "forecaster_class must be None for Custom forecaster."
        if "::" in custom_path:
            custom_path, class_name = custom_path.split("::")
        return make_custom_forecaster(
            custom_path, class_name, forecaster_config=forecaster_config
        )
    elif forecaster_class == "ConsistentForecaster":
        assert checks  # HACK sometimes checks gets converted to a tuple () so we don't check for None
        return make_consistent_forecaster(
            forecaster_config=forecaster_config,
            checks=checks,
            depth=depth,
        )
    else:
        assert (
            forecaster_class is not None
        ), "forecaster_class must be provided for predefined forecaster."

        if any(
            name_root in forecaster_class
            for name_root in [
                "BasicForecaster",
                "CoT_Forecaster",
                "ConsistentForecaster",
            ]
        ):
            assert (
                "model" in forecaster_config
            ), f"Model must be specified for forecaster class {forecaster_class}"
            print(f"Using model: {forecaster_config['model']}")
        return make_predefined_forecaster(forecaster_class, forecaster_config)
