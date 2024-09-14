from .forecaster import Forecaster, LoadForecaster, CrowdForecaster
from .basic_forecaster import BasicForecaster
from .cot_forecaster import COT_Forecaster
from .advanced_forecaster import AdvancedForecaster
from .consistent_forecaster import ConsistentForecaster
from static_checks import NegChecker
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
        case "COT_Forecaster":
            return COT_Forecaster(**forecaster_config)
        case "ConsistentForecaster":
            return ConsistentForecaster(
                hypocrite=BasicForecaster(**forecaster_config),
                checks=[
                    NegChecker(),
                ],
                instantiation_kwargs={"model": forecaster_config["model"]},
                bq_func_kwargs={"model": forecaster_config["model"]},
            )
        case "RecursiveConsistentForecaster":
            return ConsistentForecaster.recursive(
                depth=4,
                hypocrite=BasicForecaster(**forecaster_config),
                checks=[
                    NegChecker(),
                    # ParaphraseChecker(),
                ],  # , ParaphraseChecker(), ButChecker(), CondChecker()
                instantiation_kwargs={"model": forecaster_config["model"]},
                bq_func_kwargs={"model": forecaster_config["model"]},
            )
        case "AdvancedForecaster":
            return AdvancedForecaster(**forecaster_config)
        case "LoadForecaster":
            return LoadForecaster(**forecaster_config)
        case "CrowdForecaster":
            return CrowdForecaster(**forecaster_config)
        case _:
            raise ValueError(f"Invalid forecaster class: {forecaster_class}")


def make_forecaster(
    forecaster_class: str | None,
    custom_path: str | None,
    forecaster_config: dict[str, Any] | None,
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
    else:
        assert (
            forecaster_class is not None
        ), "forecaster_class must be provided for predefined forecaster."
        match forecaster_class:
            case (
                "BasicForecaster"
                | "COT_Forecaster"
                | "ConsistentForecaster"
                | "RecursiveConsistentForecaster"
            ):
                assert (
                    "model" in forecaster_config
                ), "Model must be specified for forecaster class"
                print(f"Using model: {forecaster_config['model']}")
            case _:
                pass
        return make_predefined_forecaster(forecaster_class, forecaster_config)
