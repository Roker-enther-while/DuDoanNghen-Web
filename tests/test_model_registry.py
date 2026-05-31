import pytest

from src.training.registry import get_model_builder, list_models, validate_model_name


def test_registry_has_required_models():
    names = {item["name"] for item in list_models()}
    assert {
        "naive_last_value",
        "moving_average",
        "lstm",
        "gru",
        "tcn",
        "transformer",
        "tcn_lstm",
        "tcn_attention_bilstm",
    } <= names


def test_registry_bad_name_message():
    with pytest.raises(ValueError, match="Unknown model"):
        validate_model_name("bad_model")


def test_all_registered_models_are_implemented_builders():
    for item in list_models():
        assert item["implemented"] is True
        assert item["recommended_config"]
        assert callable(get_model_builder(item["name"]))
