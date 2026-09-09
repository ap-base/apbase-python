from __future__ import annotations

import os

import pytest

from apbase.config import config, resolve_n_threads
from apbase.variogram import Variogram

ENV_VAR = "APBASE_N_THREADS"


def _reset() -> str | None:
    config._values.clear()
    return os.environ.pop(ENV_VAR, None)


def _restore(previous_env: str | None) -> None:
    config._values.clear()
    if previous_env is not None:
        os.environ[ENV_VAR] = previous_env
    else:
        os.environ.pop(ENV_VAR, None)


def test_config_setitem_getitem_roundtrip() -> None:
    previous = _reset()
    try:
        config["n_threads"] = 4
        assert config["n_threads"] == 4
        assert "n_threads" in config
    finally:
        _restore(previous)


def test_config_rejects_unknown_key() -> None:
    previous = _reset()
    try:
        try:
            config["not_a_real_key"] = 1
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for unknown config key")
    finally:
        _restore(previous)


def test_config_rejects_non_positive_value() -> None:
    previous = _reset()
    try:
        try:
            config["n_threads"] = 0
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for non-positive n_threads")
    finally:
        _restore(previous)


def test_resolve_n_threads_config_wins_over_env_var() -> None:
    previous = _reset()
    try:
        os.environ[ENV_VAR] = "8"
        config["n_threads"] = 4
        assert resolve_n_threads() == 4
    finally:
        _restore(previous)


def test_resolve_n_threads_env_var_wins_over_default() -> None:
    previous = _reset()
    try:
        os.environ[ENV_VAR] = "8"
        assert resolve_n_threads() == 8
    finally:
        _restore(previous)


def test_resolve_n_threads_falls_back_to_one() -> None:
    previous = _reset()
    try:
        assert resolve_n_threads() == 1
    finally:
        _restore(previous)


def test_variogram_picks_up_config_default() -> None:
    previous = _reset()
    try:
        config["n_threads"] = 3
        assert Variogram().n_threads == 3
    finally:
        _restore(previous)


def test_config_getitem_error_lists_valid_keys() -> None:
    previous = _reset()
    try:
        with pytest.raises(KeyError, match="n_threads"):
            config["n_threads"]
    finally:
        _restore(previous)


def test_resolve_n_threads_warns_and_falls_back_on_malformed_env_var() -> None:
    previous = _reset()
    try:
        os.environ[ENV_VAR] = "not-an-int"
        with pytest.warns(UserWarning, match=ENV_VAR):
            assert resolve_n_threads() == 1
    finally:
        _restore(previous)


if __name__ == "__main__":
    test_config_setitem_getitem_roundtrip()
    test_config_rejects_unknown_key()
    test_config_rejects_non_positive_value()
    test_resolve_n_threads_config_wins_over_env_var()
    test_resolve_n_threads_env_var_wins_over_default()
    test_resolve_n_threads_falls_back_to_one()
    test_variogram_picks_up_config_default()
    test_config_getitem_error_lists_valid_keys()
    test_resolve_n_threads_warns_and_falls_back_on_malformed_env_var()
