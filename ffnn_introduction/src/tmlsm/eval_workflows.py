from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import json
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import klax

from . import models as tm
from . import workflows as wf


# -----------------------------
# Run abstraction
# -----------------------------
@dataclass(frozen=True)
class Run:
    tag: str
    meta: dict[str, Any]
    meta_path: Path
    model_path: Path
    model: Any

    @property
    def model_id(self) -> str:
        return str(self.meta.get("model_id", "")).upper().strip()

    @property
    def steps(self) -> int | None:
        v = self.meta.get("steps", None)
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    @property
    def arch_name(self) -> str | None:
        # common across your tasks
        return self.meta.get("arch_name") or self.meta.get("architecture", {}).get("name")

    @property
    def strategy(self) -> str | None:
        # task 3, task 5.x etc.
        return self.meta.get("strategy")

    @property
    def init_idx(self) -> int | None:
        v = self.meta.get("init_idx", None)
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    @property
    def base_tag(self) -> str:
        # strip trailing _initXX if present
        return re.sub(r"_init\d+$", "", self.tag)


# -----------------------------
# Helpers: architecture parsing
# -----------------------------
def _get_arch_from_meta(meta: dict[str, Any]) -> tuple[int, int]:
    """
    Extract (l, n). Supports multiple meta formats in your repo.
    """
    if "num_hidden_layers" in meta and "nodes_per_layer" in meta:
        return int(meta["num_hidden_layers"]), int(meta["nodes_per_layer"])

    arch = meta.get("architecture", {}) or {}
    if "l" in arch and "n" in arch:
        return int(arch["l"]), int(arch["n"])

    bench = meta.get("benchmark_architecture", {}) or {}
    if "l" in bench and "n" in bench:
        return int(bench["l"]), int(bench["n"])

    tag = str(meta.get("tag", ""))
    m = re.search(r"_l(\d+)_n(\d+)_", tag)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise KeyError(
        "Could not find architecture (expected num_hidden_layers/nodes_per_layer "
        "or architecture.{l,n} or benchmark_architecture.{l,n} or parseable _l*_n* in tag)."
    )


def _activation_from_meta(meta: dict[str, Any]) -> Callable:
    # In your metas you often store "softplus" as a string.
    act = meta.get("activation", None)
    if act is None:
        act = (meta.get("architecture", {}) or {}).get("activation", None)
    if act is None:
        act = (meta.get("benchmark_architecture", {}) or {}).get("activation", None)

    act_s = str(act).lower() if act is not None else "softplus"
    if act_s in ("softplus", "jax.nn.softplus"):
        return jax.nn.softplus
    if act_s in ("tanh", "jax.nn.tanh"):
        return jax.nn.tanh
    if act_s in ("relu", "jax.nn.relu"):
        return jax.nn.relu

    # fallback (keep predictable)
    return jax.nn.softplus


def _needs_finalise(meta: dict[str, Any]) -> bool:
    """
    Conservative policy:
      - all model_ids starting with 'W' (WITI, WICUB, WF, WF_AUG, ...)
      - any loss mentioning Sobolev
      - any model_type indicating sobolev-based training
    """
    mid = str(meta.get("model_id", "")).upper().strip()
    if mid.startswith("W"):
        return True

    loss = str(meta.get("loss", "")).lower()
    if "sobolev" in loss:
        return True

    mtype = str(meta.get("model_type", "")).upper().strip()
    if mtype in ("PANN",):
        return True

    return False


# -----------------------------
# Build like-model for deserialise
# -----------------------------
def _build_like_model(
    meta: dict[str, Any],
    *,
    like_key: jrandom.PRNGKey,
    dataset_1: dict | None,
    G_cub: jnp.ndarray | None,
) -> Any:
    mid = str(meta.get("model_id", "")).upper().strip()
    l, n = _get_arch_from_meta(meta)
    activation = _activation_from_meta(meta)

    # ---- MS / MSW (C -> P) ----
    if mid in ("MS", "MSW"):
        return tm.build(
            key=like_key,
            input_dim=6,
            output_dim=9,
            num_hidden_layers=l,
            nodes_per_layer=n,
            activations=activation,
            constrain_icnn_weights=False,
        )

    # ---- WITI (W(I) + P via invariants TI) ----
    if mid == "WITI":
        if dataset_1 is None:
            raise ValueError("Loading WITI requires dataset_1 (for G_ti).")
        G_ti = dataset_1["G_ti"]
        return tm.SobolevModel_WI_ti(
            G_ti=G_ti,
            key=like_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=False,
            is_ficnn=True,
        )

    # ---- WICUB (W(I_cubic) + P via cubic invariants) ----
    if mid == "WICUB":
        if G_cub is None:
            raise ValueError("Loading WICUB requires G_cub.")
        return tm.SobolevModel_WI_Cubic(
            G_cub=G_cub,
            key=like_key,
            input_dim=6,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=False,
            is_ficnn=True,
        )

    # ---- WF / WF_AUG (W(F) polyconvex ICNN) ----
    if mid in ("WF", "WF_AUG"):
        icnn = meta.get("icnn", {}) or {}
        is_icnn = bool(icnn.get("is_icnn", True))
        is_ficnn = bool(icnn.get("is_ficnn", False))
        return tm.SobolevModel_WF(
            key=like_key,
            input_dim=19,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=is_icnn,
            is_ficnn=is_ficnn,
        )

    raise ValueError(f"Unsupported model_id='{mid}'. Extend _build_like_model for this case.")


# -----------------------------
# Discovery + loading
# -----------------------------
def load_runs(
    artifacts_dir: str | Path,
    *,
    model_id: str | None = None,
    tag_contains: str | None = None,
    dataset_1: dict | None = None,
    dataset_3: dict | None = None,
    G_cub: jnp.ndarray | None = None,
    like_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
    strict: bool = False,
) -> list[Run]:
    """
    Self-contained loader:
      - discovers runs via *.json files
      - resolves model_path
      - builds like-model from meta
      - finalises sobolev/klax models (like-model + loaded model)
      - returns list[Run]
    """
    artifacts_dir = Path(artifacts_dir)

    meta_paths = sorted(artifacts_dir.glob("*.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No *.json meta files found in: {artifacts_dir}")

    runs: list[Run] = []

    for mp in meta_paths:
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as e:
            if strict:
                raise
            print(f"[eval_workflows] SKIP meta parse: {mp.name}: {e}")
            continue

        tag = meta.get("tag", mp.stem)

        # filters
        if model_id is not None:
            if str(meta.get("model_id", "")).upper().strip() != str(model_id).upper().strip():
                continue
        if tag_contains is not None and tag_contains not in str(tag):
            continue

        # resolve model path
        model_path = meta.get("saved_model_path", None)
        if model_path:
            model_path = Path(model_path)
        else:
            model_path = artifacts_dir / f"{tag}.eqx"

        if not model_path.exists():
            msg = f"Missing model file for tag={tag}: {model_path}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[eval_workflows] SKIP: {msg}")
            continue

        # build like-model
        try:
            like_model = _build_like_model(
                meta,
                like_key=like_key,
                dataset_1=dataset_1,
                G_cub=G_cub,
            )

            # Important: finalise like structure for sobolev/klax models
            if _needs_finalise(meta):
                like_model = klax.finalize(like_model)

            model = eqx.tree_deserialise_leaves(str(model_path), like=like_model)

            # Safety net: finalise loaded model as well
            if _needs_finalise(meta):
                model = klax.finalize(model)

        except Exception as e:
            if strict:
                raise
            print(f"[eval_workflows] SKIP load tag={tag}: {e}")
            continue

        runs.append(Run(tag=str(tag), meta=meta, meta_path=mp, model_path=model_path, model=model))

    return runs


# -----------------------------
# Grouping utilities
# -----------------------------
def group_runs(
    runs: Iterable[Run],
    *,
    by: str | Callable[[Run], str] = "base_tag",
) -> dict[str, list[Run]]:
    """
    Groups runs into a dict. Common grouping keys:
      - by="base_tag" (averages over initXX)
      - by="steps"
      - by="arch_name"
      - by=callable
    """
    if isinstance(by, str):
        if by == "base_tag":
            key_fn = lambda r: r.base_tag
        elif by == "steps":
            key_fn = lambda r: str(r.steps)
        elif by == "arch_name":
            key_fn = lambda r: str(r.arch_name)
        elif by == "model_id":
            key_fn = lambda r: r.model_id
        else:
            raise ValueError(f"Unknown group key '{by}'.")
    else:
        key_fn = by

    out: dict[str, list[Run]] = {}
    for r in runs:
        k = key_fn(r)
        out.setdefault(k, []).append(r)

    # stable ordering inside groups
    for k in out:
        out[k] = sorted(out[k], key=lambda rr: (rr.steps or -1, rr.init_idx or 0, rr.tag))

    return out


# -----------------------------
# Test data helpers (delegate to workflows)
# -----------------------------
def get_test_sets(run: Run, *, dataset_1: dict | None = None, G_cub: jnp.ndarray | None = None) -> dict[str, Any]:
    """
    Delegates to wf.get_test_data_for_run to keep consistency with your pipeline.
    """
    return wf.get_test_data_for_run(run.meta_path, dataset_1=dataset_1, G_cub=G_cub)


def get_test_ms(run: Run, *, dataset_1: dict, which: str = "biax") -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    MS/MSW: returns (X, Y) arrays.
      - which in {"biax","mixed","full"}
    """
    ts = get_test_sets(run, dataset_1=dataset_1)
    if which == "biax":
        return ts["biax_test"]
    if which == "mixed":
        return ts["mixed_test"]
    if which == "full":
        Xb, Yb = ts["biax_test"]
        Xm, Ym = ts["mixed_test"]
        return jnp.concatenate([Xb, Xm], axis=0), jnp.concatenate([Yb, Ym], axis=0)
    raise ValueError("which must be one of {'biax','mixed','full'}")


def get_test_witi(run: Run, *, dataset_1: dict, which: str = "mixed") -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """
    WITI: returns ((F, I), (W, P)).
      - which in {"biax","mixed","full"}
    """
    ts = get_test_sets(run, dataset_1=dataset_1)
    if which == "biax":
        return ts["biax_test"]
    if which == "mixed":
        return ts["mixed_test"]
    if which == "full":
        (Fb, Ib), (Wb, Pb) = ts["biax_test"]
        (Fm, Im), (Wm, Pm) = ts["mixed_test"]
        F = jnp.concatenate([Fb, Fm], axis=0)
        I = jnp.concatenate([Ib, Im], axis=0)
        W = jnp.concatenate([Wb, Wm], axis=0)
        P = jnp.concatenate([Pb, Pm], axis=0)
        return (F, I), (W, P)
    raise ValueError("which must be one of {'biax','mixed','full'}")


# -----------------------------
# Prediction helpers
# -----------------------------
def predict_ms_stress(model: Any, X: jnp.ndarray) -> jnp.ndarray:
    """
    MS/MSW: model(X) -> (N,9) -> reshape to (N,3,3)
    """
    Y_pred = jax.vmap(model)(X)  # (N,9)
    return Y_pred.reshape(Y_pred.shape[0], 3, 3)


def predict_witi_energy_stress(model: Any, F: jnp.ndarray, I: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    WITI/WICUB/WF: model(input) -> (W, P) (per-sample); vmap -> (N,), (N,3,3)
    """
    W_pred, P_pred = jax.vmap(model)((F, I))  # for WITI/WICUB signatures
    return jnp.squeeze(W_pred), P_pred


def predict(run: Run, *inputs) -> Any:
    """
    Dispatch prediction based on run.model_id.
    For customizable notebook usage; you can also call predict_ms_stress / predict_witi_energy_stress directly.
    """
    mid = run.model_id
    if mid in ("MS", "MSW"):
        (X,) = inputs
        return predict_ms_stress(run.model, X)

    if mid == "WITI":
        F, I = inputs
        return predict_witi_energy_stress(run.model, F, I)

    # Extend as needed for WICUB, WF, etc. (they may have different input signatures)
    raise ValueError(f"No predict() implementation for model_id={mid}. Use a dedicated helper or extend predict().")
