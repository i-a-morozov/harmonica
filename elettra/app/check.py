#!/usr/bin/env python3

"""script/check.py"""

from __future__ import annotations

from configuration import AppConfig, WidgetDefaults, load_app_config

import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from harmonica.cs import factory
from harmonica.data import Data
from harmonica.decomposition import Decomposition
from harmonica.filter import Filter
from harmonica.util import LIMIT, bpm_select, pv_make
from harmonica.window import Window


def _as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _parse_regex_list(text: str) -> list[str]:
    parts = [item.strip() for item in re.split(r"[\s,]+", text.strip()) if item.strip()]
    return parts


def _build_patterns(selected_names: list[str], regex_text: str) -> list[str]:
    patterns: list[str] = []
    patterns.extend(f"^{re.escape(name)}$" for name in selected_names)
    patterns.extend(_parse_regex_list(regex_text))
    return patterns



def _load_active_bpm(prefix: str, tango: bool) -> tuple[object, dict[str, int]]:
    cs = factory(target=("tango" if tango else "epics"))
    monitor_count = int(cs.get(f"{prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{prefix}:MONITOR:LIST")[:monitor_count]
    monitor_names = [_as_str(name) for name in monitor_names]
    monitor_flag = np.asarray([cs.get(f"{prefix}:{name}:FLAG") for name in monitor_names])
    monitor_rise = np.asarray([cs.get(f"{prefix}:{name}:RISE") for name in monitor_names])
    bpm = {name: int(rise) for name, flag, rise in zip(monitor_names, monitor_flag, monitor_rise) if int(flag) == 1}
    return cs, bpm


def _compute_check(
    *,
    plane: str,
    sample_length: int,
    load_length: int,
    offset: int,
    use_rise: bool,
    transform: str,
    filter_type: str,
    rank: int,
    svd_type: str,
    buffer: int,
    count: int,
    window_order: float,
    factor: float,
    load_phase: bool,
    plot: bool,
    prefix: str,
    data_prefix: str,
    tango: bool,
    device: str,
    dtype_name: str,
    verbose: bool,
    skip: list[str] | None,
    only: list[str] | None,
) -> dict[str, object]:
    time_label = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if sample_length <= 0:
        raise RuntimeError("Sample length must be positive.")
    if load_length <= 0 or load_length > LIMIT:
        raise RuntimeError(f"Invalid load={load_length}, expected 1..{LIMIT}.")
    if sample_length > load_length:
        raise RuntimeError(f"Sample length {sample_length} should be <= load length {load_length}.")
    if offset < 0:
        raise RuntimeError(f"Invalid offset={offset}, expected a non-negative value.")
    if load_length + offset > LIMIT:
        raise RuntimeError(f"Invalid load+offset={load_length + offset}, expected <= {LIMIT}.")
    if window_order < 0.0:
        raise RuntimeError("Window order must be greater than or equal to zero.")
    if factor <= 0.0:
        raise RuntimeError("Threshold factor must be positive.")

    cs, bpm = _load_active_bpm(prefix, tango)

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after filtering.")

    plane_up = plane.upper()
    phase_model = torch.tensor([cs.get(f"{prefix}:{name}:MODEL:F{plane_up}") for name in bpm], dtype=dtype, device=device)
    q = float(cs.get(f"{prefix}:FREQUENCY:VALUE:{plane_up}"))
    q_model = float(cs.get(f"{prefix}:FREQUENCY:MODEL:{plane_up}"))

    pv_list: list[str] = []
    if load_phase:
        phase = torch.tensor([cs.get(f"{prefix}:{name}:PHASE:VALUE:{plane_up}") for name in bpm], dtype=dtype, device=device)
    else:
        pv_prefix = prefix if not data_prefix else data_prefix
        pv_list = [pv_make(name, plane, prefix=pv_prefix) for name in bpm]
        pv_rise = [*bpm.values()]

        rise_shift = 0
        if use_rise:
            if min(pv_rise) < 0:
                raise RuntimeError("Rise values are expected to be non-negative.")
            rise_shift = max(pv_rise)
            if load_length + offset + rise_shift > LIMIT:
                raise RuntimeError(
                    f"Invalid load+offset+rise={load_length + offset + rise_shift}, expected <= {LIMIT}."
                )

        total = load_length + offset + rise_shift
        win = Window(load_length, "cosine_window", window_order, dtype=dtype, device=device)
        matrix = np.asarray([cs.get(pv) for pv in pv_list])
        raw = torch.tensor(matrix, dtype=dtype, device=device)
        raw = torch.stack([signal[:total] for signal in raw])
        if use_rise:
            raw = torch.stack(
                [signal[offset + rise : offset + rise + load_length] for signal, rise in zip(raw, pv_rise)]
            )
        else:
            raw = raw[:, offset : offset + load_length]
        tbt = Data.from_data(win, raw)

        if transform == "mean":
            tbt.window_remove_mean()
        elif transform == "median":
            tbt.work.sub_(tbt.median())
        elif transform == "normalize":
            tbt.normalize()

        if filter_type == "svd":
            flt = Filter(tbt)
            flt.filter_svd(rank=rank)
        elif filter_type == "hankel":
            flt = Filter(tbt)
            flt.filter_svd(rank=rank)
            flt.filter_hankel(rank=rank, random=(svd_type == "randomized"), buffer=buffer, count=count)

        dec = Decomposition(tbt)
        phase, _, _ = dec.harmonic_phase(q, length=sample_length, order=window_order, factor=factor)

    check, table = Decomposition.phase_check(q, q_model, phase, phase_model, factor=factor)

    mark = [-1 if key not in check else check[key][0] / 2 - 1 for key in range(len(bpm))]
    advance_df = pd.concat(
        [
            pd.DataFrame({"CASE": case, "BPM_INDEX": range(len(bpm)), "ADVANCE": data, "BPM": [*bpm.keys()]})
            for case, data in zip(
                ["PHASE", "MODEL", "CHECK", "FLAG"],
                [table["phase"].cpu().numpy(), table["model"].cpu().numpy(), table["check"].cpu().numpy(), mark],
            )
        ],
        ignore_index=True,
    )

    flagged_rows = []
    for marked in sorted(check):
        index, value = check[marked]
        if index != 0:
            flagged_rows.append(
                {
                    "BPM_INDEX": int(marked),
                    "BPM": list(bpm.keys())[marked],
                    "SHIFT": int(index),
                    "PHASE": float(value),
                }
            )
    flagged_df = pd.DataFrame(flagged_rows)

    return {
        "time": time_label,
        "size": len(bpm),
        "flagged_count": len(flagged_rows),
        "advance_df": advance_df,
        "flagged_df": flagged_df,
        "plot": plot,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": pv_list,
    }


def main() -> None:
    st.set_page_config(page_title="Check", layout="wide")
    st.title("Check")

    try:
        app_config: AppConfig = load_app_config("check")
    except Exception as exception:
        st.error(str(exception))
        return

    prefix_name = app_config.global_.prefix
    tango = app_config.global_.control_system == "tango"
    device_name = app_config.global_.device
    dtype_name = app_config.global_.dtype

    ui = WidgetDefaults(app_config.script)

    st.sidebar.header("Controls")
    with st.sidebar.expander("Configuration", expanded=False):
        st.caption(f"Prefix: `{prefix_name}`")
        st.caption(f"Control system: `{app_config.global_.control_system}`")
        st.caption(f"Device: `{device_name}`")
        st.caption(f"Dtype: `{dtype_name}`")

    bpm_names: list[str] = []
    bpm_error: str | None = None
    try:
        _, active_bpm = _load_active_bpm(prefix_name, tango)
        bpm_names = sorted(active_bpm.keys())
    except Exception as exception:  # pragma: no cover - depends on external CS
        bpm_error = str(exception)

    if bpm_error:
        st.sidebar.warning(f"BPM list unavailable: {bpm_error}")
    else:
        st.sidebar.caption(f"Detected active BPMs: {len(bpm_names)}")

    with st.sidebar.form("check_form", enter_to_submit=False):
        st.caption("Planes")
        plane_col_x, plane_col_y = st.columns(2)
        plane_x = plane_col_x.checkbox("X", value=bool(app_config.script.x.get("enabled", True)))
        plane_y = plane_col_y.checkbox("Y", value=bool(app_config.script.y.get("enabled", False)))

        with st.expander("Selection", expanded=False):
            selection_mode = ui.radio("BPM filter mode", options=["all", "skip", "only"], index=0, horizontal=True)
            selected_names = st.multiselect(
                "BPM names (checked list)",
                options=bpm_names,
                disabled=not bpm_names,
            )
            regex_text = st.text_area(
                "Regex patterns (space/comma separated)",
                value="",
                height=68,
                help="You can combine checked BPM names and regex patterns.",
            )

        with st.expander("Filtering", expanded=False):
            transform = ui.radio(
                "Transform",
                options=["none", "mean", "median", "normalize"],
                index=0,
                horizontal=True,
            )
            filter_type = ui.selectbox("Filter", options=["none", "svd", "hankel"], index=0)
            rank = ui.number_input("Filter rank", min_value=1, value=8, step=1)
            svd_type = ui.selectbox("Hankel SVD type", options=["randomized", "full"], index=0)
            buffer = ui.number_input("Hankel randomized buffer", min_value=1, value=16, step=1)
            count = ui.number_input("Hankel randomized iterations", min_value=1, value=16, step=1)

        with st.expander("Plane X options", expanded=False):
            sample_length_x = ui.number_input("Sample length (X)", min_value=1, max_value=LIMIT, value=256, step=1)
            load_length_x = ui.number_input("Load length (X)", min_value=1, max_value=LIMIT, value=512, step=1)
            offset_x = ui.number_input("Offset (X)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_x = ui.checkbox("Use rise data (X)", value=False)
            window_x = ui.number_input("Window order (X)", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            factor_x = ui.number_input("Threshold factor (X)", min_value=0.000001, value=5.0, step=0.1, format="%0.4f")
            load_phase_x = ui.checkbox("Load phase from PV (X)", value=False)

        with st.expander("Plane Y options", expanded=False):
            sample_length_y = ui.number_input("Sample length (Y)", min_value=1, max_value=LIMIT, value=256, step=1)
            load_length_y = ui.number_input("Load length (Y)", min_value=1, max_value=LIMIT, value=512, step=1)
            offset_y = ui.number_input("Offset (Y)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_y = ui.checkbox("Use rise data (Y)", value=False)
            window_y = ui.number_input("Window order (Y)", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            factor_y = ui.number_input("Threshold factor (Y)", min_value=0.000001, value=5.0, step=0.1, format="%0.4f")
            load_phase_y = ui.checkbox("Load phase from PV (Y)", value=False)

        plot = ui.checkbox("Show check plot", value=True)
        data_prefix = ui.text_input("PV data prefix override", value="")
        verbose = ui.checkbox("Verbose output", value=False)

        run = st.form_submit_button("Run", type="primary")

    right_col, _ = st.columns([5, 1])
    with right_col:
        if not run:
            st.info("Set options on the left and click Run.")
            return

        planes = [plane for plane, enabled in (("x", plane_x), ("y", plane_y)) if enabled]
        if not planes:
            st.error("Select at least one plane.")
            return

        patterns = _build_patterns(selected_names, regex_text)
        if selection_mode in {"skip", "only"} and not patterns:
            st.error("For skip/only mode, select BPM names and/or enter regex patterns.")
            return

        skip = patterns if selection_mode == "skip" else None
        only = patterns if selection_mode == "only" else None

        plane_options: dict[str, dict[str, object]] = {
            "x": {
                "sample_length": int(sample_length_x),
                "load_length": int(load_length_x),
                "offset": int(offset_x),
                "use_rise": rise_x,
                "window_order": float(window_x),
                "factor": float(factor_x),
                "load_phase": load_phase_x,
            },
            "y": {
                "sample_length": int(sample_length_y),
                "load_length": int(load_length_y),
                "offset": int(offset_y),
                "use_rise": rise_y,
                "window_order": float(window_y),
                "factor": float(factor_y),
                "load_phase": load_phase_y,
            },
        }

        results: dict[str, dict[str, object]] = {}
        with st.spinner("Running phase synchronization check..."):
            try:
                for plane in planes:
                    option = plane_options[plane]
                    results[plane] = _compute_check(
                        plane=plane,
                        sample_length=int(option["sample_length"]),
                        load_length=int(option["load_length"]),
                        offset=int(option["offset"]),
                        use_rise=bool(option["use_rise"]),
                        transform=transform,
                        filter_type=filter_type,
                        rank=int(rank),
                        svd_type=svd_type,
                        buffer=int(buffer),
                        count=int(count),
                        window_order=float(option["window_order"]),
                        factor=float(option["factor"]),
                        load_phase=bool(option["load_phase"]),
                        plot=plot,
                        prefix=prefix_name,
                        data_prefix=data_prefix,
                        tango=tango,
                        device=device_name,
                        dtype_name=dtype_name,
                        verbose=verbose,
                        skip=skip,
                        only=only,
                    )
            except Exception as exception:  # pragma: no cover - depends on external CS
                st.error(str(exception))
                return

        first = results[planes[0]]
        st.success("Computation finished.")
        c1, c2 = st.columns(2)
        c1.metric("BPMs", value=int(first["size"]))
        c2.metric("Planes", value=", ".join(plane.upper() for plane in planes))

        st.subheader("Flagged BPMs")
        for plane in planes:
            result = results[plane]
            st.caption(f"Plane {plane.upper()} | flagged={result['flagged_count']}")
            if result["flagged_df"].empty:
                st.write("No flagged BPMs.")
            else:
                st.dataframe(result["flagged_df"], use_container_width=True)

        if verbose:
            with st.expander("Verbose details"):
                for plane in planes:
                    result = results[plane]
                    st.write(f"Plane {plane.upper()} | Time: {result['time']}")
                    st.write("Monitor list:")
                    for name, rise in result["bpm"].items():
                        st.write(f"- {name}: {rise}")
                    if result["pv_list"]:
                        st.write("PV list:")
                        for pv in result["pv_list"]:
                            st.write(f"- {pv}")

        if plot:
            st.subheader("Advance")
            for plane in planes:
                result = results[plane]
                st.caption(f"Plane {plane.upper()}")
                figure = px.scatter(
                    result["advance_df"],
                    x="BPM_INDEX",
                    y="ADVANCE",
                    color="CASE",
                    title=f"{result['time']}: ADVANCE ({plane.upper()})",
                    opacity=0.75,
                    color_discrete_sequence=["red", "green", "blue", "black"],
                )
                bpm_keys = [*result["bpm"].keys()]
                figure.update_layout(
                    xaxis=dict(tickmode="array", tickvals=list(range(len(bpm_keys))), ticktext=bpm_keys)
                )
                figure.update_traces(marker={"size": 10})
                st.plotly_chart(figure, use_container_width=True)
        else:
            st.subheader("Advance Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["advance_df"], use_container_width=True)


if __name__ == "__main__":
    main()
