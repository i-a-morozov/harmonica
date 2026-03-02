#!/usr/bin/env python3

"""script/orbit.py"""

from __future__ import annotations

from configuration import AppConfig, WidgetDefaults, load_app_config

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from harmonica.cs import factory
from harmonica.data import Data
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


def _compute_orbit(
    *,
    plane: str,
    length: int,
    offset: int,
    use_rise: bool,
    save: bool,
    use_median: bool,
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

    cs, bpm = _load_active_bpm(prefix, tango)

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after filtering.")

    if length <= 0 or length > LIMIT:
        raise RuntimeError(f"Invalid length={length}, expected 1..{LIMIT}.")
    if offset < 0:
        raise RuntimeError(f"Invalid offset={offset}, expected a non-negative value.")
    if length + offset > LIMIT:
        raise RuntimeError(f"Invalid length+offset={length + offset}, expected <= {LIMIT}.")

    position = np.asarray([cs.get(f"{prefix}:{name}:TIME") for name in bpm])
    pv_prefix = prefix if not data_prefix else data_prefix
    pv_list = [pv_make(name, plane, prefix=pv_prefix) for name in bpm]
    pv_rise = [*bpm.values()]

    shift = 0
    if use_rise:
        if min(pv_rise) < 0:
            raise RuntimeError("Rise values are expected to be non-negative.")
        shift = max(pv_rise)
        if length + offset + shift > LIMIT:
            raise RuntimeError(
                f"Invalid length+offset+shift={length + offset + shift}, expected <= {LIMIT}."
            )

    total = length + offset + shift
    win = Window(length, dtype=dtype, device=device)
    matrix = np.asarray([cs.get(pv) for pv in pv_list])
    raw = torch.tensor(matrix, dtype=dtype, device=device)
    raw = torch.stack([signal[:total] for signal in raw])
    if use_rise:
        raw = torch.stack([signal[offset + rise : offset + rise + length] for signal, rise in zip(raw, pv_rise)])
    else:
        raw = raw[:, offset : offset + length]
    tbt = Data.from_data(win, raw)

    orbit = tbt.median().flatten().cpu().numpy() if use_median else tbt.mean().flatten().cpu().numpy()
    axis = plane.upper()

    df = pd.DataFrame(
        {
            "BPM": [*bpm.keys()],
            "POSITION": position,
            "S": position,
            axis: orbit,
        }
    )

    output = np.array([position, orbit])
    saved_file = None
    if save:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        saved_file = Path(f"tbt_orbit_plane_{plane}_length_{length}_time_({timestamp}).npy")
        np.save(saved_file, output)

    return {
        "time": time_label,
        "size": len(bpm),
        "axis": axis,
        "dataframe": df,
        "output": output,
        "saved_file": saved_file,
        "plot": plot,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": pv_list,
    }


def main() -> None:
    st.set_page_config(page_title="Orbit", layout="wide")
    st.title("Orbit")

    try:
        app_config: AppConfig = load_app_config("orbit")
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

    with st.sidebar.form("orbit_form", enter_to_submit=False):
        st.caption("Planes")
        plane_col_x, plane_col_y, plane_col_i = st.columns(3)
        plane_x = plane_col_x.checkbox("X", value=bool(app_config.script.x.get("enabled", True)))
        plane_y = plane_col_y.checkbox("Y", value=bool(app_config.script.y.get("enabled", False)))
        i_cfg = app_config.script.values.get("i")
        i_default = bool(i_cfg.get("enabled", False)) if isinstance(i_cfg, dict) else False
        plane_i = plane_col_i.checkbox("I", value=bool(app_config.script.values.get("i_enabled", i_default)))
        length = ui.number_input("Length", min_value=1, max_value=LIMIT, value=256, step=1)
        offset = ui.number_input("Offset", min_value=0, max_value=LIMIT, value=0, step=1)
        use_rise = ui.checkbox("Use rise data", value=False)

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
            use_median = ui.checkbox("Use median (instead of mean)", value=False)

        plot = ui.checkbox("Show orbit plot", value=True)
        save = ui.checkbox("Save `.npy` output", value=False)
        data_prefix = ui.text_input("PV data prefix override", value="")
        verbose = ui.checkbox("Verbose output", value=False)

        run = st.form_submit_button("Run", type="primary")

    right_col, _ = st.columns([5, 1])
    with right_col:
        if not run:
            st.info("Set options on the left and click Run.")
            return

        planes = [plane for plane, enabled in (("x", plane_x), ("y", plane_y), ("i", plane_i)) if enabled]
        if not planes:
            st.error("Select at least one plane.")
            return

        patterns = _build_patterns(selected_names, regex_text)
        if selection_mode in {"skip", "only"} and not patterns:
            st.error("For skip/only mode, select BPM names and/or enter regex patterns.")
            return

        skip = patterns if selection_mode == "skip" else None
        only = patterns if selection_mode == "only" else None

        results: dict[str, dict[str, object]] = {}
        with st.spinner("Running orbit computation..."):
            try:
                for plane in planes:
                    results[plane] = _compute_orbit(
                        plane=plane,
                        length=int(length),
                        offset=int(offset),
                        use_rise=use_rise,
                        save=save,
                        use_median=use_median,
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
        c1, c2, c3 = st.columns(3)
        c1.metric("BPMs", value=int(first["size"]))
        c2.metric("Turns", value=int(length))
        c3.metric("Planes", value=", ".join(plane.upper() for plane in planes))

        for plane in planes:
            result = results[plane]
            if result["saved_file"] is not None:
                st.write(f"{plane.upper()} saved: `{result['saved_file']}`")

        if verbose:
            with st.expander("Verbose details"):
                for plane in planes:
                    result = results[plane]
                    st.write(f"Plane {plane.upper()} | Time: {result['time']}")
                    st.write("Monitor list:")
                    for name, rise in result["bpm"].items():
                        st.write(f"- {name}: {rise}")
                    st.write("PV list:")
                    for pv in result["pv_list"]:
                        st.write(f"- {pv}")

        if plot:
            st.subheader("Orbit")
            for plane in planes:
                result = results[plane]
                st.caption(f"Plane {plane.upper()}")
                orbit_plot = px.line(
                    result["dataframe"],
                    x="POSITION",
                    y=result["axis"],
                    hover_data=["S"],
                    title=f"{result['time']}: TbT (ORBIT)",
                    markers=True,
                )
                orbit_plot.update_layout(
                    xaxis=dict(
                        tickmode="array",
                        tickvals=result["dataframe"]["POSITION"],
                        ticktext=result["dataframe"]["BPM"],
                    )
                )
                st.plotly_chart(orbit_plot, use_container_width=True)
        else:
            st.subheader("Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["dataframe"], use_container_width=True)


if __name__ == "__main__":
    main()
