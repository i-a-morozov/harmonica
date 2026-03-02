#!/usr/bin/env python3

"""script/spectrum_all.py"""

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
from harmonica.filter import Filter
from harmonica.frequency import Frequency
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



def _load_joined_bpm(prefix: str, tango: bool) -> tuple[object, dict[str, int]]:
    cs = factory(target=("tango" if tango else "epics"))
    monitor_count = int(cs.get(f"{prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{prefix}:MONITOR:LIST")[:monitor_count]
    monitor_names = [_as_str(name) for name in monitor_names]
    monitor_flag = np.asarray([cs.get(f"{prefix}:{name}:FLAG") for name in monitor_names])
    monitor_join = np.asarray([cs.get(f"{prefix}:{name}:JOIN") for name in monitor_names])
    monitor_rise = np.asarray([cs.get(f"{prefix}:{name}:RISE") for name in monitor_names])
    bpm = {
        name: int(rise)
        for name, flag, rise, join in zip(monitor_names, monitor_flag, monitor_rise, monitor_join)
        if int(flag) == 1 and int(join) == 1
    }
    return cs, bpm


def _compute_joined_spectrum(
    *,
    plane: str,
    length: int,
    offset: int,
    use_rise: bool,
    transform: str,
    filter_type: str,
    rank: int,
    svd_type: str,
    buffer: int,
    count: int,
    window_order: float,
    f_min: float | None,
    f_max: float | None,
    beta_min: float,
    beta_max: float,
    use_nufft: bool,
    time_mode: str,
    use_log: bool,
    prefix: str,
    data_prefix: str,
    tango: bool,
    device: str,
    dtype_name: str,
    verbose: bool,
    circumference: float,
    skip: list[str] | None,
    only: list[str] | None,
) -> dict[str, object]:
    time_label = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    if circumference <= 0.0:
        raise RuntimeError("Circumference must be positive.")

    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if (f_min is None) ^ (f_max is None):
        raise RuntimeError("f_min and f_max must be provided together.")
    if f_min is not None and f_min < 0.0:
        raise RuntimeError(f"Invalid f_min={f_min}, expected a non-negative value.")
    if f_max is not None and f_max < f_min:
        raise RuntimeError(f"Invalid f_max={f_max}, expected >= f_min={f_min}.")
    f_range = (None, None) if f_min is None else (f_min, f_max)

    if beta_min < 0.0:
        raise RuntimeError(f"Invalid beta_min={beta_min}, expected a non-negative value.")
    if beta_max < 0.0:
        raise RuntimeError(f"Invalid beta_max={beta_max}, expected a non-negative value.")
    if beta_min > beta_max:
        raise RuntimeError(f"Invalid beta range: beta_min={beta_min} > beta_max={beta_max}.")

    cs, bpm = _load_joined_bpm(prefix, tango)

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after skip/only filtering.")

    beta = {name: float(cs.get(f"{prefix}:{name}:MODEL:B{plane.upper()}")) for name in bpm}
    for name in list(bpm):
        if not (beta_min <= beta[name] <= beta_max):
            bpm.pop(name)

    if not bpm:
        raise RuntimeError("BPM list is empty after beta threshold filtering.")

    if length <= 0 or length > LIMIT:
        raise RuntimeError(f"Invalid length={length}, expected 1..{LIMIT}.")
    if offset < 0:
        raise RuntimeError(f"Invalid offset={offset}, expected a non-negative value.")
    if length + offset > LIMIT:
        raise RuntimeError(f"Invalid length+offset={length + offset}, expected <= {LIMIT}.")
    if window_order < 0.0:
        raise RuntimeError("Window order must be greater than or equal to zero.")

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

    position = None
    if use_nufft:
        if time_mode == "position":
            position = np.asarray([cs.get(f"{prefix}:{name}:TIME") for name in bpm]) / circumference
        elif time_mode == "phase":
            total = float(cs.get(f"{prefix}:TAIL:MODEL:F{plane.upper()}"))
            if total == 0.0:
                raise RuntimeError(f"{prefix}:TAIL:MODEL:F{plane.upper()} is zero, cannot use phase time mode.")
            position = np.asarray([cs.get(f"{prefix}:{name}:MODEL:F{plane.upper()}") for name in bpm]) / total
        else:
            raise RuntimeError(f"Invalid time mode: {time_mode}.")

    total = length + offset + shift
    win = Window(length, "cosine_window", window_order, dtype=dtype, device=device)
    matrix = np.asarray([cs.get(pv) for pv in pv_list])
    raw = torch.tensor(matrix, dtype=dtype, device=device)
    raw = torch.stack([signal[:total] for signal in raw])
    if use_rise:
        raw = torch.stack([signal[offset + rise : offset + rise + length] for signal, rise in zip(raw, pv_rise)])
    else:
        raw = raw[:, offset : offset + length]
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

    freq = Frequency(tbt)
    grid, output = freq.compute_joined_spectrum(
        length=length,
        f_range=f_range,
        name="cosine_window",
        order=window_order,
        normalize=True,
        position=position,
        log=use_log,
    )

    grid_numpy = grid.cpu().numpy()
    output_numpy = output.cpu().numpy()
    axis = f"DTFT({plane.upper()})"
    dataframe = pd.DataFrame({"FREQUENCY": grid_numpy, axis: output_numpy})

    return {
        "time": time_label,
        "axis": axis,
        "size": len(bpm),
        "grid": grid_numpy,
        "output": output_numpy,
        "dataframe": dataframe,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": pv_list,
        "length": length,
        "f_min": f_min,
        "f_max": f_max,
    }


def main() -> None:
    st.set_page_config(page_title="Spectrum (combined)", layout="wide")
    st.title("Spectrum (combined)")

    try:
        app_config: AppConfig = load_app_config("spectrum_all")
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
        _, available_bpm = _load_joined_bpm(prefix_name, tango)
        bpm_names = sorted(available_bpm.keys())
    except Exception as exception:  # pragma: no cover - depends on external CS
        bpm_error = str(exception)

    if bpm_error:
        st.sidebar.warning(f"BPM list unavailable: {bpm_error}")
    else:
        st.sidebar.caption(f"Detected joined BPMs: {len(bpm_names)}")

    with st.sidebar.form("spectrum_all_form", enter_to_submit=False):
        st.caption("Planes")
        plane_col_x, plane_col_y = st.columns(2)
        plane_x = plane_col_x.checkbox("X", value=bool(app_config.script.x.get("enabled", True)))
        plane_y = plane_col_y.checkbox("Y", value=bool(app_config.script.y.get("enabled", False)))

        circumference = ui.number_input("Circumference", min_value=0.000001, value=259.2, step=0.1, format="%0.6f")

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
            length_x = ui.number_input("Length (X)", min_value=1, max_value=LIMIT, value=128, step=1)
            offset_x = ui.number_input("Offset (X)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_x = ui.checkbox("Use rise data (X)", value=False)
            window_x = ui.number_input("Window order (X)", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            use_range_x = ui.checkbox("Use custom frequency range (X)", value=False)
            f_min_x = ui.number_input("f_min (X)", min_value=0.0, value=0.0, step=0.0001, format="%0.6f")
            f_max_x = ui.number_input("f_max (X)", min_value=0.0, value=0.5, step=0.0001, format="%0.6f")
            beta_min_x = ui.number_input("beta_min (X)", min_value=0.0, value=0.0, step=0.1, format="%0.6f")
            beta_max_x = ui.number_input("beta_max (X)", min_value=0.0, value=1000.0, step=0.1, format="%0.6f")
            nufft_x = ui.checkbox("Use NUFFT (X)", value=False)
            time_x = ui.selectbox("NUFFT time mode (X)", options=["phase", "position"], index=0)
            log_x = ui.checkbox("Apply log10 (X)", value=False)

        with st.expander("Plane Y options", expanded=False):
            length_y = ui.number_input("Length (Y)", min_value=1, max_value=LIMIT, value=128, step=1)
            offset_y = ui.number_input("Offset (Y)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_y = ui.checkbox("Use rise data (Y)", value=False)
            window_y = ui.number_input("Window order (Y)", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            use_range_y = ui.checkbox("Use custom frequency range (Y)", value=False)
            f_min_y = ui.number_input("f_min (Y)", min_value=0.0, value=0.0, step=0.0001, format="%0.6f")
            f_max_y = ui.number_input("f_max (Y)", min_value=0.0, value=0.5, step=0.0001, format="%0.6f")
            beta_min_y = ui.number_input("beta_min (Y)", min_value=0.0, value=0.0, step=0.1, format="%0.6f")
            beta_max_y = ui.number_input("beta_max (Y)", min_value=0.0, value=1000.0, step=0.1, format="%0.6f")
            nufft_y = ui.checkbox("Use NUFFT (Y)", value=False)
            time_y = ui.selectbox("NUFFT time mode (Y)", options=["phase", "position"], index=0)
            log_y = ui.checkbox("Apply log10 (Y)", value=False)

        plot = ui.checkbox("Show joined spectrum plots", value=True)
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
                "length": int(length_x),
                "offset": int(offset_x),
                "use_rise": rise_x,
                "window_order": float(window_x),
                "f_min": float(f_min_x) if use_range_x else None,
                "f_max": float(f_max_x) if use_range_x else None,
                "beta_min": float(beta_min_x),
                "beta_max": float(beta_max_x),
                "use_nufft": nufft_x,
                "time_mode": time_x,
                "use_log": log_x,
            },
            "y": {
                "length": int(length_y),
                "offset": int(offset_y),
                "use_rise": rise_y,
                "window_order": float(window_y),
                "f_min": float(f_min_y) if use_range_y else None,
                "f_max": float(f_max_y) if use_range_y else None,
                "beta_min": float(beta_min_y),
                "beta_max": float(beta_max_y),
                "use_nufft": nufft_y,
                "time_mode": time_y,
                "use_log": log_y,
            },
        }

        results: dict[str, dict[str, object]] = {}
        with st.spinner("Running joined spectrum computation..."):
            try:
                for plane in planes:
                    option = plane_options[plane]
                    results[plane] = _compute_joined_spectrum(
                        plane=plane,
                        length=int(option["length"]),
                        offset=int(option["offset"]),
                        use_rise=bool(option["use_rise"]),
                        transform=transform,
                        filter_type=filter_type,
                        rank=int(rank),
                        svd_type=svd_type,
                        buffer=int(buffer),
                        count=int(count),
                        window_order=float(option["window_order"]),
                        f_min=option["f_min"],  # type: ignore[arg-type]
                        f_max=option["f_max"],  # type: ignore[arg-type]
                        beta_min=float(option["beta_min"]),
                        beta_max=float(option["beta_max"]),
                        use_nufft=bool(option["use_nufft"]),
                        time_mode=str(option["time_mode"]),
                        use_log=bool(option["use_log"]),
                        prefix=prefix_name,
                        data_prefix=data_prefix,
                        tango=tango,
                        device=device_name,
                        dtype_name=dtype_name,
                        verbose=verbose,
                        circumference=float(circumference),
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
        c2.metric("Planes", value=", ".join(plane.upper() for plane in planes))
        c3.metric("Grid size", value=int(len(first["grid"])))

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
            st.subheader("Combined Spectrum")
            for plane in planes:
                result = results[plane]
                f_min = result["f_min"]
                f_max = result["f_max"]
                f_range = "default" if f_min is None else f"[{f_min}, {f_max}]"
                st.caption(f"Plane {plane.upper()} | length={result['length']} | range={f_range}")
                spectrum_plot = px.line(
                    result["dataframe"],
                    x="FREQUENCY",
                    y=result["axis"],
                    title=f"{result['time']}: COMBINED SPECTRUM",
                )
                st.plotly_chart(spectrum_plot, use_container_width=True)
        else:
            st.subheader("Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["dataframe"], use_container_width=True)


if __name__ == "__main__":
    main()
