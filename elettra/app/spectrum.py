#!/usr/bin/env python3

"""script/spectrum.py"""

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



def _load_active_bpm(prefix: str, tango: bool) -> tuple[object, dict[str, int]]:
    cs = factory(target=("tango" if tango else "epics"))
    monitor_count = int(cs.get(f"{prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{prefix}:MONITOR:LIST")[:monitor_count]
    monitor_names = [_as_str(name) for name in monitor_names]
    monitor_flag = np.asarray([cs.get(f"{prefix}:{name}:FLAG") for name in monitor_names])
    monitor_rise = np.asarray([cs.get(f"{prefix}:{name}:RISE") for name in monitor_names])
    bpm = {name: int(rise) for name, flag, rise in zip(monitor_names, monitor_flag, monitor_rise) if int(flag) == 1}
    return cs, bpm


def _compute_spectrum(
    *,
    plane: str,
    length: int,
    offset: int,
    use_rise: bool,
    save: bool,
    transform: str,
    filter_type: str,
    rank: int,
    svd_type: str,
    buffer: int,
    count: int,
    window_order: float,
    pad: int,
    f_min: float,
    f_max: float,
    use_log: bool,
    flip: bool,
    plot: bool,
    show_map: bool,
    average: bool,
    peaks: int,
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

    if not (0.0 <= f_min <= 0.5):
        raise RuntimeError(f"Invalid f_min={f_min}, expected 0.0..0.5.")
    if not (0.0 <= f_max <= 0.5):
        raise RuntimeError(f"Invalid f_max={f_max}, expected 0.0..0.5.")
    if f_max < f_min:
        raise RuntimeError(f"Invalid range: f_max={f_max} must be >= f_min={f_min}.")
    if pad < 0:
        raise RuntimeError(f"Invalid pad={pad}, expected a non-negative value.")
    if (f_min, f_max) != (0.0, 0.5) and pad != 0:
        raise RuntimeError("Custom frequency range and padding cannot be used together.")

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

    freq = Frequency(tbt, pad=length + 2 * pad)
    if window_order > 0.0:
        freq.data.window_remove_mean()
        freq.data.window_apply()

    freq.fft_get_spectrum()
    grid = freq.fft_grid
    output = freq.fft_spectrum

    if (f_min, f_max) != (0.0, 0.5):
        span = f_max - f_min
        center = f_min + 0.5 * span
        freq.ffrft_get_spectrum(center=center, span=span)
        grid = freq.ffrft_get_grid()
        output = freq.ffrft_spectrum

    grid_numpy = grid.cpu().numpy()[1:]
    data_numpy = output.cpu().numpy()[:, 1:]

    mean_grid_numpy = None
    mean_data_numpy = None
    if average:
        freq("ffrft")
        mean_grid, mean_data = freq.compute_mean_spectrum(log=use_log)
        mean_grid_numpy = mean_grid.cpu().numpy()[1:]
        mean_data_numpy = mean_data.cpu().numpy()[1:]

    if flip:
        grid_numpy = 1.0 - grid_numpy[::-1]
        data_numpy = data_numpy[:, ::-1]
        if average and mean_grid_numpy is not None and mean_data_numpy is not None:
            mean_grid_numpy = 1.0 - mean_grid_numpy[::-1]
            mean_data_numpy = mean_data_numpy[::-1]

    if use_log:
        data_numpy = np.log10(data_numpy + 1.0e-12)

    peak_grid = None
    peak_data = None
    if average and peaks > 0 and mean_data_numpy is not None:
        from scipy.signal import find_peaks

        peak_index, *_ = find_peaks(mean_data_numpy)
        if len(peak_index):
            peak_table = np.array([mean_grid_numpy[peak_index], mean_data_numpy[peak_index]]).T
            best = np.array(sorted(peak_table, key=lambda item: item[1], reverse=True)[:peaks])
            peak_grid = best[:, 0]
            peak_data = best[:, 1]

    axis = f"DTFT({plane.upper()})"
    scatter_df = pd.concat(
        [pd.DataFrame({"FREQUENCY": grid_numpy, "BPM": name, axis: data_numpy[i]}) for i, name in enumerate(bpm)],
        ignore_index=True,
    )

    average_df = None
    if average and mean_grid_numpy is not None and mean_data_numpy is not None:
        average_df = pd.DataFrame({"FREQUENCY": mean_grid_numpy, axis: mean_data_numpy})

    saved_file = None
    if save:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        saved_file = Path(f"spectrum_plane_{plane}_length_{length}_time_({timestamp}).npy")
        np.save(saved_file, data_numpy)

    return {
        "time": time_label,
        "size": len(bpm),
        "axis": axis,
        "grid": grid_numpy,
        "matrix": data_numpy,
        "scatter_df": scatter_df,
        "average_df": average_df,
        "peak_grid": peak_grid,
        "peak_data": peak_data,
        "show_map": show_map,
        "average": average,
        "plot": plot,
        "saved_file": saved_file,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": pv_list,
    }


def main() -> None:
    st.set_page_config(page_title="Spectrum", layout="wide")
    st.title("Spectrum")

    try:
        app_config: AppConfig = load_app_config("spectrum")
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

    with st.sidebar.form("spectrum_form", enter_to_submit=False):
        st.caption("Planes")
        plane_col_x, plane_col_y = st.columns(2)
        plane_x = plane_col_x.checkbox("X", value=bool(app_config.script.x.get("enabled", True)))
        plane_y = plane_col_y.checkbox("Y", value=bool(app_config.script.y.get("enabled", False)))
        length = ui.number_input("Length", min_value=1, max_value=LIMIT, value=1024, step=1)
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
            transform = ui.radio(
                "Transform",
                options=["none", "mean", "median", "normalize"],
                index=0,
                horizontal=True,
            )
            filter_type = ui.selectbox("Filter", options=["none", "svd", "hankel"], index=0)
            rank = ui.number_input("Filter rank", min_value=1, value=8, step=1)
            svd_type = ui.selectbox(
                "Hankel SVD type",
                options=["randomized", "full"],
                index=0,
            )
            buffer = ui.number_input(
                "Hankel randomized buffer",
                min_value=1,
                value=16,
                step=1,
            )
            count = ui.number_input(
                "Hankel randomized iterations",
                min_value=1,
                value=16,
                step=1,
            )
            window_order = ui.number_input("Window order", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            pad = ui.number_input("Zero padding (each side)", min_value=0, value=0, step=1)
            f_min = ui.number_input("f_min", min_value=0.0, max_value=0.5, value=0.0, step=0.01, format="%0.6f")
            f_max = ui.number_input("f_max", min_value=0.0, max_value=0.5, value=0.5, step=0.01, format="%0.6f")
            use_log = ui.checkbox("Apply log10 scale", value=False)
            flip = ui.checkbox("Flip around 0.5", value=False)

        plot = ui.checkbox("Show spectrum plot", value=True)
        show_map = ui.checkbox("Show heat map", value=False)
        average = ui.checkbox("Show average spectrum", value=False)
        peaks = ui.number_input("Average peaks", min_value=0, value=1, step=1)
        save = ui.checkbox("Save `.npy` output", value=False)
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

        results: dict[str, dict[str, object]] = {}
        with st.spinner("Running spectrum computation..."):
            try:
                for plane in planes:
                    results[plane] = _compute_spectrum(
                        plane=plane,
                        length=int(length),
                        offset=int(offset),
                        use_rise=use_rise,
                        save=save,
                        transform=transform,
                        filter_type=filter_type,
                        rank=int(rank),
                        svd_type=svd_type,
                        buffer=int(buffer),
                        count=int(count),
                        window_order=float(window_order),
                        pad=int(pad),
                        f_min=float(f_min),
                        f_max=float(f_max),
                        use_log=use_log,
                        flip=flip,
                        plot=plot,
                        show_map=show_map,
                        average=average,
                        peaks=int(peaks),
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
        c2.metric("Frequency bins", value=int(first["matrix"].shape[1]))
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
            st.subheader("Spectrum")
            for plane in planes:
                result = results[plane]
                st.caption(f"Plane {plane.upper()}")
                spectrum_plot = px.scatter(
                    result["scatter_df"],
                    x="FREQUENCY",
                    y=result["axis"],
                    color="BPM",
                    title=f"{result['time']}: SPECTRUM",
                    opacity=0.75,
                )
                st.plotly_chart(spectrum_plot, use_container_width=True)

            if show_map:
                st.subheader("Map")
                for plane in planes:
                    result = results[plane]
                    st.caption(f"Plane {plane.upper()}")
                    map_plot = px.imshow(
                        result["matrix"],
                        labels=dict(x="FREQUENCY", y="BPM", color=result["axis"]),
                        x=result["grid"],
                        y=[*result["bpm"].keys()],
                        aspect=0.5,
                        title=f"{result['time']}: SPECTRUM (MAP)",
                    )
                    map_plot.update_layout(dragmode="zoom")
                    st.plotly_chart(map_plot, use_container_width=True)

            if average:
                st.subheader("Average")
                for plane in planes:
                    result = results[plane]
                    if result["average_df"] is None:
                        continue
                    st.caption(f"Plane {plane.upper()}")
                    avg_plot = px.scatter(
                        result["average_df"],
                        x="FREQUENCY",
                        y=result["axis"],
                        title=f"{result['time']}: SPECTRUM (AVERAGE)",
                    )
                    if result["peak_grid"] is not None and result["peak_data"] is not None:
                        avg_plot.add_scatter(
                            x=result["peak_grid"],
                            y=result["peak_data"],
                            mode="markers",
                            marker=dict(color="red", size=10),
                            showlegend=False,
                            name="PEAK",
                        )
                    st.plotly_chart(avg_plot, use_container_width=True)
        else:
            st.subheader("Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["scatter_df"], use_container_width=True)


if __name__ == "__main__":
    main()
