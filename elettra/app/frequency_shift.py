#!/usr/bin/env python3

"""script/frequency_shift.py"""

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
from harmonica.anomaly import score, threshold
from harmonica.cs import factory
from harmonica.data import Data
from harmonica.filter import Filter
from harmonica.frequency import Frequency
from harmonica.statistics import biweight_midvariance, median, rescale, standardize, weighted_mean, weighted_variance
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


def _compute_frequency_shift(
    *,
    plane: str,
    sample_length: int,
    load_length: int,
    shift_step: int,
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
    method: str,
    clean: bool,
    factor: float,
    process: str,
    limit: int,
    flip: bool,
    plot: bool,
    noise_plot: bool,
    prefix: str,
    data_prefix: str,
    tango: bool,
    device: str,
    dtype_name: str,
    update: bool,
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
    if sample_length <= 0:
        raise RuntimeError("Sample length must be positive.")
    if load_length <= 0 or load_length > LIMIT:
        raise RuntimeError(f"Invalid load={load_length}, expected 1..{LIMIT}.")
    if sample_length > load_length:
        raise RuntimeError(f"Sample length {sample_length} should be <= load length {load_length}.")
    if shift_step <= 0:
        raise RuntimeError("Shift step must be positive.")
    if offset < 0:
        raise RuntimeError(f"Invalid offset={offset}, expected a non-negative value.")
    if load_length + offset > LIMIT:
        raise RuntimeError(f"Invalid load+offset={load_length + offset}, expected <= {LIMIT}.")
    if window_order < 0.0:
        raise RuntimeError("Window order must be greater than or equal to zero.")
    if limit <= 0:
        raise RuntimeError("Noise estimation limit must be positive.")

    cs, bpm = _load_active_bpm(prefix, tango)

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after filtering.")

    pv_prefix = prefix if not data_prefix else data_prefix
    pv_list = [pv_make(name, plane, prefix=pv_prefix) for name in bpm]
    pv_rise = [*bpm.values()]

    shift = 0
    if use_rise:
        if min(pv_rise) < 0:
            raise RuntimeError("Rise values are expected to be non-negative.")
        shift = max(pv_rise)
        if load_length + offset + shift > LIMIT:
            raise RuntimeError(
                f"Invalid load+offset+shift={load_length + offset + shift}, expected <= {LIMIT}."
            )

    total = load_length + offset + shift
    win = Window(load_length, "cosine_window", window_order, dtype=dtype, device=device)
    matrix = np.asarray([cs.get(pv) for pv in pv_list])
    raw = torch.tensor(matrix, dtype=dtype, device=device)
    raw = torch.stack([signal[:total] for signal in raw])
    if use_rise:
        raw = torch.stack([signal[offset + rise : offset + rise + load_length] for signal, rise in zip(raw, pv_rise)])
    else:
        raw = raw[:, offset : offset + load_length]
    tbt = Data.from_data(win, raw)

    if transform == "mean":
        tbt.window_remove_mean()
    elif transform == "median":
        tbt.work.sub_(tbt.median())
    elif transform == "normalize":
        tbt.normalize()

    noise = None
    if process == "noise":
        table = []
        sample_win = Window(sample_length)
        for signal in tbt.work:
            d = Data.from_data(sample_win, Data.make_matrix(sample_length, shift_step, signal))
            flt = Filter(d)
            _, sigma = flt.estimate_noise(limit=limit)
            table.append(sigma)
        noise = torch.stack(table)

    if filter_type == "svd":
        flt = Filter(tbt)
        flt.filter_svd(rank=rank)
    elif filter_type == "hankel":
        flt = Filter(tbt)
        flt.filter_svd(rank=rank)
        flt.filter_hankel(rank=rank, random=(svd_type == "randomized"), buffer=buffer, count=count)

    freq = Frequency(tbt, pad=load_length + 2 * pad)
    frequency = freq.compute_shifted_frequency(
        sample_length,
        shift_step,
        method=method,
        name="cosine_window",
        order=window_order,
        f_range=(f_min, f_max),
    )
    step_count = len(frequency.T)

    if clean:
        data = standardize(frequency.flatten(), center_estimator=median, spread_estimator=biweight_midvariance)
        factor_tensor = torch.tensor(factor, dtype=dtype, device=device)
        mask = threshold(data, -factor_tensor, +factor_tensor).reshape_as(frequency)
        mark = 0.5 >= rescale(score(tbt.size, mask.flatten()).to(dtype), scale_min=0.0, scale_max=1.0).nan_to_num()
    else:
        mask = torch.ones_like(frequency)
        mark = torch.ones(tbt.size, dtype=dtype, device=device)

    if process == "none":
        signal_center = weighted_mean(frequency, weight=mask)
        signal_spread = weighted_variance(frequency, weight=mask, center=signal_center).sqrt()
        weight = mark / signal_spread**2
        center = weighted_mean(signal_center, weight=weight)
        spread = weighted_variance(signal_center, weight=weight, center=center).sqrt()
    elif process == "noise":
        if noise is None:
            raise RuntimeError("Noise data is unavailable for process='noise'.")
        weight = mask / noise**2
        weight = weight / weight.sum(-1, keepdim=True)
        signal_center = weighted_mean(frequency, weight=weight)
        signal_spread = weighted_variance(frequency, weight=weight, center=signal_center).sqrt()
        weight = mark / signal_spread**2
        center = weighted_mean(signal_center, weight=weight)
        spread = weighted_variance(signal_center, weight=weight, center=center).sqrt()
    else:
        raise RuntimeError(f"Unsupported process: {process}.")

    if flip:
        frequency = 1.0 - frequency
        signal_center = 1.0 - signal_center
        center = 1.0 - center

    output = frequency.cpu().numpy()
    signal_center_numpy = signal_center.cpu().numpy()
    signal_spread_numpy = signal_spread.cpu().numpy()
    center_numpy = float(center.cpu().numpy())
    spread_numpy = float(spread.cpu().numpy())
    noise_numpy = noise.cpu().numpy() if noise is not None else None

    flags = mark.to(torch.bool).logical_not().cpu().numpy()
    shift_df = pd.concat(
        [pd.DataFrame({"STEP": range(1, step_count + 1), "BPM": name, "FREQUENCY": output[i]}) for i, name in enumerate(bpm)],
        ignore_index=True,
    )
    summary_df = pd.DataFrame(
        {
            "BPM": [*bpm.keys()],
            "FREQUENCY": signal_center_numpy,
            "ERROR": signal_spread_numpy,
            "FLAGGED": flags,
            "FLAG": (~flags).astype(np.float64),
        }
    )

    noise_df = None
    if noise_numpy is not None:
        noise_df = pd.concat(
            [pd.DataFrame({"STEP": range(1, step_count + 1), "BPM": name, "NOISE": noise_numpy[i]}) for i, name in enumerate(bpm)],
            ignore_index=True,
        )

    saved_file = None
    if save:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        saved_file = Path(f"frequency_shifted_plane_{plane}_length_{sample_length}_time_({timestamp}).npy")
        np.save(saved_file, output)

    if update:
        target = plane.upper()
        cs.set(f"{prefix}:FREQUENCY:VALUE:{target}", center_numpy)
        cs.set(f"{prefix}:FREQUENCY:ERROR:{target}", spread_numpy)
        for name, value, error in zip(bpm, signal_center_numpy, signal_spread_numpy):
            cs.set(f"{prefix}:{name}:FREQUENCY:VALUE:{target}", float(value))
            cs.set(f"{prefix}:{name}:FREQUENCY:ERROR:{target}", float(error))

    return {
        "time": time_label,
        "size": len(bpm),
        "step_count": step_count,
        "shift_df": shift_df,
        "summary_df": summary_df,
        "noise_df": noise_df,
        "center": center_numpy,
        "spread": spread_numpy,
        "saved_file": saved_file,
        "plot": plot,
        "noise_plot": noise_plot,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": pv_list,
        "output_mean": float(np.mean(output)),
        "output_std": float(np.std(output)),
    }


def main() -> None:
    st.set_page_config(page_title="Frequency (shift)", layout="wide")
    st.title("Frequency (shift)")

    try:
        app_config: AppConfig = load_app_config("frequency_shift")
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

    with st.sidebar.form("frequency_shift_form", enter_to_submit=False):
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
            sample_length_x = ui.number_input("Sample length (X)", min_value=1, max_value=LIMIT, value=512, step=1)
            load_length_x = ui.number_input("Load length (X)", min_value=1, max_value=LIMIT, value=1024, step=1)
            shift_step_x = ui.number_input("Shift step (X)", min_value=1, value=8, step=1)
            offset_x = ui.number_input("Offset (X)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_x = ui.checkbox("Use rise data (X)", value=False)
            save_x = ui.checkbox("Save output (X)", value=False)
            update_x = ui.checkbox("Update PVs (X)", value=False)
            window_x = ui.number_input("Window order (X)", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            pad_x = ui.number_input("Zero padding (X)", min_value=0, value=0, step=1)
            f_min_x = ui.number_input("f_min (X)", min_value=0.0, max_value=0.5, value=0.0, step=0.01, format="%0.6f")
            f_max_x = ui.number_input("f_max (X)", min_value=0.0, max_value=0.5, value=0.5, step=0.01, format="%0.6f")
            method_x = ui.selectbox("Method (X)", options=["parabola", "ffrft", "fft"], index=0)
            clean_x = ui.checkbox("Clean frequencies (X)", value=False)
            factor_x = ui.number_input("Clean factor (X)", min_value=0.0, value=5.0, step=0.1, format="%0.4f")
            process_x = ui.selectbox("Process (X)", options=["none", "noise"], index=0)
            limit_x = ui.number_input("Noise limit (X)", min_value=1, value=32, step=1)
            flip_x = ui.checkbox("Flip around 0.5 (X)", value=False)

        with st.expander("Plane Y options", expanded=False):
            sample_length_y = ui.number_input("Sample length (Y)", min_value=1, max_value=LIMIT, value=512, step=1)
            load_length_y = ui.number_input("Load length (Y)", min_value=1, max_value=LIMIT, value=1024, step=1)
            shift_step_y = ui.number_input("Shift step (Y)", min_value=1, value=8, step=1)
            offset_y = ui.number_input("Offset (Y)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_y = ui.checkbox("Use rise data (Y)", value=False)
            save_y = ui.checkbox("Save output (Y)", value=False)
            update_y = ui.checkbox("Update PVs (Y)", value=False)
            window_y = ui.number_input("Window order (Y)", min_value=0.0, value=0.0, step=0.1, format="%0.4f")
            pad_y = ui.number_input("Zero padding (Y)", min_value=0, value=0, step=1)
            f_min_y = ui.number_input("f_min (Y)", min_value=0.0, max_value=0.5, value=0.0, step=0.01, format="%0.6f")
            f_max_y = ui.number_input("f_max (Y)", min_value=0.0, max_value=0.5, value=0.5, step=0.01, format="%0.6f")
            method_y = ui.selectbox("Method (Y)", options=["parabola", "ffrft", "fft"], index=0)
            clean_y = ui.checkbox("Clean frequencies (Y)", value=False)
            factor_y = ui.number_input("Clean factor (Y)", min_value=0.0, value=5.0, step=0.1, format="%0.4f")
            process_y = ui.selectbox("Process (Y)", options=["none", "noise"], index=0)
            limit_y = ui.number_input("Noise limit (Y)", min_value=1, value=32, step=1)
            flip_y = ui.checkbox("Flip around 0.5 (Y)", value=False)

        plot = ui.checkbox("Show frequency plots", value=True)
        noise_plot = ui.checkbox("Show shifted noise plots", value=False)
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
                "shift_step": int(shift_step_x),
                "offset": int(offset_x),
                "use_rise": rise_x,
                "save": save_x,
                "update": update_x,
                "window_order": float(window_x),
                "pad": int(pad_x),
                "f_min": float(f_min_x),
                "f_max": float(f_max_x),
                "method": method_x,
                "clean": clean_x,
                "factor": float(factor_x),
                "process": process_x,
                "limit": int(limit_x),
                "flip": flip_x,
            },
            "y": {
                "sample_length": int(sample_length_y),
                "load_length": int(load_length_y),
                "shift_step": int(shift_step_y),
                "offset": int(offset_y),
                "use_rise": rise_y,
                "save": save_y,
                "update": update_y,
                "window_order": float(window_y),
                "pad": int(pad_y),
                "f_min": float(f_min_y),
                "f_max": float(f_max_y),
                "method": method_y,
                "clean": clean_y,
                "factor": float(factor_y),
                "process": process_y,
                "limit": int(limit_y),
                "flip": flip_y,
            },
        }

        results: dict[str, dict[str, object]] = {}
        with st.spinner("Running shifted frequency computation..."):
            try:
                for plane in planes:
                    option = plane_options[plane]
                    results[plane] = _compute_frequency_shift(
                        plane=plane,
                        sample_length=int(option["sample_length"]),
                        load_length=int(option["load_length"]),
                        shift_step=int(option["shift_step"]),
                        offset=int(option["offset"]),
                        use_rise=bool(option["use_rise"]),
                        save=bool(option["save"]),
                        transform=transform,
                        filter_type=filter_type,
                        rank=int(rank),
                        svd_type=svd_type,
                        buffer=int(buffer),
                        count=int(count),
                        window_order=float(option["window_order"]),
                        pad=int(option["pad"]),
                        f_min=float(option["f_min"]),
                        f_max=float(option["f_max"]),
                        method=str(option["method"]),
                        clean=bool(option["clean"]),
                        factor=float(option["factor"]),
                        process=str(option["process"]),
                        limit=int(option["limit"]),
                        flip=bool(option["flip"]),
                        plot=plot,
                        noise_plot=noise_plot,
                        prefix=prefix_name,
                        data_prefix=data_prefix,
                        tango=tango,
                        device=device_name,
                        dtype_name=dtype_name,
                        update=bool(option["update"]),
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
        c2.metric("Planes", value=", ".join(plane.upper() for plane in planes))
        c3.metric("Shifted samples", value=int(first["step_count"]))

        for plane in planes:
            result = results[plane]
            st.write(
                f"{plane.upper()}: center={result['center']:.9f}, spread={result['spread']:.9f}, "
                f"mean={result['output_mean']:.9f}, std={result['output_std']:.9f}"
            )
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
            st.subheader("Shifted Frequency")
            for plane in planes:
                result = results[plane]
                st.caption(f"Plane {plane.upper()}")
                shifted_plot = px.scatter(
                    result["shift_df"],
                    x="STEP",
                    y="FREQUENCY",
                    color="BPM",
                    title=(
                        f"{result['time']}: FREQUENCY (SHIFTED) "
                        f"({result['output_mean']:.9f}, {result['output_std']:.9f})"
                    ),
                    opacity=0.75,
                    marginal_y="box",
                )
                shifted_plot.add_hline(result["center"] - result["spread"], line_color="black", line_dash="dash", line_width=1.0)
                shifted_plot.add_hline(result["center"], line_color="black", line_dash="dash", line_width=1.0)
                shifted_plot.add_hline(result["center"] + result["spread"], line_color="black", line_dash="dash", line_width=1.0)
                st.plotly_chart(shifted_plot, use_container_width=True)

            st.subheader("Summary")
            for plane in planes:
                result = results[plane]
                st.caption(f"Plane {plane.upper()}")
                summary_plot = px.scatter(
                    result["summary_df"],
                    x="BPM",
                    y="FREQUENCY",
                    error_y="ERROR",
                    color="FLAGGED",
                    color_discrete_map={False: "blue", True: "red"},
                    hover_data=["BPM", "FREQUENCY", "ERROR", "FLAG"],
                    title=f"{result['time']}: FREQUENCY (SUMMARY)",
                    opacity=0.75,
                    marginal_y="box",
                )
                summary_plot.add_hline(result["center"] - result["spread"], line_color="black", line_dash="dash", line_width=1.0)
                summary_plot.add_hline(result["center"], line_color="black", line_dash="dash", line_width=1.0)
                summary_plot.add_hline(result["center"] + result["spread"], line_color="black", line_dash="dash", line_width=1.0)
                summary_plot.update_traces(marker={"size": 10})
                st.plotly_chart(summary_plot, use_container_width=True)

            if noise_plot:
                st.subheader("Noise")
                for plane in planes:
                    result = results[plane]
                    if result["noise_df"] is None:
                        continue
                    st.caption(f"Plane {plane.upper()}")
                    noise_fig = px.scatter(
                        result["noise_df"],
                        x="STEP",
                        y="NOISE",
                        color="BPM",
                        title=f"{result['time']}: NOISE (SHIFTED)",
                        opacity=0.75,
                    )
                    st.plotly_chart(noise_fig, use_container_width=True)
        else:
            st.subheader("Shifted Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["shift_df"], use_container_width=True)

            st.subheader("Summary Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["summary_df"], use_container_width=True)


if __name__ == "__main__":
    main()
