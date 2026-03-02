#!/usr/bin/env python3

"""script/amplitude.py"""

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
from harmonica.decomposition import Decomposition
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


def _compute_amplitude(
    *,
    plane: str,
    sample_length: int,
    load_length: int,
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
    limit: int,
    use_error: bool,
    use_shift: bool,
    shift_count: int,
    shift_step: int,
    method: str,
    use_dht: bool,
    drop: int,
    plot: bool,
    use_snr: bool,
    snr_lg: bool,
    coupled: bool,
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

    if snr_lg and not use_snr:
        raise RuntimeError("lg option requires snr option.")

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
    if limit <= 0:
        raise RuntimeError("Noise estimation limit must be positive.")
    if shift_count <= 0:
        raise RuntimeError("Shift sample count must be positive.")
    if shift_step <= 0:
        raise RuntimeError("Shift sample step must be positive.")
    if use_dht and drop * 2 >= sample_length:
        raise RuntimeError("For DHT mode, 2*drop must be smaller than sample length.")

    cs, bpm = _load_active_bpm(prefix, tango)

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after filtering.")

    plane_up = plane.upper()
    target = {"X": "Y", "Y": "X"}[plane_up] if coupled else plane_up
    frequency_value = float(cs.get(f"{prefix}:FREQUENCY:VALUE:{target}"))
    frequency_error = float(cs.get(f"{prefix}:FREQUENCY:ERROR:{target}"))

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

    if not use_dht:
        dec = Decomposition(tbt)
        value, error, table = dec.harmonic_amplitude(
            frequency_value,
            length=sample_length,
            order=window_order,
            window="cosine_window",
            error=use_error,
            sigma_frequency=frequency_error,
            limit=limit,
            shift=use_shift,
            count=shift_count,
            step=shift_step,
            clean=True,
            factor=5.0,
            method=method,
        )
    else:
        dht = Frequency.dht(tbt.work[:, :sample_length])
        table = dht.abs()
        value = table[:, +drop:-drop].mean(-1)
        error = table[:, +drop:-drop].std(-1)

    amplitude = value.cpu().numpy()
    error_output = error.cpu().numpy() if error is not None else torch.zeros_like(value).cpu().numpy()

    snr = None
    if use_snr:
        noise = np.asarray([cs.get(f"{prefix}:{name}:NOISE:{plane_up}") for name in bpm], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            if snr_lg:
                ratio = amplitude / noise
                snr = np.where(ratio > 0.0, 20.0 * np.log10(ratio), np.nan)
            else:
                snr = (amplitude / noise) ** 2

    saved_file = None
    snr_file = None
    if save:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        mode = "amplitude_coupled" if coupled else "amplitude"
        saved_file = Path(f"{mode}_plane_{plane}_length_{sample_length}_time_({timestamp}).npy")
        np.save(saved_file, amplitude)
        if use_snr and snr is not None:
            suffix = "snr_lg" if snr_lg else "snr"
            mode = f"{suffix}_coupled" if coupled else suffix
            snr_file = Path(f"{mode}_plane_{plane}_length_{sample_length}_time_({timestamp}).npy")
            np.save(snr_file, snr)

    if update and coupled:
        raise RuntimeError("Update with coupled mode is not supported for amplitude data.")

    if update and not coupled:
        for name, amp, sigma in zip(bpm, amplitude, error_output):
            cs.set(f"{prefix}:{name}:AMPLITUDE:VALUE:{plane_up}", float(amp))
            cs.set(f"{prefix}:{name}:AMPLITUDE:ERROR:{plane_up}", float(sigma))
        if use_snr and snr is not None:
            for name, ratio in zip(bpm, snr):
                cs.set(f"{prefix}:{name}:SNR:{plane_up}", float(ratio))

    mode_text = f"{plane_up}, FREQUENCY={target}" if coupled else plane_up
    summary_df = pd.DataFrame(
        {
            "BPM": [*bpm.keys()],
            "AMPLITUDE": amplitude,
            "ERROR": error_output,
        }
    )
    table_df = None
    if table is not None:
        _, step = table.shape
        table_df = pd.concat(
            [
                pd.DataFrame({"STEP": range(1, step + 1), "BPM": name, "AMPLITUDE": table[i].cpu().numpy()})
                for i, name in enumerate(bpm)
            ],
            ignore_index=True,
        )
    snr_df = None
    if use_snr and snr is not None:
        snr_df = pd.DataFrame({"BPM": [*bpm.keys()], "SNR": snr})

    return {
        "time": time_label,
        "size": len(bpm),
        "mode_text": mode_text,
        "table_df": table_df,
        "summary_df": summary_df,
        "snr_df": snr_df,
        "saved_file": saved_file,
        "snr_file": snr_file,
        "plot": plot,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": pv_list,
    }


def main() -> None:
    st.set_page_config(page_title="Amplitude", layout="wide")
    st.title("Amplitude")

    try:
        app_config: AppConfig = load_app_config("amplitude")
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

    with st.sidebar.form("amplitude_form", enter_to_submit=False):
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
            sample_length_x = ui.number_input("Sample length (X)", min_value=1, max_value=LIMIT, value=32, step=1)
            load_length_x = ui.number_input("Load length (X)", min_value=1, max_value=LIMIT, value=64, step=1)
            offset_x = ui.number_input("Offset (X)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_x = ui.checkbox("Use rise data (X)", value=False)
            save_x = ui.checkbox("Save output (X)", value=False)
            update_x = ui.checkbox("Update PVs (X)", value=False)
            window_x = ui.number_input("Window order (X)", min_value=0.0, value=1.0, step=0.1, format="%0.4f")
            limit_x = ui.number_input("Noise limit (X)", min_value=1, value=32, step=1)
            use_error_x = ui.checkbox("Propagate frequency error (X)", value=False)
            use_shift_x = ui.checkbox("Use shifted samples (X)", value=False)
            shift_count_x = ui.number_input("Shift sample count (X)", min_value=1, value=64, step=1)
            shift_step_x = ui.number_input("Shift sample step (X)", min_value=1, value=8, step=1)
            method_x = ui.selectbox("Shifted method (X)", options=["none", "noise", "error"], index=0)
            dht_x = ui.checkbox("Use DHT estimator (X)", value=False)
            drop_x = ui.number_input("DHT drop (X)", min_value=0, value=32, step=1)
            use_snr_x = ui.checkbox("Compute SNR (X)", value=False)
            snr_lg_x = ui.checkbox("Compute SNR in dB (X)", value=False)
            coupled_x = ui.checkbox("Use coupled frequency (X->Y)", value=False)

        with st.expander("Plane Y options", expanded=False):
            sample_length_y = ui.number_input("Sample length (Y)", min_value=1, max_value=LIMIT, value=32, step=1)
            load_length_y = ui.number_input("Load length (Y)", min_value=1, max_value=LIMIT, value=64, step=1)
            offset_y = ui.number_input("Offset (Y)", min_value=0, max_value=LIMIT, value=0, step=1)
            rise_y = ui.checkbox("Use rise data (Y)", value=False)
            save_y = ui.checkbox("Save output (Y)", value=False)
            update_y = ui.checkbox("Update PVs (Y)", value=False)
            window_y = ui.number_input("Window order (Y)", min_value=0.0, value=1.0, step=0.1, format="%0.4f")
            limit_y = ui.number_input("Noise limit (Y)", min_value=1, value=32, step=1)
            use_error_y = ui.checkbox("Propagate frequency error (Y)", value=False)
            use_shift_y = ui.checkbox("Use shifted samples (Y)", value=False)
            shift_count_y = ui.number_input("Shift sample count (Y)", min_value=1, value=64, step=1)
            shift_step_y = ui.number_input("Shift sample step (Y)", min_value=1, value=8, step=1)
            method_y = ui.selectbox("Shifted method (Y)", options=["none", "noise", "error"], index=0)
            dht_y = ui.checkbox("Use DHT estimator (Y)", value=False)
            drop_y = ui.number_input("DHT drop (Y)", min_value=0, value=32, step=1)
            use_snr_y = ui.checkbox("Compute SNR (Y)", value=False)
            snr_lg_y = ui.checkbox("Compute SNR in dB (Y)", value=False)
            coupled_y = ui.checkbox("Use coupled frequency (Y->X)", value=False)

        plot = ui.checkbox("Show amplitude plots", value=True)
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
                "save": save_x,
                "update": update_x,
                "window_order": float(window_x),
                "limit": int(limit_x),
                "use_error": use_error_x,
                "use_shift": use_shift_x,
                "shift_count": int(shift_count_x),
                "shift_step": int(shift_step_x),
                "method": method_x,
                "use_dht": dht_x,
                "drop": int(drop_x),
                "use_snr": use_snr_x,
                "snr_lg": snr_lg_x,
                "coupled": coupled_x,
            },
            "y": {
                "sample_length": int(sample_length_y),
                "load_length": int(load_length_y),
                "offset": int(offset_y),
                "use_rise": rise_y,
                "save": save_y,
                "update": update_y,
                "window_order": float(window_y),
                "limit": int(limit_y),
                "use_error": use_error_y,
                "use_shift": use_shift_y,
                "shift_count": int(shift_count_y),
                "shift_step": int(shift_step_y),
                "method": method_y,
                "use_dht": dht_y,
                "drop": int(drop_y),
                "use_snr": use_snr_y,
                "snr_lg": snr_lg_y,
                "coupled": coupled_y,
            },
        }

        results: dict[str, dict[str, object]] = {}
        with st.spinner("Running amplitude computation..."):
            try:
                for plane in planes:
                    option = plane_options[plane]
                    results[plane] = _compute_amplitude(
                        plane=plane,
                        sample_length=int(option["sample_length"]),
                        load_length=int(option["load_length"]),
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
                        limit=int(option["limit"]),
                        use_error=bool(option["use_error"]),
                        use_shift=bool(option["use_shift"]),
                        shift_count=int(option["shift_count"]),
                        shift_step=int(option["shift_step"]),
                        method=str(option["method"]),
                        use_dht=bool(option["use_dht"]),
                        drop=int(option["drop"]),
                        plot=plot,
                        use_snr=bool(option["use_snr"]),
                        snr_lg=bool(option["snr_lg"]),
                        coupled=bool(option["coupled"]),
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
        c1, c2 = st.columns(2)
        c1.metric("BPMs", value=int(first["size"]))
        c2.metric("Planes", value=", ".join(plane.upper() for plane in planes))

        for plane in planes:
            result = results[plane]
            if result["saved_file"] is not None:
                st.write(f"{plane.upper()} amplitude saved: `{result['saved_file']}`")
            if result["snr_file"] is not None:
                st.write(f"{plane.upper()} SNR saved: `{result['snr_file']}`")

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
            st.subheader("Amplitude Samples")
            for plane in planes:
                result = results[plane]
                if result["table_df"] is None:
                    continue
                st.caption(f"Plane {plane.upper()} ({result['mode_text']})")
                sample_plot = px.scatter(
                    result["table_df"],
                    x="STEP",
                    y="AMPLITUDE",
                    color="BPM",
                    title=f"{result['time']}: AMPLITUDE ({result['mode_text']})",
                    opacity=0.75,
                    marginal_y="box",
                )
                st.plotly_chart(sample_plot, use_container_width=True)

            st.subheader("Amplitude Summary")
            for plane in planes:
                result = results[plane]
                st.caption(f"Plane {plane.upper()} ({result['mode_text']})")
                summary_plot = px.scatter(
                    result["summary_df"],
                    x="BPM",
                    y="AMPLITUDE",
                    error_y="ERROR",
                    title=f"{result['time']}: AMPLITUDE ({result['mode_text']})",
                    opacity=0.75,
                    hover_data=["BPM", "AMPLITUDE", "ERROR"],
                )
                summary_plot.update_traces(mode="lines+markers", line={"width": 1.5}, marker={"size": 5})
                st.plotly_chart(summary_plot, use_container_width=True)

            st.subheader("SNR")
            for plane in planes:
                result = results[plane]
                if result["snr_df"] is None:
                    continue
                st.caption(f"Plane {plane.upper()} ({result['mode_text']})")
                snr_plot = px.scatter(
                    result["snr_df"],
                    x="BPM",
                    y="SNR",
                    title=f"{result['time']}: SNR ({result['mode_text']})",
                    opacity=0.75,
                    hover_data=["BPM", "SNR"],
                )
                snr_plot.update_traces(mode="lines+markers", line={"width": 1.5}, marker={"size": 5})
                st.plotly_chart(snr_plot, use_container_width=True)
        else:
            st.subheader("Amplitude Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["summary_df"], use_container_width=True)
            st.subheader("SNR Tables")
            for plane in planes:
                if results[plane]["snr_df"] is None:
                    continue
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["snr_df"], use_container_width=True)


if __name__ == "__main__":
    main()
