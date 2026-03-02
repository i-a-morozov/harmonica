#!/usr/bin/env python3

"""script/ratio.py"""

from __future__ import annotations

from configuration import AppConfig, WidgetDefaults, load_app_config

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from harmonica.cs import factory
from harmonica.util import bpm_select


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



def _load_active_bpm(prefix: str, tango: bool) -> tuple[object, dict[str, None]]:
    cs = factory(target=("tango" if tango else "epics"))
    monitor_count = int(cs.get(f"{prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{prefix}:MONITOR:LIST")[:monitor_count]
    monitor_names = [_as_str(name) for name in monitor_names]
    monitor_flag = [int(cs.get(f"{prefix}:{name}:FLAG")) for name in monitor_names]
    bpm = {name: None for name, flag in zip(monitor_names, monitor_flag) if flag == 1}
    return cs, bpm


def _current_statistics(value: object) -> tuple[float, float]:
    if value is None:
        return np.nan, np.nan
    data = np.asarray(value, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan, np.nan
    return float(data.mean()), float(data.std())


def _compute_ratio(
    *,
    plot: bool,
    save: bool,
    update: bool,
    prefix: str,
    data_prefix: str,
    tango: bool,
    verbose: bool,
    skip: list[str] | None,
    only: list[str] | None,
) -> dict[str, object]:
    time_label = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    cs, bpm = _load_active_bpm(prefix, tango)
    source_prefix = prefix if not data_prefix else data_prefix

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after filtering.")

    names = [*bpm.keys()]

    pv_bx_a = [f"{source_prefix}:{name}:AMPLITUDE:BX:VALUE" for name in names]
    pv_sx_a = [f"{source_prefix}:{name}:AMPLITUDE:BX:ERROR" for name in names]
    pv_by_a = [f"{source_prefix}:{name}:AMPLITUDE:BY:VALUE" for name in names]
    pv_sy_a = [f"{source_prefix}:{name}:AMPLITUDE:BY:ERROR" for name in names]
    bx_a = np.asarray([cs.get(pv) for pv in pv_bx_a], dtype=np.float64)
    sx_a = np.asarray([cs.get(pv) for pv in pv_sx_a], dtype=np.float64)
    by_a = np.asarray([cs.get(pv) for pv in pv_by_a], dtype=np.float64)
    sy_a = np.asarray([cs.get(pv) for pv in pv_sy_a], dtype=np.float64)

    pv_bx = [f"{source_prefix}:{name}:PHASE:BX:VALUE" for name in names]
    pv_sx = [f"{source_prefix}:{name}:PHASE:BX:ERROR" for name in names]
    pv_by = [f"{source_prefix}:{name}:PHASE:BY:VALUE" for name in names]
    pv_sy = [f"{source_prefix}:{name}:PHASE:BY:ERROR" for name in names]
    bx = np.asarray([cs.get(pv) for pv in pv_bx], dtype=np.float64)
    sx = np.asarray([cs.get(pv) for pv in pv_sx], dtype=np.float64)
    by = np.asarray([cs.get(pv) for pv in pv_by], dtype=np.float64)
    sy = np.asarray([cs.get(pv) for pv in pv_sy], dtype=np.float64)

    pv_i = [f"{source_prefix}:{name}:DATA:I" for name in names]
    current = [cs.get(pv) for pv in pv_i]
    value_i, error_i = zip(*(_current_statistics(value) for value in current))
    value_i = np.asarray(value_i, dtype=np.float64)
    error_i = np.asarray(error_i, dtype=np.float64)

    valid_i = np.isfinite(value_i)
    if valid_i.any():
        median_i = np.nanmedian(value_i)
        mad_i = np.nanmedian(np.abs(value_i - median_i))
        scale_i = 1.4826 * mad_i if mad_i > 0.0 else np.nanstd(value_i)
        if np.isfinite(scale_i) and scale_i > 0.0:
            z_i = (value_i - median_i) / scale_i
            mask_i = np.abs(z_i) > 5.0
        else:
            mask_i = np.zeros_like(value_i, dtype=bool)
    else:
        mask_i = np.zeros_like(value_i, dtype=bool)

    weight_i = np.zeros_like(value_i, dtype=np.float64)
    good_i = (~mask_i) & np.isfinite(value_i) & np.isfinite(error_i) & (error_i > 0.0)
    weight_i[good_i] = 1.0 / error_i[good_i] ** 2
    if weight_i.sum() > 0.0:
        center_i = float(np.average(value_i, weights=weight_i))
        spread_i = float(np.sqrt(np.average((value_i - center_i) ** 2, weights=weight_i)))
    else:
        center_i = float(np.nanmean(value_i))
        spread_i = float(np.nanstd(value_i))

    with np.errstate(divide="ignore", invalid="ignore"):
        rx = bx_a / bx
        ry = by_a / by
        sigma_rx = np.sqrt((sx_a / bx) ** 2 + (bx_a * sx / bx**2) ** 2)
        sigma_ry = np.sqrt((sy_a / by) ** 2 + (by_a * sy / by**2) ** 2)

    center_rx = float(np.nanmean(rx))
    spread_rx = float(np.nanstd(rx))
    center_ry = float(np.nanmean(ry))
    spread_ry = float(np.nanstd(ry))

    saved_file = None
    if save:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        saved_file = Path(f"ratio_time_({timestamp}).npy")
        np.save(saved_file, np.array([rx, sigma_rx, ry, sigma_ry]))

    if update:
        for name, value, error in zip(names, rx, sigma_rx):
            cs.set(f"{prefix}:{name}:RATIO:VALUE:X", float(value))
            cs.set(f"{prefix}:{name}:RATIO:ERROR:X", float(error))
        for name, value, error in zip(names, ry, sigma_ry):
            cs.set(f"{prefix}:{name}:RATIO:VALUE:Y", float(value))
            cs.set(f"{prefix}:{name}:RATIO:ERROR:Y", float(error))

    summary_df = pd.DataFrame(
        {
            "BPM": names,
            "I": value_i,
            "SIGMA_I": error_i,
            "MASK_I": mask_i,
            "RX": rx,
            "SIGMA_RX": sigma_rx,
            "RY": ry,
            "SIGMA_RY": sigma_ry,
        }
    )

    return {
        "time": time_label,
        "size": len(names),
        "summary_df": summary_df,
        "center_i": center_i,
        "spread_i": spread_i,
        "center_rx": center_rx,
        "spread_rx": spread_rx,
        "center_ry": center_ry,
        "spread_ry": spread_ry,
        "saved_file": saved_file,
        "plot": plot,
        "verbose": verbose,
        "bpm": bpm,
        "pv_list": [
            *pv_bx_a,
            *pv_sx_a,
            *pv_by_a,
            *pv_sy_a,
            *pv_bx,
            *pv_sx,
            *pv_by,
            *pv_sy,
            *pv_i,
        ],
    }


def _make_line_figure(
    df: pd.DataFrame,
    *,
    time_label: str,
    y_name: str,
    y_error_name: str,
    title_suffix: str,
    reference: float | None = None,
    center: float | None = None,
    spread: float | None = None,
    mask_name: str | None = None,
) -> go.Figure:
    fig = px.scatter(
        df,
        x="BPM",
        y=y_name,
        error_y=y_error_name,
        color_discrete_sequence=["blue"],
        title=f"{time_label}: {title_suffix}",
        opacity=0.8,
    )
    fig.update_traces(mode="lines+markers", marker={"size": 8})

    if mask_name is not None:
        bad = df[df[mask_name]]
        if not bad.empty:
            fig.add_trace(
                go.Scatter(
                    x=bad["BPM"],
                    y=bad[y_name],
                    error_y={"type": "data", "array": bad[y_error_name].to_numpy()},
                    mode="markers",
                    marker={"size": 9, "color": "red"},
                    name=f"{y_name} (OUTLIER)",
                )
            )

    if reference is not None:
        fig.add_hline(y=reference, line_color="black", line_dash="dash", line_width=1.0)
    if center is not None and spread is not None:
        fig.add_hline(y=center - spread, line_color="black", line_dash="dash", line_width=1.0)
        fig.add_hline(y=center, line_color="black", line_dash="dash", line_width=1.0)
        fig.add_hline(y=center + spread, line_color="black", line_dash="dash", line_width=1.0)

    fig.update_layout(xaxis_title="BPM", yaxis_title=title_suffix)
    return fig


def main() -> None:
    st.set_page_config(page_title="Ratio", layout="wide")
    st.title("Ratio")

    try:
        app_config: AppConfig = load_app_config("ratio")
    except Exception as exception:
        st.error(str(exception))
        return

    prefix_name = app_config.global_.prefix
    tango = app_config.global_.control_system == "tango"

    ui = WidgetDefaults(app_config.script)

    st.sidebar.header("Controls")
    with st.sidebar.expander("Configuration", expanded=False):
        st.caption(f"Prefix: `{prefix_name}`")
        st.caption(f"Control system: `{app_config.global_.control_system}`")

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

    with st.sidebar.form("ratio_form", enter_to_submit=False):
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

        plot = ui.checkbox("Show ratio plots", value=True)
        save = ui.checkbox("Save output", value=False)
        data_prefix = ui.text_input("PV data prefix override", value="")
        update = ui.checkbox("Update PVs", value=False)
        verbose = ui.checkbox("Verbose output", value=False)

        run = st.form_submit_button("Run", type="primary")

    right_col, _ = st.columns([5, 1])
    with right_col:
        if not run:
            st.info("Set options on the left and click Run.")
            return

        patterns = _build_patterns(selected_names, regex_text)
        if selection_mode in {"skip", "only"} and not patterns:
            st.error("For skip/only mode, select BPM names and/or enter regex patterns.")
            return

        skip = patterns if selection_mode == "skip" else None
        only = patterns if selection_mode == "only" else None

        with st.spinner("Running ratio computation..."):
            try:
                result = _compute_ratio(
                    plot=plot,
                    save=save,
                    update=update,
                    prefix=prefix_name,
                    data_prefix=data_prefix,
                    tango=tango,
                    verbose=verbose,
                    skip=skip,
                    only=only,
                )
            except Exception as exception:  # pragma: no cover - depends on external CS
                st.error(str(exception))
                return

        st.success("Computation finished.")
        c1, c2, c3 = st.columns(3)
        c1.metric("BPMs", value=int(result["size"]))
        c2.metric("RX center", value=f"{result['center_rx']:.9f}")
        c3.metric("RY center", value=f"{result['center_ry']:.9f}")
        st.write(f"I center={result['center_i']:.9f}, spread={result['spread_i']:.9f}")
        st.write(f"RX spread={result['spread_rx']:.9f}, RY spread={result['spread_ry']:.9f}")
        if result["saved_file"] is not None:
            st.write(f"Saved: `{result['saved_file']}`")

        if result["plot"]:
            st.subheader("Current")
            fig = _make_line_figure(
                result["summary_df"],
                time_label=result["time"],
                y_name="I",
                y_error_name="SIGMA_I",
                title_suffix="I",
                center=float(result["center_i"]),
                spread=float(result["spread_i"]),
                mask_name="MASK_I",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Ratio X")
            fig = _make_line_figure(
                result["summary_df"],
                time_label=result["time"],
                y_name="RX",
                y_error_name="SIGMA_RX",
                title_suffix="BX_A/BX",
                reference=1.0,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Ratio Y")
            fig = _make_line_figure(
                result["summary_df"],
                time_label=result["time"],
                y_name="RY",
                y_error_name="SIGMA_RY",
                title_suffix="BY_A/BY",
                reference=1.0,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Table")
            st.dataframe(result["summary_df"], use_container_width=True)

        if result["verbose"]:
            with st.expander("Verbose details"):
                st.write(f"Time: {result['time']}")
                st.write("Monitor list:")
                for name in result["bpm"]:
                    st.write(f"- {name}")
                st.write("PV list:")
                for pv in result["pv_list"]:
                    st.write(f"- {pv}")


if __name__ == "__main__":
    main()
