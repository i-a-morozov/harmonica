#!/usr/bin/env python3

"""Streamlit GUI for script/twiss_phase.py."""

from __future__ import annotations

from configuration import AppConfig, WidgetDefaults, load_app_config

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from harmonica.cs import factory
from harmonica.model import Model
from harmonica.table import Table
from harmonica.twiss import Twiss


def _as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)



def _resolve_model_path(path: str) -> str:
    model_path = path
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        local = os.path.join(os.path.dirname(__file__), model_path)
        parent = os.path.join(os.path.dirname(__file__), "..", model_path)
        if os.path.exists(local):
            model_path = local
        elif os.path.exists(parent):
            model_path = parent
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {path}")
    return model_path


def _make_type_labels(size: int, monitor_index: np.ndarray, virtual_index: np.ndarray) -> np.ndarray:
    labels = np.asarray(["MODEL"] * size, dtype=object)
    if monitor_index.size:
        labels[np.clip(monitor_index, 0, size - 1)] = "MONITOR"
    if virtual_index.size:
        labels[np.clip(virtual_index, 0, size - 1)] = "VIRTUAL"
    return labels


def _compute_twiss_phase(
    *,
    model_path: str,
    limit: int,
    unit: str,
    clean: bool,
    threshold_factor: float,
    plot: bool,
    amplitude_overlay: bool,
    phase_plot: bool,
    use_position: bool,
    prefix: str,
    data_prefix: str,
    tango: bool,
    device: str,
    dtype_name: str,
    update: bool,
    verbose: bool,
) -> dict[str, object]:
    time_label = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if limit <= 0:
        raise RuntimeError("Range limit must be positive.")
    if threshold_factor <= 0.0:
        raise RuntimeError("Threshold factor must be positive.")

    cs = factory(target=("tango" if tango else "epics"))
    scale = {"m": 1.0, "mm": 1.0e-2, "mk": 1.0e-6}[unit]
    source_prefix = prefix if not data_prefix else data_prefix

    # Keep monitor source consistent with script/twiss_phase.py.
    monitor_count = int(cs.get(f"{prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{prefix}:MONITOR:LIST")[:monitor_count]
    names = [_as_str(name) for name in monitor_names]
    flag = torch.tensor([cs.get(f"{prefix}:{name}:FLAG") for name in names], dtype=torch.int64, device=device)
    position = np.asarray([cs.get(f"{prefix}:{name}:TIME") for name in names], dtype=np.float64)
    selected = {name: int(mark) for name, mark in zip(names, flag.cpu().numpy()) if int(mark) == 1}

    value_nux = torch.tensor(cs.get(f"{source_prefix}:FREQUENCY:VALUE:X"), dtype=dtype, device=device)
    error_nux = torch.tensor(cs.get(f"{source_prefix}:FREQUENCY:ERROR:X"), dtype=dtype, device=device)
    value_nuy = torch.tensor(cs.get(f"{source_prefix}:FREQUENCY:VALUE:Y"), dtype=dtype, device=device)
    error_nuy = torch.tensor(cs.get(f"{source_prefix}:FREQUENCY:ERROR:Y"), dtype=dtype, device=device)

    pv_value_ax = [f"{source_prefix}:{name}:AMPLITUDE:VALUE:X" for name in names]
    pv_error_ax = [f"{source_prefix}:{name}:AMPLITUDE:ERROR:X" for name in names]
    value_ax = scale * torch.tensor([cs.get(pv) for pv in pv_value_ax], dtype=dtype, device=device)
    error_ax = scale * torch.tensor([cs.get(pv) for pv in pv_error_ax], dtype=dtype, device=device)

    pv_value_ay = [f"{source_prefix}:{name}:AMPLITUDE:VALUE:Y" for name in names]
    pv_error_ay = [f"{source_prefix}:{name}:AMPLITUDE:ERROR:Y" for name in names]
    value_ay = scale * torch.tensor([cs.get(pv) for pv in pv_value_ay], dtype=dtype, device=device)
    error_ay = scale * torch.tensor([cs.get(pv) for pv in pv_error_ay], dtype=dtype, device=device)

    pv_value_fx = [f"{source_prefix}:{name}:PHASE:VALUE:X" for name in names]
    pv_error_fx = [f"{source_prefix}:{name}:PHASE:ERROR:X" for name in names]
    value_fx = torch.tensor([cs.get(pv) for pv in pv_value_fx], dtype=dtype, device=device)
    error_fx = torch.tensor([cs.get(pv) for pv in pv_error_fx], dtype=dtype, device=device)

    pv_value_fy = [f"{source_prefix}:{name}:PHASE:VALUE:Y" for name in names]
    pv_error_fy = [f"{source_prefix}:{name}:PHASE:ERROR:Y" for name in names]
    value_fy = torch.tensor([cs.get(pv) for pv in pv_value_fy], dtype=dtype, device=device)
    error_fy = torch.tensor([cs.get(pv) for pv in pv_error_fy], dtype=dtype, device=device)

    resolved_model = _resolve_model_path(model_path)
    model = Model(path=resolved_model, dtype=dtype, device=device)
    model.flag[model.monitor_index] = flag

    table = Table(
        names,
        value_nux,
        value_nuy,
        value_ax,
        value_ay,
        value_fx,
        value_fy,
        error_nux,
        error_nuy,
        error_ax,
        error_ay,
        error_fx,
        error_fy,
        dtype=dtype,
        device=device,
    )

    twiss = Twiss(model, table, limit=limit)
    twiss.get_action(dict_threshold={"use": clean, "factor": threshold_factor})
    twiss.get_twiss_from_amplitude()

    twiss.phase_virtual()
    twiss.get_twiss_from_phase()
    mask_x = twiss.filter_twiss(plane="x")
    mask_y = twiss.filter_twiss(plane="y")
    if limit != 1:
        twiss.process_twiss(plane="x", mask=mask_x, weight=True)
        twiss.process_twiss(plane="y", mask=mask_y, weight=True)
    else:
        twiss.process_twiss(plane="x")
        twiss.process_twiss(plane="y")

    ax_m = twiss.model.ax[1:-1].cpu().numpy()
    bx_m = twiss.model.bx[1:-1].cpu().numpy()
    ay_m = twiss.model.ay[1:-1].cpu().numpy()
    by_m = twiss.model.by[1:-1].cpu().numpy()
    model_position = twiss.model.time[1:-1].cpu().numpy()
    model_names = [_as_str(name) for name in twiss.model.name[1:-1]]

    fx_m = twiss.model.monitor_phase_x.cpu().numpy()
    fy_m = twiss.model.monitor_phase_y.cpu().numpy()

    bx_m_a = twiss.model.bx[twiss.model.monitor_index].cpu().numpy()
    by_m_a = twiss.model.by[twiss.model.monitor_index].cpu().numpy()
    bx_a = twiss.data_amplitude["bx"].cpu().numpy()
    sigma_bx_a = twiss.data_amplitude["sigma_bx"].cpu().numpy()
    by_a = twiss.data_amplitude["by"].cpu().numpy()
    sigma_by_a = twiss.data_amplitude["sigma_by"].cpu().numpy()

    fx = twiss.table.phase_x.cpu().numpy()
    sigma_fx = twiss.table.sigma_x.cpu().numpy()
    fy = twiss.table.phase_y.cpu().numpy()
    sigma_fy = twiss.table.sigma_y.cpu().numpy()

    ax = twiss.ax[1:-1].cpu().numpy()
    sigma_ax = twiss.sigma_ax[1:-1].cpu().numpy()
    bx = twiss.bx[1:-1].cpu().numpy()
    sigma_bx = twiss.sigma_bx[1:-1].cpu().numpy()
    ay = twiss.ay[1:-1].cpu().numpy()
    sigma_ay = twiss.sigma_ay[1:-1].cpu().numpy()
    by = twiss.by[1:-1].cpu().numpy()
    sigma_by = twiss.sigma_by[1:-1].cpu().numpy()

    monitor_index = np.asarray([int(index) - 1 for index in model.monitor_index], dtype=np.int64)
    monitor_index = monitor_index[(monitor_index >= 0) & (monitor_index < len(model_names))]
    virtual_index = np.asarray(
        [int(index) - 1 for index in model.virtual_index if 0 < int(index) < model.size - 1], dtype=np.int64
    )
    virtual_index = virtual_index[(virtual_index >= 0) & (virtual_index < len(model_names))]

    with np.errstate(divide="ignore", invalid="ignore"):
        error_bx = (bx_m - bx) / bx_m
        delta_bx = sigma_bx / bx_m
        error_by = (by_m - by) / by_m
        delta_by = sigma_by / by_m
        error_bx_a = (bx_m_a - bx_a) / bx_m_a
        delta_bx_a = sigma_bx_a / bx_m_a
        error_by_a = (by_m_a - by_a) / by_m_a
        delta_by_a = sigma_by_a / by_m_a

    if monitor_index.size == 0:
        rms_bx = float(100.0 * np.sqrt(np.nanmean(np.square(error_bx))))
        rms_by = float(100.0 * np.sqrt(np.nanmean(np.square(error_by))))
    else:
        rms_bx = float(100.0 * np.sqrt(np.nanmean(np.square(error_bx[monitor_index]))))
        rms_by = float(100.0 * np.sqrt(np.nanmean(np.square(error_by[monitor_index]))))

    if update:
        for bpm, value, error in zip(names, bx_a, sigma_bx_a):
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BX:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BX:ERROR", float(error))
        for bpm, value, error in zip(names, by_a, sigma_by_a):
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BY:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BY:ERROR", float(error))
        for bpm, value, error in zip(model_names, bx, sigma_bx):
            cs.set(f"{prefix}:{bpm}:PHASE:BX:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:PHASE:BX:ERROR", float(error))
        for bpm, value, error in zip(model_names, by, sigma_by):
            cs.set(f"{prefix}:{bpm}:PHASE:BY:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:PHASE:BY:ERROR", float(error))
        for bpm, value, error in zip(model_names, ax, sigma_ax):
            cs.set(f"{prefix}:{bpm}:PHASE:AX:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:PHASE:AX:ERROR", float(error))
        for bpm, value, error in zip(model_names, ay, sigma_ay):
            cs.set(f"{prefix}:{bpm}:PHASE:AY:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:PHASE:AY:ERROR", float(error))

    type_labels = _make_type_labels(len(model_names), monitor_index, virtual_index)

    phase_df = pd.DataFrame(
        {
            "BPM": names,
            "POSITION": position,
            "FX": fx,
            "SIGMA_FX": sigma_fx,
            "FX_M": fx_m,
            "ERROR_FX": fx_m - fx,
            "FY": fy,
            "SIGMA_FY": sigma_fy,
            "FY_M": fy_m,
            "ERROR_FY": fy_m - fy,
        }
    )

    twiss_phase_df = pd.DataFrame(
        {
            "BPM": model_names,
            "POSITION": model_position,
            "TYPE": type_labels,
            "AX": ax,
            "SIGMA_AX": sigma_ax,
            "AX_M": ax_m,
            "ERROR_AX": ax_m - ax,
            "AY": ay,
            "SIGMA_AY": sigma_ay,
            "AY_M": ay_m,
            "ERROR_AY": ay_m - ay,
            "BX": bx,
            "SIGMA_BX": sigma_bx,
            "BX_M": bx_m,
            "ERROR_BX": error_bx,
            "DELTA_BX": delta_bx,
            "BY": by,
            "SIGMA_BY": sigma_by,
            "BY_M": by_m,
            "ERROR_BY": error_by,
            "DELTA_BY": delta_by,
        }
    )

    amplitude_df = pd.DataFrame(
        {
            "BPM": names,
            "POSITION": position,
            "BX_A": bx_a,
            "SIGMA_BX_A": sigma_bx_a,
            "ERROR_BX_A": error_bx_a,
            "DELTA_BX_A": delta_bx_a,
            "BY_A": by_a,
            "SIGMA_BY_A": sigma_by_a,
            "ERROR_BY_A": error_by_a,
            "DELTA_BY_A": delta_by_a,
        }
    )

    selected_df = pd.DataFrame(
        {
            "BPM": names,
            "FLAG": [int(mark) for mark in flag.cpu().numpy()],
            "SELECTED": [name in selected for name in names],
        }
    )

    return {
        "time": time_label,
        "size": len(names),
        "selected_count": len(selected),
        "limit": limit,
        "rms_bx": rms_bx,
        "rms_by": rms_by,
        "phase_sum_x": float(np.sum(np.abs(fx_m - fx))),
        "phase_sum_y": float(np.sum(np.abs(fy_m - fy))),
        "alpha_sum_x": float(np.sum(np.abs(ax_m - ax))),
        "alpha_sum_y": float(np.sum(np.abs(ay_m - ay))),
        "phase_df": phase_df,
        "twiss_phase_df": twiss_phase_df,
        "amplitude_df": amplitude_df,
        "selected_df": selected_df,
        "plot": plot,
        "phase_plot": phase_plot,
        "amplitude_overlay": amplitude_overlay,
        "use_position": use_position,
        "verbose": verbose,
        "pv_list": [
            *pv_value_ax,
            *pv_error_ax,
            *pv_value_ay,
            *pv_error_ay,
            *pv_value_fx,
            *pv_error_fx,
            *pv_value_fy,
            *pv_error_fy,
        ],
    }


def _add_model_line(fig: go.Figure, df: pd.DataFrame, x_axis: str, y_model: str, name: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=df[x_axis],
            y=df[y_model],
            mode="lines+markers",
            name=name,
            marker={"size": 8, "symbol": "square-open"},
            line={"color": "black"},
        )
    )


def _add_virtual_points(fig: go.Figure, df: pd.DataFrame, x_axis: str, y: str, y_error: str, name: str) -> None:
    virtual = df[df["TYPE"] == "VIRTUAL"]
    if virtual.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=virtual[x_axis],
            y=virtual[y],
            error_y={"type": "data", "array": virtual[y_error].to_numpy()},
            mode="markers",
            name=name,
            marker={"size": 10, "color": "green"},
        )
    )


def _add_amplitude_overlay(fig: go.Figure, df: pd.DataFrame, x_axis: str, y: str, y_error: str, name: str) -> None:
    if df.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=df[x_axis],
            y=df[y],
            error_y={"type": "data", "array": df[y_error].to_numpy()},
            mode="markers",
            name=name,
            marker={"size": 9, "color": "red"},
        )
    )


def main() -> None:
    st.set_page_config(page_title="Twiss (phase)", layout="wide")
    st.title("Twiss (phase)")

    try:
        app_config: AppConfig = load_app_config("twiss_phase")
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

    with st.sidebar.form("twiss_phase_form", enter_to_submit=False):
        model_path = ui.text_input("Model path", value="elettra.yaml")
        limit = ui.number_input("Range limit", min_value=1, value=4, step=1)
        unit = ui.selectbox("Amplitude unit", options=["m", "mm", "mk"], index=0)

        with st.expander("Filtering", expanded=False):
            clean = ui.checkbox("Clean action outliers", value=False)
            threshold_factor = ui.number_input(
                "Threshold factor",
                min_value=0.000001,
                value=5.0,
                step=0.1,
                format="%0.6f",
            )

        with st.expander("Display", expanded=False):
            plot = ui.checkbox("Show plots", value=True)
            phase_plot = ui.checkbox("Show phase-advance plots", value=False)
            amplitude_overlay = ui.checkbox("Overlay amplitude twiss", value=False)
            use_position = ui.checkbox("Use BPM position on x-axis", value=False)

        data_prefix = ui.text_input("PV data prefix override", value="")
        update = ui.checkbox("Update PVs", value=False)
        verbose = ui.checkbox("Verbose output", value=False)

        run = st.form_submit_button("Run", type="primary")

    right_col, _ = st.columns([5, 1])
    with right_col:
        if not run:
            st.info("Set options on the left and click Run.")
            return

        with st.spinner("Running twiss-phase computation..."):
            try:
                result = _compute_twiss_phase(
                    model_path=model_path,
                    limit=int(limit),
                    unit=unit,
                    clean=clean,
                    threshold_factor=float(threshold_factor),
                    plot=plot,
                    amplitude_overlay=amplitude_overlay,
                    phase_plot=phase_plot,
                    use_position=use_position,
                    prefix=prefix_name,
                    data_prefix=data_prefix,
                    tango=tango,
                    device=device_name,
                    dtype_name=dtype_name,
                    update=update,
                    verbose=verbose,
                )
            except Exception as exception:  # pragma: no cover - depends on external CS
                st.error(str(exception))
                return

        st.success("Computation finished.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BPMs", value=int(result["size"]))
        c2.metric("Selected", value=int(result["selected_count"]))
        c3.metric("RMS BX [%]", value=f"{result['rms_bx']:.6f}")
        c4.metric("RMS BY [%]", value=f"{result['rms_by']:.6f}")

        st.write(
            f"Phase diff sums: X={result['phase_sum_x']:.9f}, Y={result['phase_sum_y']:.9f} | "
            f"Alpha diff sums: X={result['alpha_sum_x']:.9f}, Y={result['alpha_sum_y']:.9f}"
        )

        x_axis = "POSITION" if result["use_position"] else "BPM"
        phase_df = result["phase_df"]
        twiss_df = result["twiss_phase_df"]
        amp_df = result["amplitude_df"]

        if result["plot"]:
            if result["phase_plot"]:
                st.subheader("Phase Advance")
                fig = px.scatter(
                    phase_df,
                    x=x_axis,
                    y="FX",
                    error_y="SIGMA_FX",
                    color_discrete_sequence=["blue"],
                    title=f"{result['time']}: FX",
                )
                fig.update_traces(marker={"size": 9})
                _add_model_line(fig, phase_df, x_axis, "FX_M", "FX_M")
                st.plotly_chart(fig, use_container_width=True)

                fig = px.scatter(
                    phase_df,
                    x=x_axis,
                    y="FY",
                    error_y="SIGMA_FY",
                    color_discrete_sequence=["blue"],
                    title=f"{result['time']}: FY",
                )
                fig.update_traces(marker={"size": 9})
                _add_model_line(fig, phase_df, x_axis, "FY_M", "FY_M")
                st.plotly_chart(fig, use_container_width=True)

                fig = px.scatter(
                    phase_df,
                    x=x_axis,
                    y="ERROR_FX",
                    error_y="SIGMA_FX",
                    color_discrete_sequence=["blue"],
                    title=f"{result['time']}: FX_M-FX",
                )
                fig.update_traces(marker={"size": 9})
                st.plotly_chart(fig, use_container_width=True)

                fig = px.scatter(
                    phase_df,
                    x=x_axis,
                    y="ERROR_FY",
                    error_y="SIGMA_FY",
                    color_discrete_sequence=["blue"],
                    title=f"{result['time']}: FY_M-FY",
                )
                fig.update_traces(marker={"size": 9})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Twiss Beta (Phase)")
            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="BX",
                error_y="SIGMA_BX",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: BX | RMS={result['rms_bx']:.6f}%",
            )
            fig.update_traces(marker={"size": 9})
            _add_model_line(fig, twiss_df, x_axis, "BX_M", "BX_M")
            _add_virtual_points(fig, twiss_df, x_axis, "BX", "SIGMA_BX", "BX (VIRTUAL)")
            if result["amplitude_overlay"]:
                _add_amplitude_overlay(fig, amp_df, x_axis, "BX_A", "SIGMA_BX_A", "BX_A")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="BY",
                error_y="SIGMA_BY",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: BY | RMS={result['rms_by']:.6f}%",
            )
            fig.update_traces(marker={"size": 9})
            _add_model_line(fig, twiss_df, x_axis, "BY_M", "BY_M")
            _add_virtual_points(fig, twiss_df, x_axis, "BY", "SIGMA_BY", "BY (VIRTUAL)")
            if result["amplitude_overlay"]:
                _add_amplitude_overlay(fig, amp_df, x_axis, "BY_A", "SIGMA_BY_A", "BY_A")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="ERROR_BX",
                error_y="DELTA_BX",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: (BX_M-BX)/BX_M",
            )
            fig.update_traces(marker={"size": 9})
            _add_virtual_points(fig, twiss_df, x_axis, "ERROR_BX", "DELTA_BX", "(BX_M-BX)/BX_M (VIRTUAL)")
            if result["amplitude_overlay"]:
                _add_amplitude_overlay(fig, amp_df, x_axis, "ERROR_BX_A", "DELTA_BX_A", "(BX_M-BX_A)/BX_M")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="ERROR_BY",
                error_y="DELTA_BY",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: (BY_M-BY)/BY_M",
            )
            fig.update_traces(marker={"size": 9})
            _add_virtual_points(fig, twiss_df, x_axis, "ERROR_BY", "DELTA_BY", "(BY_M-BY)/BY_M (VIRTUAL)")
            if result["amplitude_overlay"]:
                _add_amplitude_overlay(fig, amp_df, x_axis, "ERROR_BY_A", "DELTA_BY_A", "(BY_M-BY_A)/BY_M")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Twiss Alpha (Phase)")
            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="AX",
                error_y="SIGMA_AX",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: AX",
            )
            fig.update_traces(marker={"size": 9})
            _add_model_line(fig, twiss_df, x_axis, "AX_M", "AX_M")
            _add_virtual_points(fig, twiss_df, x_axis, "AX", "SIGMA_AX", "AX (VIRTUAL)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="AY",
                error_y="SIGMA_AY",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: AY",
            )
            fig.update_traces(marker={"size": 9})
            _add_model_line(fig, twiss_df, x_axis, "AY_M", "AY_M")
            _add_virtual_points(fig, twiss_df, x_axis, "AY", "SIGMA_AY", "AY (VIRTUAL)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="ERROR_AX",
                error_y="SIGMA_AX",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: AX_M-AX",
            )
            fig.update_traces(marker={"size": 9})
            _add_virtual_points(fig, twiss_df, x_axis, "ERROR_AX", "SIGMA_AX", "AX_M-AX (VIRTUAL)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                twiss_df,
                x=x_axis,
                y="ERROR_AY",
                error_y="SIGMA_AY",
                color_discrete_sequence=["blue"],
                title=f"{result['time']}: AY_M-AY",
            )
            fig.update_traces(marker={"size": 9})
            _add_virtual_points(fig, twiss_df, x_axis, "ERROR_AY", "SIGMA_AY", "AY_M-AY (VIRTUAL)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Tables")
            st.dataframe(phase_df, use_container_width=True)
            st.dataframe(twiss_df, use_container_width=True)
            if result["amplitude_overlay"]:
                st.dataframe(amp_df, use_container_width=True)

        st.subheader("Monitor Flags")
        st.dataframe(result["selected_df"], use_container_width=True)

        if result["verbose"]:
            with st.expander("Verbose details"):
                st.write(f"Time: {result['time']}")
                st.write("PV list:")
                for pv in result["pv_list"]:
                    st.write(f"- {pv}")


if __name__ == "__main__":
    main()
