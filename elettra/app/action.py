#!/usr/bin/env python3

"""script/action.py"""

from __future__ import annotations

from configuration import AppConfig, WidgetDefaults, load_app_config

import os
from datetime import datetime

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


def _compute_action(
    *,
    model_path: str,
    unit: str,
    clean: bool,
    threshold_factor: float,
    plot: bool,
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
    if threshold_factor <= 0.0:
        raise RuntimeError("Threshold factor must be positive.")

    cs = factory(target=("tango" if tango else "epics"))
    scale = {"m": 1.0, "mm": 1.0e-2, "mk": 1.0e-6}[unit]
    source_prefix = prefix if not data_prefix else data_prefix

    monitor_count = int(cs.get(f"{prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{prefix}:MONITOR:LIST")[:monitor_count]
    names = [_as_str(name) for name in monitor_names]
    flag = torch.tensor([cs.get(f"{prefix}:{name}:FLAG") for name in names], dtype=torch.int64, device=device)
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

    twiss = Twiss(model, table)
    twiss.get_action(dict_threshold={"use": clean, "factor": threshold_factor})

    jx = twiss.action["jx"].cpu().numpy()
    sx = twiss.action["sigma_jx"].cpu().numpy()
    jy = twiss.action["jy"].cpu().numpy()
    sy = twiss.action["sigma_jy"].cpu().numpy()

    center_jx = float(twiss.action["center_jx"].cpu().numpy())
    spread_jx = float(twiss.action["spread_jx"].cpu().numpy())
    center_jy = float(twiss.action["center_jy"].cpu().numpy())
    spread_jy = float(twiss.action["spread_jy"].cpu().numpy())

    mask_x, mask_y = twiss.action["mask"]
    outlier_x = mask_x.logical_not().cpu().numpy().astype(bool)
    outlier_y = mask_y.logical_not().cpu().numpy().astype(bool)

    if update:
        cs.set(f"{prefix}:ACTION:LIST:VALUE:X", jx)
        cs.set(f"{prefix}:ACTION:LIST:ERROR:X", sx)
        cs.set(f"{prefix}:ACTION:VALUE:X", center_jx)
        cs.set(f"{prefix}:ACTION:ERROR:X", spread_jx)
        cs.set(f"{prefix}:ACTION:LIST:VALUE:Y", jy)
        cs.set(f"{prefix}:ACTION:LIST:ERROR:Y", sy)
        cs.set(f"{prefix}:ACTION:VALUE:Y", center_jy)
        cs.set(f"{prefix}:ACTION:ERROR:Y", spread_jy)

    summary_df = pd.DataFrame(
        {
            "BPM": names,
            "JX": jx,
            "SIGMA_JX": sx,
            "JY": jy,
            "SIGMA_JY": sy,
            "OUTLIER_X": outlier_x,
            "OUTLIER_Y": outlier_y,
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
        "center_jx": center_jx,
        "spread_jx": spread_jx,
        "center_jy": center_jy,
        "spread_jy": spread_jy,
        "summary_df": summary_df,
        "selected_df": selected_df,
        "plot": plot,
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


def _make_action_figure(
    df: pd.DataFrame,
    *,
    time_label: str,
    y_name: str,
    y_error_name: str,
    outlier_name: str,
    center: float,
    spread: float,
) -> go.Figure:
    fig = px.scatter(
        df,
        x="BPM",
        y=y_name,
        error_y=y_error_name,
        color_discrete_sequence=["blue"],
        title=f"{time_label}: {y_name}",
        opacity=0.8,
    )
    fig.update_traces(mode="lines+markers", marker={"size": 9})

    outliers = df[df[outlier_name]]
    if not outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=outliers["BPM"],
                y=outliers[y_name],
                error_y={"type": "data", "array": outliers[y_error_name].to_numpy()},
                mode="markers",
                marker={"size": 10, "color": "red"},
                name=f"{y_name} (OUTLIER)",
            )
        )

    fig.add_hline(y=center - spread, line_color="black", line_dash="dash", line_width=1.0)
    fig.add_hline(y=center, line_color="black", line_dash="dash", line_width=1.0)
    fig.add_hline(y=center + spread, line_color="black", line_dash="dash", line_width=1.0)
    fig.update_layout(xaxis_title="BPM", yaxis_title=y_name)
    return fig


def main() -> None:
    st.set_page_config(page_title="Action", layout="wide")
    st.title("Action")

    try:
        app_config: AppConfig = load_app_config("action")
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

    with st.sidebar.form("action_form", enter_to_submit=False):
        model_path = ui.text_input("Model path", value="elettra.yaml")
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

        plot = ui.checkbox("Show action plots", value=True)
        data_prefix = ui.text_input("PV data prefix override", value="")
        update = ui.checkbox("Update PVs", value=False)
        verbose = ui.checkbox("Verbose output", value=False)

        run = st.form_submit_button("Run", type="primary")

    right_col, _ = st.columns([5, 1])
    with right_col:
        if not run:
            st.info("Set options on the left and click Run.")
            return

        with st.spinner("Running action computation..."):
            try:
                result = _compute_action(
                    model_path=model_path,
                    unit=unit,
                    clean=clean,
                    threshold_factor=float(threshold_factor),
                    plot=plot,
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
        c1, c2 = st.columns(2)
        c1.metric("BPMs", value=int(result["size"]))
        c2.metric("Selected", value=int(result["selected_count"]))

        st.write(
            f"JX center={result['center_jx']:.9f}, spread={result['spread_jx']:.9f} | "
            f"JY center={result['center_jy']:.9f}, spread={result['spread_jy']:.9f}"
        )

        if result["plot"]:
            st.subheader("Action")
            fig = _make_action_figure(
                result["summary_df"],
                time_label=result["time"],
                y_name="JX",
                y_error_name="SIGMA_JX",
                outlier_name="OUTLIER_X",
                center=float(result["center_jx"]),
                spread=float(result["spread_jx"]),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = _make_action_figure(
                result["summary_df"],
                time_label=result["time"],
                y_name="JY",
                y_error_name="SIGMA_JY",
                outlier_name="OUTLIER_Y",
                center=float(result["center_jy"]),
                spread=float(result["spread_jy"]),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Tables")
            st.dataframe(result["summary_df"], use_container_width=True)

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
