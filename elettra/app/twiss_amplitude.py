#!/usr/bin/env python3

"""script/twiss_amplitude.py."""

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


def _compute_twiss_amplitude(
    *,
    model_path: str,
    unit: str,
    clean: bool,
    threshold_factor: float,
    plot: bool,
    action_plot: bool,
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

    if threshold_factor <= 0.0:
        raise RuntimeError("Threshold factor must be positive.")

    cs = factory(target=("tango" if tango else "epics"))
    scale = {"m": 1.0, "mm": 1.0e-2, "mk": 1.0e-6}[unit]
    source_prefix = prefix if not data_prefix else data_prefix

    monitor_count = int(cs.get(f"{source_prefix}:MONITOR:COUNT"))
    monitor_names = cs.get(f"{source_prefix}:MONITOR:LIST")[:monitor_count]
    names = [_as_str(name) for name in monitor_names]
    flag = torch.tensor([cs.get(f"{source_prefix}:{name}:FLAG") for name in names], dtype=torch.int64, device=device)
    position = np.asarray([cs.get(f"{source_prefix}:{name}:TIME") for name in names], dtype=np.float64)
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
    twiss.get_twiss_from_amplitude()

    value_jx = twiss.action["jx"].cpu().numpy()
    error_jx = twiss.action["sigma_jx"].cpu().numpy()
    value_jy = twiss.action["jy"].cpu().numpy()
    error_jy = twiss.action["sigma_jy"].cpu().numpy()

    center_jx = float(twiss.action["center_jx"].cpu().numpy())
    spread_jx = float(twiss.action["spread_jx"].cpu().numpy())
    center_jy = float(twiss.action["center_jy"].cpu().numpy())
    spread_jy = float(twiss.action["spread_jy"].cpu().numpy())

    mask_x, mask_y = twiss.action["mask"]
    outlier_x = mask_x.logical_not().cpu().numpy().astype(bool)
    outlier_y = mask_y.logical_not().cpu().numpy().astype(bool)

    bx_m = twiss.model.bx[twiss.model.monitor_index].cpu().numpy()
    by_m = twiss.model.by[twiss.model.monitor_index].cpu().numpy()
    bx = twiss.data_amplitude["bx"].cpu().numpy()
    sigma_bx = twiss.data_amplitude["sigma_bx"].cpu().numpy()
    by = twiss.data_amplitude["by"].cpu().numpy()
    sigma_by = twiss.data_amplitude["sigma_by"].cpu().numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        error_bx = (bx_m - bx) / bx_m
        delta_bx = sigma_bx / bx_m
        error_by = (by_m - by) / by_m
        delta_by = sigma_by / by_m

    rms_bx = float(100.0 * np.sqrt(np.nanmean(np.square(error_bx))))
    rms_by = float(100.0 * np.sqrt(np.nanmean(np.square(error_by))))

    if update:
        cs.set(f"{prefix}:ACTION:LIST:VALUE:X", value_jx)
        cs.set(f"{prefix}:ACTION:LIST:ERROR:X", error_jx)
        cs.set(f"{prefix}:ACTION:VALUE:X", center_jx)
        cs.set(f"{prefix}:ACTION:ERROR:X", spread_jx)
        cs.set(f"{prefix}:ACTION:LIST:VALUE:Y", value_jy)
        cs.set(f"{prefix}:ACTION:LIST:ERROR:Y", error_jy)
        cs.set(f"{prefix}:ACTION:VALUE:Y", center_jy)
        cs.set(f"{prefix}:ACTION:ERROR:Y", spread_jy)
        for bpm, value, error in zip(names, bx, sigma_bx):
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BX:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BX:ERROR", float(error))
        for bpm, value, error in zip(names, by, sigma_by):
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BY:VALUE", float(value))
            cs.set(f"{prefix}:{bpm}:AMPLITUDE:BY:ERROR", float(error))

    action_df = pd.DataFrame(
        {
            "BPM": names,
            "POSITION": position,
            "JX": value_jx,
            "SIGMA_JX": error_jx,
            "JY": value_jy,
            "SIGMA_JY": error_jy,
            "OUTLIER_X": outlier_x,
            "OUTLIER_Y": outlier_y,
        }
    )

    twiss_df = pd.DataFrame(
        {
            "BPM": names,
            "POSITION": position,
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
        "rms_bx": rms_bx,
        "rms_by": rms_by,
        "action_df": action_df,
        "twiss_df": twiss_df,
        "selected_df": selected_df,
        "plot": plot,
        "action_plot": action_plot,
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


def _plot_scatter_with_model(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_value: str,
    y_error: str,
    y_model: str,
    outlier_col: str,
    title: str,
    y_label: str,
) -> go.Figure:
    fig = px.scatter(df, x=x_axis, y=y_value, error_y=y_error, color_discrete_sequence=["blue"], opacity=0.8)
    fig.update_traces(name=y_value, showlegend=True, marker={"size": 9})
    fig.add_trace(
        go.Scatter(
            x=df[x_axis],
            y=df[y_model],
            mode="lines+markers",
            name=y_model,
            marker={"size": 8, "symbol": "square-open"},
            line={"color": "black"},
        )
    )
    masked = df[df[outlier_col]]
    if not masked.empty:
        fig.add_trace(
            go.Scatter(
                x=masked[x_axis],
                y=masked[y_value],
                error_y={"type": "data", "array": masked[y_error].to_numpy()},
                mode="markers",
                name=f"{y_value} (OUTLIER)",
                marker={"size": 10, "color": "red"},
            )
        )
    fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=y_label)
    return fig


def _plot_error(
    df: pd.DataFrame,
    *,
    x_axis: str,
    y_value: str,
    y_error: str,
    outlier_col: str,
    title: str,
    y_label: str,
) -> go.Figure:
    fig = px.scatter(df, x=x_axis, y=y_value, error_y=y_error, color_discrete_sequence=["blue"], opacity=0.8)
    fig.update_traces(name=y_label, showlegend=True, marker={"size": 9})
    masked = df[df[outlier_col]]
    if not masked.empty:
        fig.add_trace(
            go.Scatter(
                x=masked[x_axis],
                y=masked[y_value],
                error_y={"type": "data", "array": masked[y_error].to_numpy()},
                mode="markers",
                name=f"{y_label} (OUTLIER)",
                marker={"size": 10, "color": "red"},
            )
        )
    fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=y_label)
    return fig


def main() -> None:
    st.set_page_config(page_title="Twiss (amplitude)", layout="wide")
    st.title("Twiss (amplitude)")

    try:
        app_config: AppConfig = load_app_config("twiss_amplitude")
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

    with st.sidebar.form("twiss_amplitude_form", enter_to_submit=False):
        model_path = ui.text_input("Model path", value="lattice.yaml")
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
            action_plot = ui.checkbox("Show action plots", value=False)
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

        with st.spinner("Running twiss-amplitude computation..."):
            try:
                result = _compute_twiss_amplitude(
                    model_path=model_path,
                    unit=unit,
                    clean=clean,
                    threshold_factor=float(threshold_factor),
                    plot=plot,
                    action_plot=action_plot,
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
            f"JX center={result['center_jx']:.9f}, spread={result['spread_jx']:.9f} | "
            f"JY center={result['center_jy']:.9f}, spread={result['spread_jy']:.9f}"
        )

        x_axis = "POSITION" if result["use_position"] else "BPM"
        action_df = result["action_df"]
        twiss_df = result["twiss_df"]

        if result["plot"]:
            if result["action_plot"]:
                st.subheader("Action")
                fig = px.scatter(
                    action_df,
                    x=x_axis,
                    y="JX",
                    error_y="SIGMA_JX",
                    color_discrete_sequence=["blue"],
                    title=f"{result['time']}: ACTION X",
                )
                fig.update_traces(marker={"size": 9})
                outliers = action_df[action_df["OUTLIER_X"]]
                if not outliers.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=outliers[x_axis],
                            y=outliers["JX"],
                            error_y={"type": "data", "array": outliers["SIGMA_JX"].to_numpy()},
                            mode="markers",
                            marker={"size": 10, "color": "red"},
                            name="JX (OUTLIER)",
                        )
                    )
                fig.add_hline(y=result["center_jx"] - result["spread_jx"], line_dash="dash", line_color="black")
                fig.add_hline(y=result["center_jx"], line_dash="dash", line_color="black")
                fig.add_hline(y=result["center_jx"] + result["spread_jx"], line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)

                fig = px.scatter(
                    action_df,
                    x=x_axis,
                    y="JY",
                    error_y="SIGMA_JY",
                    color_discrete_sequence=["blue"],
                    title=f"{result['time']}: ACTION Y",
                )
                fig.update_traces(marker={"size": 9})
                outliers = action_df[action_df["OUTLIER_Y"]]
                if not outliers.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=outliers[x_axis],
                            y=outliers["JY"],
                            error_y={"type": "data", "array": outliers["SIGMA_JY"].to_numpy()},
                            mode="markers",
                            marker={"size": 10, "color": "red"},
                            name="JY (OUTLIER)",
                        )
                    )
                fig.add_hline(y=result["center_jy"] - result["spread_jy"], line_dash="dash", line_color="black")
                fig.add_hline(y=result["center_jy"], line_dash="dash", line_color="black")
                fig.add_hline(y=result["center_jy"] + result["spread_jy"], line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Twiss")
            fig = _plot_scatter_with_model(
                twiss_df,
                x_axis=x_axis,
                y_value="BX",
                y_error="SIGMA_BX",
                y_model="BX_M",
                outlier_col="OUTLIER_X",
                title=f"{result['time']}: BX",
                y_label="BX",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = _plot_scatter_with_model(
                twiss_df,
                x_axis=x_axis,
                y_value="BY",
                y_error="SIGMA_BY",
                y_model="BY_M",
                outlier_col="OUTLIER_Y",
                title=f"{result['time']}: BY",
                y_label="BY",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = _plot_error(
                twiss_df,
                x_axis=x_axis,
                y_value="ERROR_BX",
                y_error="DELTA_BX",
                outlier_col="OUTLIER_X",
                title=f"{result['time']}: (BX_M-BX)/BX_M",
                y_label="(BX_M-BX)/BX_M",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig = _plot_error(
                twiss_df,
                x_axis=x_axis,
                y_value="ERROR_BY",
                y_error="DELTA_BY",
                outlier_col="OUTLIER_Y",
                title=f"{result['time']}: (BY_M-BY)/BY_M",
                y_label="(BY_M-BY)/BY_M",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Tables")
            st.dataframe(action_df, use_container_width=True)
            st.dataframe(twiss_df, use_container_width=True)

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
