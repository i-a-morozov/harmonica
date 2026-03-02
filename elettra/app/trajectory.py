#!/usr/bin/env python3

"""script/trajectory.py"""

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


def _compute_trajectory(
    *,
    plane: str,
    trajectory_length: int,
    load_length: int,
    offset: int,
    use_rise: bool,
    save: bool,
    save_matrix: bool,
    compare: bool,
    transform: str,
    filter_type: str,
    rank: int,
    svd_type: str,
    buffer: int,
    count: int,
    plot: bool,
    difference: bool,
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

    cs, bpm = _load_active_bpm(prefix, tango)

    try:
        bpm = bpm_select(bpm, skip=skip, only=only)
    except ValueError as exception:
        raise RuntimeError(str(exception)) from exception

    if not bpm:
        raise RuntimeError("BPM list is empty after filtering.")

    if load_length <= 0 or load_length > LIMIT:
        raise RuntimeError(f"Invalid load={load_length}, expected 1..{LIMIT}.")
    if trajectory_length <= 0 or trajectory_length > load_length:
        raise RuntimeError(f"Invalid length={trajectory_length}, expected 1..{load_length}.")
    if offset < 0:
        raise RuntimeError(f"Invalid offset={offset}, expected a non-negative value.")
    if load_length + offset > LIMIT:
        raise RuntimeError(f"Invalid load+offset={load_length + offset}, expected <= {LIMIT}.")

    position = np.asarray([cs.get(f"{prefix}:{name}:TIME") for name in bpm])
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

    size = len(bpm)
    total = load_length + offset + shift
    win = Window(load_length, dtype=dtype, device=device)
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

    if filter_type == "svd":
        flt = Filter(tbt)
        flt.filter_svd(rank=rank)
    elif filter_type == "hankel":
        flt = Filter(tbt)
        flt.filter_svd(rank=rank)
        flt.filter_hankel(rank=rank, random=(svd_type == "randomized"), buffer=buffer, count=count)

    mixed = tbt.make_signal(trajectory_length, tbt.work)
    trajectory = mixed.cpu().numpy().reshape(trajectory_length, size)
    bpm_name = list(bpm.keys())
    axis = plane.upper()

    turns = np.array([np.zeros(size, dtype=np.int32) + i for i in range(trajectory_length)]).flatten()
    time = (1.0 / circumference) * np.array([position + circumference * i for i in range(trajectory_length)]).flatten()

    trajectory_df = pd.DataFrame(
        {
            "BPM": bpm_name * trajectory_length,
            "TURN": turns.astype(str),
            "TIME": time,
            axis: trajectory.flatten(),
        }
    )

    pair_diff_df = None
    if difference:
        pair_size = size // 2
        if pair_size == 0:
            raise RuntimeError("At least two BPMs are required for pair differences.")
        trajectory_diff = trajectory[:, 0 : 2 * pair_size : 2] - trajectory[:, 1 : 2 * pair_size : 2]
        pair_name = [f"{n1}-{n2}" for n1, n2 in zip(bpm_name[0 : 2 * pair_size : 2], bpm_name[1 : 2 * pair_size : 2])]
        pair_position = 0.5 * (position[0 : 2 * pair_size : 2] + position[1 : 2 * pair_size : 2])
        pair_turn = np.array([np.zeros(pair_size, dtype=np.int32) + i for i in range(trajectory_length)]).flatten()
        pair_time = (1.0 / circumference) * np.array(
            [pair_position + circumference * i for i in range(trajectory_length)]
        ).flatten()
        pair_diff_df = pd.DataFrame(
            {
                "BPM": pair_name * trajectory_length,
                "TURN": pair_turn.astype(str),
                "TIME": pair_time,
                "DIFF": trajectory_diff.flatten(),
            }
        )

    compare_df = None
    if plot and compare:
        try:
            reference = np.load("trajectory.npy")
        except FileNotFoundError as exception:
            raise RuntimeError("trajectory.npy is not found for compare mode.") from exception
        except Exception as exception:
            raise RuntimeError(f"Failed to load trajectory.npy: {exception}") from exception

        reference = np.asarray(reference)
        try:
            difference_data = trajectory[:trajectory_length] - reference[:trajectory_length]
        except Exception as exception:
            raise RuntimeError(
                f"Cannot compare current trajectory shape {trajectory.shape} with reference shape {reference.shape}."
            ) from exception

        compare_df = pd.DataFrame(
            {
                "BPM": bpm_name * trajectory_length,
                "TURN": turns.astype(str),
                "TIME": time,
                "DIFF": difference_data.flatten(),
            }
        )

    saved_file = None
    if save:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
        saved_file = Path(f"tbt_trajectory_plane_{plane}_length_{trajectory_length}_time_({timestamp}).npy")
        np.save(saved_file, np.array([time, trajectory.flatten()]))

    trajectory_file = None
    if save_matrix:
        trajectory_file = Path("trajectory.npy")
        np.save(trajectory_file, trajectory)

    return {
        "time": time_label,
        "size": size,
        "axis": axis,
        "bpm": bpm,
        "pv_list": pv_list,
        "trajectory_df": trajectory_df,
        "pair_diff_df": pair_diff_df,
        "compare_df": compare_df,
        "saved_file": saved_file,
        "trajectory_file": trajectory_file,
        "plot": plot,
        "difference": difference,
        "compare": compare,
        "verbose": verbose,
    }


def main() -> None:
    st.set_page_config(page_title="Trajectory", layout="wide")
    st.title("Trajectory")

    try:
        app_config: AppConfig = load_app_config("trajectory")
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

    with st.sidebar.form("trajectory_form", enter_to_submit=False):
        st.caption("Planes")
        plane_col_x, plane_col_y, plane_col_i = st.columns(3)
        plane_x = plane_col_x.checkbox("X", value=bool(app_config.script.x.get("enabled", True)))
        plane_y = plane_col_y.checkbox("Y", value=bool(app_config.script.y.get("enabled", False)))
        i_cfg = app_config.script.values.get("i")
        i_default = bool(i_cfg.get("enabled", False)) if isinstance(i_cfg, dict) else False
        plane_i = plane_col_i.checkbox("I", value=bool(app_config.script.values.get("i_enabled", i_default)))
        trajectory_length = ui.number_input("Length", min_value=1, max_value=LIMIT, value=4, step=1)
        load_length = ui.number_input("Load", min_value=1, max_value=LIMIT, value=128, step=1)
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

        plot = ui.checkbox("Show trajectory plot", value=True)
        difference = ui.checkbox("Show pairwise BPM difference", value=False)
        compare = ui.checkbox("Compare with trajectory.npy", value=False)
        save = ui.checkbox("Save trajectory output", value=False)
        save_matrix = ui.checkbox("Save trajectory matrix as trajectory.npy", value=False)
        data_prefix = ui.text_input("PV data prefix override", value="")
        circumference = ui.number_input("Circumference", min_value=0.000001, value=259.2, step=0.1, format="%0.6f")
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
        with st.spinner("Running trajectory computation..."):
            try:
                for plane in planes:
                    results[plane] = _compute_trajectory(
                        plane=plane,
                        trajectory_length=int(trajectory_length),
                        load_length=int(load_length),
                        offset=int(offset),
                        use_rise=use_rise,
                        save=save,
                        save_matrix=save_matrix,
                        compare=compare,
                        transform=transform,
                        filter_type=filter_type,
                        rank=int(rank),
                        svd_type=svd_type,
                        buffer=int(buffer),
                        count=int(count),
                        plot=plot,
                        difference=difference,
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
        c2.metric("Length", value=int(trajectory_length))
        c3.metric("Planes", value=", ".join(plane.upper() for plane in planes))

        for plane in planes:
            result = results[plane]
            if result["saved_file"] is not None:
                st.write(f"{plane.upper()} saved: `{result['saved_file']}`")
            if result["trajectory_file"] is not None:
                st.write(f"{plane.upper()} matrix: `{result['trajectory_file']}`")

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
            if compare:
                st.subheader("Compare")
                for plane in planes:
                    result = results[plane]
                    if result["compare_df"] is None:
                        continue
                    st.caption(f"Plane {plane.upper()}")
                    compare_plot = px.line(
                        result["compare_df"],
                        x="TIME",
                        y="DIFF",
                        color="TURN",
                        hover_data=["TURN", "BPM"],
                        title=f"{result['time']}: TbT (TRAJECTORY DIFF)",
                        markers=True,
                    )
                    st.plotly_chart(compare_plot, use_container_width=True)

            if difference:
                st.subheader("Pair Difference")
                for plane in planes:
                    result = results[plane]
                    if result["pair_diff_df"] is None:
                        continue
                    st.caption(f"Plane {plane.upper()}")
                    diff_plot = px.line(
                        result["pair_diff_df"],
                        x="TIME",
                        y="DIFF",
                        color="TURN",
                        hover_data=["TURN", "BPM"],
                        title=f"{result['time']}: TbT (TRAJECTORY PAIR DIFF)",
                        markers=True,
                    )
                    st.plotly_chart(diff_plot, use_container_width=True)
            else:
                st.subheader("Trajectory")
                for plane in planes:
                    result = results[plane]
                    st.caption(f"Plane {plane.upper()}")
                    traj_plot = px.line(
                        result["trajectory_df"],
                        x="TIME",
                        y=result["axis"],
                        color="TURN",
                        hover_data=["TURN", "BPM"],
                        title=f"{result['time']}: TbT (TRAJECTORY)",
                        markers=True,
                    )
                    st.plotly_chart(traj_plot, use_container_width=True)
        else:
            st.subheader("Tables")
            for plane in planes:
                st.caption(f"Plane {plane.upper()}")
                st.dataframe(results[plane]["trajectory_df"], use_container_width=True)


if __name__ == "__main__":
    main()
