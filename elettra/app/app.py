#!/usr/bin/env python3

"""Multipage entrypoint"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="TbT GUI", layout="wide")
    st.title("TbT GUI")
    st.caption("Use the left navigation panel to open an app.")


if __name__ == "__main__":
    main()
