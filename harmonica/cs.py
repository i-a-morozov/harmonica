from __future__ import annotations

from typing import Protocol
from typing import Any
from typing import Optional
from typing import Literal

from dataclasses import field
from dataclasses import dataclass


def convert(
    pv: str,
    *,
    gd: str = "bpm/harmonica/global",
    bd: str = "bpm/{name}/tbt",
    sections: tuple[str, ...] = ("LOCATION", "MONITOR", "FREQUENCY", "ACTION")
) -> str:
    pv = str(pv).strip()
    tokens = [token.strip() for token in pv.split(":")]
    _, head, *tail = tokens
    attribute = "_".join(token.lower() for token in tail if token)
    if not attribute:
        raise ValueError(f"invalid epics pv: {pv!r}")
    section = head.upper()
    if section in {name.strip().upper() for name in sections if name.strip()}:
        return f"{str(gd).strip().lower()}/{head.lower()}_{attribute}"
    return f"{str(bd).format(name=head.lower()).strip().lower()}/{attribute}"


class SC(Protocol):

    def get(self, channel: str) -> Any:
        ...


    def set(self, channel: str, value: Any) -> None:
        ...

@dataclass
class EpicsSC:

    library: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import epics
        self.epics = epics


    def get(self, channel: str) -> Any:
        return self.epics.caget(channel)


    def set(self, channel: str, value: Any) -> None:
        error = self.epics.caput(channel, value)
        if error is None:
            raise RuntimeError(f"failed to set {channel=}")


@dataclass
class TangoSC:


    host: str = "127.0.0.1"
    port: int = 12345
    link: str = "tango://{host}:{port}/{device}#dbase=no"
    attribute: Optional[str] = None
    tango: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import tango
        self.tango = tango


    def split(self, channel: str) -> tuple[str, str]:
        string = str(channel).strip()
        if not string:
            raise ValueError("empty channel")
        head, name, tail, *rest =  [token for token in string.strip("/").split("/") if token]
        device = "/".join([head, name, tail]).lower()
        attribute = "/".join(rest).strip()
        if not attribute:
            if self.attribute is None:
                raise ValueError("empty attribute")
            attribute = str(self.attribute).strip()
            if not attribute:
                raise ValueError("empty attribute")
        return device, attribute


    def proxy(self, device: str) -> Any:
        link = self.link.format(host=self.host, port=self.port, device=device)
        return self.tango.DeviceProxy(link)


    def get(self, channel: str, epics:bool=True) -> Any:
        if epics:
            channel = convert(channel)
        device, attribute = self.split(channel)
        return self.proxy(device).read_attribute(attribute).value


    def set(self, channel: str, value: Any, epics:bool=True) -> None:
        if epics:
            channel = convert(channel)
        device, attribute = self.split(channel)
        self.proxy(device).write_attribute(attribute, value)


def factory(target: Literal["epics", "tango"] = "epics", **kwargs: Any) -> SC:
    if target == "tango":
        return TangoSC(**kwargs)
    if target == "epics":
        return EpicsSC(**kwargs)
