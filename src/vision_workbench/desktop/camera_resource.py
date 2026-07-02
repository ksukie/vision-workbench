"""Shared camera access coordination for native Qt pages."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Sequence

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class CameraDeviceSnapshot:
    """Module-neutral camera device record cached across pages."""

    index: int
    backend_name: str
    backend_api_id: int
    name: str

    @classmethod
    def from_device(cls, device: object) -> "CameraDeviceSnapshot":
        backend = getattr(device, "backend")
        return cls(
            index=int(getattr(device, "index")),
            backend_name=str(getattr(backend, "name")),
            backend_api_id=int(getattr(backend, "api_id")),
            name=str(getattr(device, "name")),
        )

    def key(self) -> str:
        return f"{self.backend_name}:{self.index}"

    def to_device(self, device_cls, backend_cls):
        return device_cls(
            index=self.index,
            backend=backend_cls(self.backend_name, self.backend_api_id),
            name=self.name,
        )


@dataclass(frozen=True)
class CaptureProfileSnapshot:
    """Module-neutral camera profile record cached across pages."""

    width: int
    height: int
    fps: float
    fourcc: str
    backend_name: str = ""
    is_default: bool = False

    @classmethod
    def from_profile(cls, profile: object) -> "CaptureProfileSnapshot":
        return cls(
            width=int(getattr(profile, "width")),
            height=int(getattr(profile, "height")),
            fps=float(getattr(profile, "fps")),
            fourcc=str(getattr(profile, "fourcc")),
            backend_name=str(getattr(profile, "backend_name", "")),
            is_default=bool(getattr(profile, "is_default", False)),
        )

    def to_profile(self, profile_cls):
        return profile_cls(
            width=self.width,
            height=self.height,
            fps=self.fps,
            fourcc=self.fourcc,
            backend_name=self.backend_name,
            is_default=self.is_default,
        )


class CameraResourceCoordinator(QObject):
    """Tracks which native page is currently using camera hardware."""

    changed = Signal()
    devices_changed = Signal()
    profiles_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.RLock()
        self._owner_id = None  # type: str | None
        self._owner_label = None  # type: str | None
        self._devices = tuple()  # type: tuple[CameraDeviceSnapshot, ...]
        self._profiles_by_device = {}  # type: dict[str, tuple[CaptureProfileSnapshot, ...]]

    def owner_id(self) -> str | None:
        with self._lock:
            return self._owner_id

    def owner_label(self) -> str | None:
        with self._lock:
            return self._owner_label

    def is_owner(self, owner_id: str) -> bool:
        with self._lock:
            return self._owner_id == owner_id

    def is_available_for(self, owner_id: str) -> bool:
        with self._lock:
            return self._owner_id is None or self._owner_id == owner_id

    def reserve(self, owner_id: str, owner_label: str) -> bool:
        changed = False
        with self._lock:
            if self._owner_id is not None and self._owner_id != owner_id:
                return False
            if self._owner_id != owner_id or self._owner_label != owner_label:
                self._owner_id = owner_id
                self._owner_label = owner_label
                changed = True
        if changed:
            self.changed.emit()
        return True

    def release(self, owner_id: str) -> None:
        changed = False
        with self._lock:
            if self._owner_id == owner_id:
                self._owner_id = None
                self._owner_label = None
                changed = True
        if changed:
            self.changed.emit()

    def busy_message(self, requester_id: str) -> str:
        label = self.owner_label()
        if label and not self.is_available_for(requester_id):
            return f"相机正在被「{label}」使用，请先关闭该页面里的相机。"
        return ""

    def update_devices(self, devices: Sequence[object]) -> None:
        snapshots = tuple(_dedupe_devices(CameraDeviceSnapshot.from_device(device) for device in devices))
        with self._lock:
            self._devices = snapshots
            valid_keys = {device.key() for device in snapshots}
            self._profiles_by_device = {
                key: profiles
                for key, profiles in self._profiles_by_device.items()
                if key in valid_keys
            }
        self.devices_changed.emit()

    def cached_devices_as(self, device_cls, backend_cls) -> list:
        with self._lock:
            snapshots = tuple(self._devices)
        return [snapshot.to_device(device_cls, backend_cls) for snapshot in snapshots]

    def update_profiles(self, device: object, profiles: Sequence[object]) -> None:
        key = CameraDeviceSnapshot.from_device(device).key()
        snapshots = tuple(CaptureProfileSnapshot.from_profile(profile) for profile in profiles)
        with self._lock:
            self._profiles_by_device[key] = snapshots
        self.profiles_changed.emit(key)

    def cached_profiles_as(self, device: object | None, profile_cls) -> list:
        if device is None:
            return []
        key = CameraDeviceSnapshot.from_device(device).key()
        with self._lock:
            snapshots = tuple(self._profiles_by_device.get(key, tuple()))
        return [snapshot.to_profile(profile_cls) for snapshot in snapshots]


def _dedupe_devices(devices) -> list[CameraDeviceSnapshot]:
    seen = set()
    result = []
    for device in devices:
        key = device.key()
        if key in seen:
            continue
        seen.add(key)
        result.append(device)
    return result


_shared_camera_coordinator = None  # type: CameraResourceCoordinator | None


def shared_camera_coordinator() -> CameraResourceCoordinator:
    """Return the app-wide camera coordinator used by native Qt pages."""

    global _shared_camera_coordinator
    if _shared_camera_coordinator is None:
        _shared_camera_coordinator = CameraResourceCoordinator()
    return _shared_camera_coordinator
