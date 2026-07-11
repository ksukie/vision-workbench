"""Pinned metadata for official model assets used by Vision Workbench."""

from __future__ import annotations

import os


EXTRA_MODEL_HOSTS_ENV = "VISION_WORKBENCH_EXTRA_MODEL_HOSTS"

# GitHub release metadata for ultralytics/assets tag v8.4.0.
# Values are (size_bytes, sha256).
YOLO26_ASSETS_V8_4_0 = {
    "yolo26n.pt": (5_544_453, "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    "yolo26s.pt": (20_422_725, "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"),
    "yolo26m.pt": (44_255_705, "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"),
    "yolo26l.pt": (53_211_173, "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),
    "yolo26x.pt": (118_667_365, "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"),
    "yolo26n-seg.pt": (6_719_965, "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"),
    "yolo26s-seg.pt": (23_467_933, "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"),
    "yolo26m-seg.pt": (54_750_385, "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"),
    "yolo26l-seg.pt": (63_700_037, "636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c"),
    "yolo26x-seg.pt": (142_129_861, "92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3"),
    "yolo26n-sem.pt": (3_487_283, "f3f293cca764de1f93044030d8d5612de9c5ffbf37c9c8ea1b69418b73038999"),
    "yolo26s-sem.pt": (13_252_403, "bc3e2152329831303de83e1af91d4b57d547e8794de8bb6edb1f2a622d1d5435"),
    "yolo26m-sem.pt": (28_924_971, "3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233"),
    "yolo26l-sem.pt": (36_167_967, "e4dfbd78b4bd54cbcb984aef035248af7e2559d227dc8fbf41b4f1864bc2cd21"),
    "yolo26x-sem.pt": (80_789_279, "eae8ff0d95bf96863c59c40bade61037870c7a39f2c53a8c49fd1b5022ed331d"),
}

DEFAULT_MODEL_DOWNLOAD_HOSTS = frozenset(
    {
        "download.pytorch.org",
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)


def trusted_model_download_hosts() -> frozenset[str]:
    """Return built-in hosts plus explicitly configured trusted mirrors."""

    extra = {
        value.strip().lower()
        for value in os.environ.get(EXTRA_MODEL_HOSTS_ENV, "").split(",")
        if value.strip()
    }
    return DEFAULT_MODEL_DOWNLOAD_HOSTS | extra
