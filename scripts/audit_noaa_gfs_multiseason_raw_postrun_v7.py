#!/usr/bin/env python
"""Offline, read-only V7 postrun audit for the frozen Track A raw producer.

The producer deliberately writes its canonical outputs only after decoding and
physical validation.  This auditor is a separate consumer of those outputs. It
does not import the producer, open a network client, mutate the artifact tree,
or recover an interrupted transaction.  An active launch lock is therefore a
conditional skip, never an invitation to inspect a moving tree.

V7 is an append-only source fork of the frozen V6 auditor.  The transaction,
raw, decoded and full offline replay audit remains unchanged.  V7 supersedes
only V6's authorization-sealer timestamp-precision false reject: immutable
RFC3339 UTC timestamps with zero through seven fractional digits are compared
as exact integer 100-nanosecond ticks before every Parquet/data/replay route.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
RUNNER_DEFAULT = REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
RECOVERY_RUNNER = (
    REPO / "scripts" / "run_noaa_gfs_multiseason_decode_recovery_v2.py"
)
RECOVERY_MODULE = REPO / "src" / "noaa_gfs_decode_recovery.py"
RECOVERY_BOOTSTRAP = REPO / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
RECOVERY_TEST = REPO / "tests" / "test_noaa_gfs_decode_recovery.py"
RECOVERY_BOOTSTRAP_TEST = (
    REPO / "tests" / "test_noaa_gfs_decode_recovery_bootstrap.py"
)
RECOVERY_RUNNER_TEST = (
    REPO / "tests" / "test_noaa_gfs_multiseason_decode_recovery_runner_v2.py"
)
RECOVERY_SEALER = (
    REPO / "scripts" / "seal_noaa_gfs_multiseason_decode_recovery_v2.py"
)
RECOVERY_SEALER_TEST = (
    REPO / "tests" / "test_seal_noaa_gfs_multiseason_decode_recovery_v2.py"
)
AUDITOR_TEST = REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v7.py"
SUPERSEDED_V2_AUDITOR = (
    REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v2.py"
)
SUPERSEDED_V2_AUDITOR_TEST = (
    REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v2.py"
)
V4_AUTH_RELATIVE = "prereg/decode_recovery_postrun_audit_authorization_v4.json"
V4_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V4.json"
)
V4_GO_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V4.json"
)
V4_SEALER = REPO / "scripts" / "seal_noaa_gfs_multiseason_postrun_audit_v4.py"
V4_SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v4.py"
FROZEN_V4_AUDITOR = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v4.py"
FROZEN_V4_AUDITOR_TEST = REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v4.py"
FROZEN_V4_SEALER = V4_SEALER
FROZEN_V4_SEALER_TEST = V4_SEALER_TEST
V5_AUTH_RELATIVE = "prereg/decode_recovery_postrun_audit_authorization_v5.json"
V5_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V5.json"
)
V5_GO_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V5.json"
)
V5_SEALER = REPO / "scripts" / "seal_noaa_gfs_multiseason_postrun_audit_v5.py"
V5_SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v5.py"
FROZEN_V5_AUDITOR = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v5.py"
FROZEN_V5_AUDITOR_TEST = REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v5.py"
FROZEN_V5_SEALER = V5_SEALER
FROZEN_V5_SEALER_TEST = V5_SEALER_TEST
V6_AUTH_RELATIVE = "prereg/decode_recovery_postrun_audit_authorization_v6.json"
V6_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V6.json"
)
V6_GO_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V6.json"
)
V6_SEALER = REPO / "scripts" / "seal_noaa_gfs_multiseason_postrun_audit_v6.py"
V6_SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v6.py"
FROZEN_V6_AUDITOR = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v6.py"
FROZEN_V6_AUDITOR_TEST = REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v6.py"
FROZEN_V6_SEALER = V6_SEALER
FROZEN_V6_SEALER_TEST = V6_SEALER_TEST
V7_AUTH_RELATIVE = "prereg/decode_recovery_postrun_audit_authorization_v7.json"
V7_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V7.json"
)
V7_GO_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V7.json"
)
V7_SEALER = REPO / "scripts" / "seal_noaa_gfs_multiseason_postrun_audit_v7.py"
V7_SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v7.py"
V4_POSTRUN_REPORT_CANDIDATES = tuple(
    relative
    for version in (2, 3, 4)
    for relative in (
        f"TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
    )
)
V5_POSTRUN_REPORT_CANDIDATES = tuple(
    relative
    for version in (2, 3, 4, 5)
    for relative in (
        f"TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
    )
)
V6_POSTRUN_REPORT_CANDIDATES = tuple(
    relative
    for version in (2, 3, 4, 5, 6)
    for relative in (
        f"TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
    )
)
V7_POSTRUN_REPORT_CANDIDATES = tuple(
    relative
    for version in (2, 3, 4, 5, 6, 7)
    for relative in (
        f"TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
        f"decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V{version}.json",
    )
)
V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE = (
    "incidents/"
    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_AUTH_SEALER_TIMESTAMP_"
    "PRECISION_FALSE_REJECT_V1.json"
)
V7_TIMESTAMP_FALSE_REJECT_INCIDENT_SIZE_BYTES = 11_799
V7_TIMESTAMP_FALSE_REJECT_INCIDENT_SHA256 = (
    "722f1540cf73ed78aeff25ea3a2ad9300e40aa13fcc068f998898251e68ffd19"
)
V7_TIMESTAMP_FALSE_REJECT_INCIDENT_CANONICAL_BYTES = 9_607
V7_TIMESTAMP_FALSE_REJECT_INCIDENT_CANONICAL_SHA256 = (
    "ba20996218170b773cde52e99acafaa7e4b426d679c87c6fbeaafb31a6813113"
)
V7_TIMESTAMP_INCIDENT_POSTFAILURE_CANONICAL_BYTES = 1_825
V7_TIMESTAMP_INCIDENT_POSTFAILURE_CANONICAL_SHA256 = (
    "08b5d2c4bf653243deae63ed18067f9e098edb16b972a66ce121af88ee337837"
)
V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE = (
    "incidents/"
    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_HISTORICAL_IDENTITY_FALSE_REJECT_V1.json"
)
V6_HISTORICAL_FALSE_REJECT_INCIDENT_SIZE_BYTES = 10_523
V6_HISTORICAL_FALSE_REJECT_INCIDENT_SHA256 = (
    "09b92677c719631c512a4b227653a5e5b2def9dd4b64eca787bd7c323ba1d61d"
)
V2_FALSE_REJECT_INCIDENT_RELATIVE = (
    "incidents/"
    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V2_IDENTITY_SCHEMA_FALSE_REJECT_V1.json"
)
V2_FALSE_REJECT_INCIDENT_SIZE_BYTES = 17_309
V2_FALSE_REJECT_INCIDENT_SHA256 = (
    "4c8133de1dba5e7677f1dab4b001c694a3f655dad415ca2a62ddefb6249d5c7a"
)
V4_PATH_MISPUBLISH_INCIDENT_RELATIVE = (
    "incidents/"
    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V3_GO_PATH_MISPUBLISH_V1.json"
)
V4_PATH_MISPUBLISH_INCIDENT_SIZE_BYTES = 20_409
V4_PATH_MISPUBLISH_INCIDENT_SHA256 = (
    "8863465b9060457c08ebecafda5bfcb4fc3aed15565a5b1bc5bd126711575263"
)
FROZEN_V3_AUTH_RELATIVE = "prereg/decode_recovery_postrun_audit_authorization_v3.json"
FROZEN_V3_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V3.json"
)
FROZEN_V3_CANONICAL_GO_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V3.json"
)
FROZEN_V3_MISPLACED_GO_RELATIVE = "prereg/decode_recovery_postrun_audit_go_v3.json"
FROZEN_V3_AUTH_IDENTITY = {
    "path": FROZEN_V3_AUTH_RELATIVE,
    "size_bytes": 47_777,
    "sha256": "c126d434be4ced4948446532474ab4159db7e13c0056b0b85ff4443a82856921",
}
FROZEN_V3_REVIEW_IDENTITY = {
    "path": FROZEN_V3_REVIEW_RELATIVE,
    "size_bytes": 35_910,
    "sha256": "1605384d1032e704e6b44ca2bcaa680154ec0d34528a6766a9fedd6cc76aadf1",
}
FROZEN_V3_MISPLACED_GO_IDENTITY = {
    "path": FROZEN_V3_MISPLACED_GO_RELATIVE,
    "size_bytes": 34_360,
    "sha256": "22f30f028fc09a0c872ad36cf073c67732c6c644b98c4da7d79ef382801cba58",
}
FROZEN_V3_MISPLACED_GO_PAYLOAD_CANONICAL_SHA256 = (
    "da2cdd39576460882b7485f55201c4cdcfaa27cf236bd617e715f2bdfbd9a7b8"
)
FROZEN_V3_AUDITOR = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v3.py"
FROZEN_V3_AUDITOR_TEST = REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v3.py"
FROZEN_V3_SEALER = REPO / "scripts" / "seal_noaa_gfs_multiseason_postrun_audit_v3.py"
FROZEN_V3_SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v3.py"
FROZEN_V3_SOURCE_IDENTITIES = {
    "superseded_v3_auditor": {
        "path": str(FROZEN_V3_AUDITOR.resolve()),
        "size_bytes": 361_313,
        "sha256": "a1ae671d629cc0bbe83fc1bbdc213d2879a9c79140bd9483f058b549475b4df4",
    },
    "superseded_v3_auditor_test": {
        "path": str(FROZEN_V3_AUDITOR_TEST.resolve()),
        "size_bytes": 171_989,
        "sha256": "1b093fe86bb42ccf79b6ddcab17f7a58220ac436bcfd4eecff83adb7e182bcbf",
    },
    "superseded_v3_sealer": {
        "path": str(FROZEN_V3_SEALER.resolve()),
        "size_bytes": 47_629,
        "sha256": "229fa7f6530ec63739e6c7ab6b4e76f7ab8f43fefd4df933a4373f8a5298ef75",
    },
    "superseded_v3_sealer_test": {
        "path": str(FROZEN_V3_SEALER_TEST.resolve()),
        "size_bytes": 26_503,
        "sha256": "6a3dd006bc8c23f3ec15030172d1ba899a285cae085152c003c23f3ca0cf4f18",
    },
}
FROZEN_V4_SOURCE_IDENTITIES = {
    "superseded_v4_auditor": {
        "path": str(FROZEN_V4_AUDITOR.resolve()),
        "size_bytes": 389_142,
        "sha256": "a1fb69d5ab0806bd0f6df8ad528ee12796c02032db62b3a1969c0d1dbbdcd725",
    },
    "superseded_v4_auditor_test": {
        "path": str(FROZEN_V4_AUDITOR_TEST.resolve()),
        "size_bytes": 188_022,
        "sha256": "ab836bd6a0794e0501f157b3eb606fce50f36fd0e143873f1a638903a8101d00",
    },
    "superseded_v4_sealer": {
        "path": str(FROZEN_V4_SEALER.resolve()),
        "size_bytes": 58_029,
        "sha256": "62cb1d6e4db476ed788336eadc07b529e1a46c8e6583c4a247b4bb9b08928816",
    },
    "superseded_v4_sealer_test": {
        "path": str(FROZEN_V4_SEALER_TEST.resolve()),
        "size_bytes": 40_506,
        "sha256": "ae54f49b6344ec32293ce2f3abce0a30dab8b748689e9a4843b0317fce7608f7",
    },
}
FROZEN_V5_SOURCE_IDENTITIES = {
    "superseded_v5_auditor": {
        "path": str(FROZEN_V5_AUDITOR.resolve()),
        "size_bytes": 481_173,
        "sha256": "81e18256bf53bcdc3ba27a37787e1b2240bf0ea286d7f8c758d9f7d5e96ce6b1",
    },
    "superseded_v5_auditor_test": {
        "path": str(FROZEN_V5_AUDITOR_TEST.resolve()),
        "size_bytes": 228_289,
        "sha256": "45ce935aeb945a06d4eaafb0885320348d27e12d8343387f53a8cac108182fc1",
    },
    "superseded_v5_sealer": {
        "path": str(FROZEN_V5_SEALER.resolve()),
        "size_bytes": 76_117,
        "sha256": "bab2302dad801f554ea2c62546abd66edbca6e94e1841dcd05f4f5dbe7e215a4",
    },
    "superseded_v5_sealer_test": {
        "path": str(FROZEN_V5_SEALER_TEST.resolve()),
        "size_bytes": 39_039,
        "sha256": "215aa7295aeed09cfc799d7950999f19d9f243b62388d7b88a5e39db0a8dc963",
    },
}
FROZEN_V6_SOURCE_IDENTITIES = {
    "superseded_v6_auditor": {
        "path": str(FROZEN_V6_AUDITOR.resolve()),
        "size_bytes": 556_811,
        "sha256": "97519582cb57c426bd456a30f1cc34a27d966ae7fd1cc2df450d69795bfba415",
    },
    "superseded_v6_auditor_test": {
        "path": str(FROZEN_V6_AUDITOR_TEST.resolve()),
        "size_bytes": 278_146,
        "sha256": "eda9db77675d69fc161c9838ae54d3e824859a44025235630b50ddb02875d97a",
    },
    "superseded_v6_sealer": {
        "path": str(FROZEN_V6_SEALER.resolve()),
        "size_bytes": 95_744,
        "sha256": "bd3160ac40fdb6d72dbcb3233baeecb7320c9168499b44211496445fd03fb4c3",
    },
    "superseded_v6_sealer_test": {
        "path": str(FROZEN_V6_SEALER_TEST.resolve()),
        "size_bytes": 59_845,
        "sha256": "e0f3174da212de32ba51a6af1650aba7b358520f86aabd03412a7ef47ad7e673",
    },
}
FROZEN_V5_CONTROL_IDENTITIES = {
    "authorization": {
        "path": V5_AUTH_RELATIVE,
        "size_bytes": 23_792,
        "sha256": "7b23fcf77b35e529398bd27ee523c6c21773a152f791c599fc5af1f7b906cb3c",
    },
    "independent_review": {
        "path": V5_REVIEW_RELATIVE,
        "size_bytes": 8_455,
        "sha256": "5808d9b370ab769971a1502a659cf3849609cad4702c3cbb411b9db9f40c2ee1",
    },
    "independent_go": {
        "path": V5_GO_RELATIVE,
        "size_bytes": 7_346,
        "sha256": "ebb147b212d2c767e1fad1aaa631f2cc01f9292a769941f064e1fef01a0b022f",
    },
}
FROZEN_V5_AUDIT_ATTEMPT_ID = "postrun_audit_v5__20260811T021615245290Z"
FROZEN_V5_RECOVERY_ATTEMPT_ID = "decode_recovery_v2__20260810T211017500046Z"
FROZEN_V5_AUTH_CREATED_UTC = "2026-08-11T02:16:15.245290Z"
FROZEN_V5_REVIEW_CREATED_UTC = "2026-08-11T02:25:11.379223Z"
FROZEN_V5_GO_CREATED_UTC = "2026-08-11T02:31:34.302619Z"
FROZEN_V5_AUTH_PAYLOAD_CANONICAL_BYTES = 20_720
FROZEN_V5_AUTH_PAYLOAD_CANONICAL_SHA256 = (
    "472565c81d5231fc210d23e562c872e3de2e83702f4dada4372372102b02399f"
)
FROZEN_V5_REVIEW_PAYLOAD_CANONICAL_BYTES = 7_552
FROZEN_V5_REVIEW_PAYLOAD_CANONICAL_SHA256 = (
    "41574855eac5b5f84d958a2829dd90bec8a2c83cc5b4e51eb9d12206ee688f41"
)
FROZEN_V5_GO_PAYLOAD_CANONICAL_BYTES = 6_589
FROZEN_V5_GO_PAYLOAD_CANONICAL_SHA256 = (
    "77e4fda3cc9000fbc84990504c6523febb669d83dee9c16752d4acf04c4a270f"
)
FROZEN_V5_TEST_EVIDENCE_CANONICAL_SHA256 = (
    "324f1429e0ff3b3b58a9a65120fee234f44a2cb580e17594a024106b8e76726f"
)
V5_TRANSPORT_INCIDENT_RELATIVE = (
    "incidents/"
    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_REVIEW_TRANSPORT_TRUNCATION_V1.json"
)
V5_TRANSPORT_INCIDENT_SIZE_BYTES = 8_392
V5_TRANSPORT_INCIDENT_SHA256 = (
    "cf88b37f15da75e17fb6c02588f8a3fb011b2c329d7d16f2f220760933d21b4b"
)
V5_TRANSPORT_CORRECTION_RELATIVE = (
    "incidents/"
    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_REVIEW_TRANSPORT_TRUNCATION_"
    "INVENTORY_DIGEST_CORRECTION_V1.json"
)
V5_TRANSPORT_CORRECTION_SIZE_BYTES = 3_368
V5_TRANSPORT_CORRECTION_SHA256 = (
    "05af8e1711916e6c65c9133aabbb1adbcf6bd4bbac51c2b7396210f6f9940a19"
)
FROZEN_V4_AUTH_IDENTITY = {
    "path": V4_AUTH_RELATIVE,
    "size_bytes": 55_965,
    "sha256": "63a0ae3937cc85a92b2f40cbb4d5c92420f58fc79fe478251427bcde4221839e",
}
FROZEN_V4_REVIEW_IDENTITY = {
    "path": V4_REVIEW_RELATIVE,
    "size_bytes": 38_297,
    "sha256": "a5bb1ff2bc52d78fbd2dea8ce8450e46d1d0dcf9645a979bb0e4c4f0a99a5063",
}
FROZEN_V4_CANONICAL_GO_RELATIVE = V4_GO_RELATIVE
FROZEN_V4_AUTH_PAYLOAD_CANONICAL_BYTES = 47_235
FROZEN_V4_AUTH_PAYLOAD_CANONICAL_SHA256 = (
    "330d42d2d976c3f86f38386918f29dc8c0e8d7b7ac6760aec5f8e83cae0faeef"
)
FROZEN_V4_TEST_EVIDENCE_CANONICAL_SHA256 = (
    "9b8d9e519fb1624a7bc443a2eddb839f70771d726a88fb60611ea703515062db"
)
FROZEN_V4_AUDIT_ATTEMPT_ID = "postrun_audit_v4__20260811T002740753499Z"
FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_BYTES = 30_931
FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_SHA256 = (
    "75422dbc8b27a3975e697841c58969b61125df682cb0a1d713c3be65ab13919a"
)
FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_BYTES = 23_622
FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_SHA256 = (
    "336e21f28da5b0b67414a412cb7ac806c5fef81d64eb431d452c107c9b23e1b8"
)
FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_BYTES = 20_605
FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256 = (
    "cad4e0ef0d6d63ca6655fdc4555b55fc7c8e1350625e439b022fd2a6296492d8"
)
FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_BYTES = 31_543
FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_SHA256 = (
    "b5e893bca005a689e82a1665a0a0bf3e8eb836ba444383f17c50ae5c9860267c"
)
FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY = {
    "size_bytes": 38_918,
    "sha256": "d0e1a6355dd0172af29cc2a2e7d992f1215a8660e80deb41fa2e45ded423dc42",
}
V5_CORRECTED_PRIMARY_INCIDENT_CANONICAL_SHA256 = (
    "0e1abfb87c79bf3ee15cd5c59eb9aac9fee7ef13d53aa54e2cd84538cfee3cdd"
)
V5_PRIMARY_INCIDENT_CANONICAL_BYTES = 7_350
V5_PRIMARY_INCIDENT_CANONICAL_SHA256 = (
    "3f81d9c4a76d6eacf7b16a79f4e9b78a97a24eccd0895b5bcc8ba3bc0530f404"
)
V5_CORRECTION_SEMANTIC_CANONICAL_BYTES = 3_021
V5_CORRECTION_SEMANTIC_CANONICAL_SHA256 = (
    "5982d9d77f482bd3abd2b73bc782ad02f3d47f2a3c95218083d1fee6da250c71"
)
V5_ERRONEOUS_PROGRESS_INVENTORY_SHA256 = (
    "cad4e0ef3874116322d04e9b2ed6f0cf8656b23a045568c35aad22507be318cf"
)
V4_RECOVERY_ATTEMPT_ID = "decode_recovery_v2__20260810T211017500046Z"
V4_RUNTIME_IDENTITY_SHA256 = (
    "a81cdbeafa3906e4d1bbc02455b156c58d226323a90b7f8911017283e3135d4b"
)
V4_PROGRESS_INVENTORY_CANONICAL_SHA256 = (
    "49e1f24641657aa1aa7ceb014a0ef1d5e3d743b64887cc4c1237824fe420c2b6"
)
V4_PROGRESS_INVENTORY_CANONICAL_BYTES = 21_217
RECOVERY_AUTH_RELATIVE = "prereg/decode_recovery_authorization_v2.json"
RECOVERY_GO_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V2.json"
)
RECOVERY_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V2.json"
)
RECOVERY_INCIDENT_RELATIVE = (
    "incidents/TRACK_A_RAW_DECODE_MEMFS_THREAD_INIT_FAILURE_V1.json"
)
RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE = (
    "incidents/TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_V1.json"
)
RECOVERY_FLAWED_LAUNCH_INCIDENT_RELATIVE = (
    "incidents/TRACK_A_DECODE_RECOVERY_LAUNCH_BOOTSTRAP_IMPORT_CONTRACT_NAMEERROR_V1.json"
)
RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE = (
    "incidents/TRACK_A_DECODE_RECOVERY_LAUNCH_NAMEERROR_IDENTITY_CORRECTION_V1.json"
)
RECOVERY_RAW_CACHE_LOCK_RELATIVE = (
    "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V2.json"
)
RECOVERY_REAL_PREFLIGHT_RELATIVE = (
    "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2.json"
)
RECOVERY_V1_CONTROL_RELATIVES = (
    "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V1.json",
    "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V1.json",
    "prereg/decode_recovery_authorization_v1.json",
    "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V1.json",
    "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V1.json",
)
RECOVERY_V2_CONTROL_RELATIVES = (
    RECOVERY_RAW_CACHE_LOCK_RELATIVE,
    RECOVERY_REAL_PREFLIGHT_RELATIVE,
    RECOVERY_AUTH_RELATIVE,
    RECOVERY_REVIEW_RELATIVE,
    RECOVERY_GO_RELATIVE,
)
RECOVERY_CONTROL_RELATIVES = RECOVERY_V1_CONTROL_RELATIVES + RECOVERY_V2_CONTROL_RELATIVES
V6_HISTORICAL_FAILED_HEAD_IDENTITIES = {
    "planned_postrun_auditor": {
        "path": str((REPO / "scripts/audit_noaa_gfs_multiseason_raw_postrun_v1.py").resolve()),
        "size_bytes": 247_925,
        "sha256": "84ba333a794430d95cb83a7c98d6d0477345d8b6c494b51505a83c49f1a2bc2b",
    },
    "planned_postrun_auditor_test": {
        "path": str((REPO / "tests/test_noaa_gfs_multiseason_raw_postrun_v1.py").resolve()),
        "size_bytes": 106_360,
        "sha256": "770bdfc2c657565237de7f90b488fb8aa47a6c43a8e50e1492eabebc4eb9f46e",
    },
    "recovery_bootstrap": {
        "path": str((REPO / "src/noaa_gfs_decode_recovery_bootstrap.py").resolve()),
        "size_bytes": 16_925,
        "sha256": "e7637ea8f96a41d9cd9cb45467ede90b0dc08aaa143a97f28f438eacc79dc78e",
    },
    "recovery_core": {
        "path": str((REPO / "src/noaa_gfs_decode_recovery.py").resolve()),
        "size_bytes": 26_415,
        "sha256": "6d1f05e36a1d60cdb73a6d8cbd6e3cc9bdfeac36871d75a6fc27a76952a8825c",
    },
    "recovery_runner": {
        "path": str((REPO / "scripts/run_noaa_gfs_multiseason_decode_recovery_v1.py").resolve()),
        "size_bytes": 117_445,
        "sha256": "9a1ea2a118d497dbe93fd9230dd82fbc832985f599a5c8ba13236a8d58919520",
    },
    "recovery_sealer": {
        "path": str((REPO / "scripts/seal_noaa_gfs_multiseason_decode_recovery_v1.py").resolve()),
        "size_bytes": 40_285,
        "sha256": "da4b3d5eef14ab0b156faed158af668a5627bb5dc2a50f7d8c5e7a4c8589370d",
    },
    "recovery_sealer_test": {
        "path": str((REPO / "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v1.py").resolve()),
        "size_bytes": 7_261,
        "sha256": "3ea32a3a9ac17d1a2b36efacbde066bb89e7ca772173d2317425afcb8380b446",
    },
}
V6_HISTORICAL_ZERO_STATE = {
    "absent_control_paths": [
        "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V1.json",
        "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V1.json",
        "prereg/decode_recovery_authorization_v1.json",
        "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V1.json",
        "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V1.json",
    ],
    "canonical_recovery_outputs_absent": True,
    "recovery_active_locks_absent": True,
    "seal_temporary_files_absent": True,
    "matching_bootstrap_pyc_files": 0,
    "only_original_failed_output_transaction_present": True,
}
V6_HISTORICAL_FAILED_HEADS_CANONICAL_BYTES = 1_672
V6_HISTORICAL_FAILED_HEADS_CANONICAL_SHA256 = (
    "e3c36f0e55d12d4ff62c04feabdeadd4c5cbca9edad712fd50bf6dbde7f7cd1c"
)
V6_HISTORICAL_ZERO_STATE_CANONICAL_BYTES = 513
V6_HISTORICAL_ZERO_STATE_CANONICAL_SHA256 = (
    "c053b483bdeff428ada753609bc698677ac3f01169a953758412206cebb32f63"
)
V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_BYTES = 288
V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_SHA256 = (
    "1a0505b4cfd9a93cf1838a9689058f7447645077008ac39272939f547ff9f524"
)
RECOVERY_REAL_PILOT = (
    REPO
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
    / "provenance"
    / "raw"
    / "f039"
    / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
)
RECOVERY_FAILED_ATTEMPT = "20260810T151057590738Z__pid3536__5f3e0fb20132"
RECOVERY_INCIDENT_SHA256 = (
    "52bd92b00d3351e6b1061e47e3306ccd621311913113d6652033081be1cd836c"
)
RECOVERY_SEALER_ENTRYPOINT_INCIDENT_SHA256 = (
    "9e74f57dcddbd0480464fda5953014e5be411b927de99beef58d7ab5bb10afbb"
)
RECOVERY_FLAWED_LAUNCH_INCIDENT_SHA256 = (
    "ecaafefe69e5076d28feab458415d4ac1a120a523027150932dde9668fddc973"
)
RECOVERY_LAUNCH_IDENTITY_CORRECTION_SHA256 = (
    "91e764be2e4db3e77bd47892f2d4cf551ddfe225d48c8c431c5263ce4a1c2a7b"
)
RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY = {
    "path": "scripts/run_noaa_gfs_multiseason_decode_recovery_v1.py",
    "size_bytes": 128_824,
    "sha256": "82944af662caf1fedc33b6dc3cff21c8c1cf44ee66541c241ef77c8041341291",
}
RECOVERY_V1_CHAIN_IDENTITIES = {
    "authorization": {
        "path": "prereg/decode_recovery_authorization_v1.json",
        "size_bytes": 14_144,
        "sha256": "b2458408b40cef960b949adcb08b6002ff5bc16814653b566f6d5846edcacae0",
    },
    "independent_go": {
        "path": "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V1.json",
        "size_bytes": 4_172,
        "sha256": "c3086d0f3f302e86659802c9b626927173e3321b864d0bdda4706380f957b1ee",
    },
    "independent_review": {
        "path": "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V1.json",
        "size_bytes": 5_231,
        "sha256": "d8e626a7b5e54ba1c9d568ee8939376414dd2f82addadff8b7a3d6df8b514ef4",
    },
    "raw_cache_lock": {
        "path": "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V1.json",
        "size_bytes": 3_721,
        "sha256": "e0b9b2af37dd8033712754a0b69a8f4f50078aa0318687d7d32377384539d6a8",
    },
    "real_spawn_preflight": {
        "path": "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V1.json",
        "size_bytes": 34_327,
        "sha256": "7c5f25ce1dfb8d55c5f5db3f60a8312336f3d4770e5fc99ed44e235f1392a8e0",
    },
    "recovery_runner": {
        "path": "scripts/run_noaa_gfs_multiseason_decode_recovery_v1.py",
        "size_bytes": 128_749,
        "sha256": "af61a52044d68b4a798ba24395d6c17ae56b827ab291ae67bbf46fd3e2aa1dcd",
    },
}
RECOVERY_FLAWED_V1_CHAIN_IDENTITIES = {
    **RECOVERY_V1_CHAIN_IDENTITIES,
    "recovery_runner": RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY,
}
RECOVERY_V1_SUPPORT_IDENTITIES = {
    "recovery_runner": RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"],
    "recovery_runner_test": {
        "path": "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v1.py",
        "size_bytes": 28_195,
        "sha256": "fb2dc7c671496bf95aae4dd7c3f9a9e3d3baeeef2697f47c964f2ed4d54c79f8",
    },
    "recovery_sealer": {
        "path": "scripts/seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "size_bytes": 51_226,
        "sha256": "8d677ccf5de897886310a744a81d9adf361ab31d6e44bc361d5d89f7e147064a",
    },
    "recovery_sealer_test": {
        "path": "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "size_bytes": 14_488,
        "sha256": "7820670226a184b32b9749a04829a2a830b8ef0dd30cd621398ca497d4ed82b5",
    },
    "postrun_auditor": {
        "path": "scripts/audit_noaa_gfs_multiseason_raw_postrun_v1.py",
        "size_bytes": 271_137,
        "sha256": "208e8a4e7098c1d9ffa0d6a2ce75d89be59fd58e6c7f5ccffd8a8d541bff1f40",
    },
    "postrun_auditor_test": {
        "path": "tests/test_noaa_gfs_multiseason_raw_postrun_v1.py",
        "size_bytes": 120_044,
        "sha256": "eb6c7c90819cc2b3bb4789a6323b4a7ee580a8675b00a945b6bcfeb24478e800",
    },
}
RECOVERY_BOOTSTRAP_TRUST_POLICY = {
    "sole_explicit_preinitializer_trust_anchor": True,
    "adversarial_code_swap_proof_claimed": False,
    "source_top_level_stdlib_allowlist_required": True,
    "matching_bootstrap_pyc_required_absent_before_and_after": True,
    "python_dont_write_bytecode_env_required": "1",
    "python_pycache_prefix_required_absent": True,
    "child_detects_drift_before_decoder_core_work": True,
    "top_level_module_name": "noaa_gfs_decode_recovery_bootstrap",
    "workspace_src_search_path_required_first": True,
    "pythonpath_required_exact_workspace_src": True,
    "src_package_initializer_executed": False,
    "bootstrap_module_package": "",
    "stdlib_shadow_candidates_required_absent": True,
    "top_level_module_competing_candidates_required_absent": True,
}
RECOVERY_BOOTSTRAP_STDLIB_IMPORTS = (
    "__future__",
    "ast",
    "collections.abc",
    "concurrent.futures",
    "hashlib",
    "importlib.machinery",
    "multiprocessing",
    "os",
    "pathlib",
    "socket",
    "sys",
    "types",
    "typing",
)
RECOVERY_PROCESS_SCHEDULING_FLAKE_INCIDENT = {
    "artifact_type": "DECODE_RECOVERY_PRESEAL_PROCESS_SCHEDULING_FLAKE_INCIDENT",
    "status": "CLOSED_BY_SYNCHRONIZED_FIRST_WORKER_WAVE_GATE",
    "observed_during_unsealed_dry_run": True,
    "observed_test": "test_real_grib_cold_start_passes_in_exactly_seven_spawn_processes",
    "observed_failure": "LESS_THAN_SEVEN_DISTINCT_WORKER_PIDS_FROM_FAST_UNSYNCHRONIZED_POOL",
    "test_process_exit_code": 1,
    "seal_artifacts_published": 0,
    "network_requests": 0,
    "root_cause": "PROCESS_POOL_TASK_SCHEDULING_DID_NOT_PROVE_ALL_REQUESTED_WORKERS",
    "corrective_action": "FIRST_TASK_IN_EACH_OF_SEVEN_WORKERS_WAITS_ON_SHARED_BARRIER_BEFORE_DECODE",
    "retry_or_selected_result_used_as_final_evidence": False,
}
RECOVERY_NONCONTRACT_MONOLITHIC_DIAGNOSTIC = {
    "artifact_type": "DECODE_RECOVERY_NONCONTRACT_MONOLITHIC_TEST_DIAGNOSTIC",
    "status": "EXPECTED_FAIL_PREIMPORT_RESIDUE_NOT_USED_AS_SEAL_EVIDENCE",
    "topology": "ONE_INTERPRETER_MULTI_FILE_NONCONTRACT",
    "command": [
        r".venv\Scripts\python.exe",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_noaa_gfs_decode_recovery.py",
        "tests/test_noaa_gfs_decode_recovery_bootstrap.py",
        "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v1.py",
        "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "tests/test_noaa_gfs_multiseason_raw_v2.py",
        "tests/test_noaa_gfs_multiseason_raw_postrun_v1.py",
    ],
    "environment": {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPYCACHEPREFIX": "ABSENT",
    },
    "exit_code": 1,
    "pytest_duration_seconds": 62.83,
    "shell_duration_seconds": 63.3,
    "passed": 146,
    "skipped": 1,
    "failed": 13,
    "failure_partition": {
        "bootstrap_tests_rejected_preloaded_src": 11,
        "canonical_spawn_child_passed_then_parent_src_residue_failed": 1,
        "static_audit_rejected_preimported_runner": 1,
    },
    "canonical_spawn_subreport": {
        "worker_count": 7,
        "tasks": 70,
        "first_wave_synchronized": True,
        "child_src_package_initializer_executed": False,
        "controls_created": 0,
        "network_requests": 0,
        "writes": 0,
    },
    "outer_network_guard_installed": False,
    "outer_network_requests": "NOT_INSTRUMENTED_NO_ZERO_CLAIM",
    "outer_data_reads": "NOT_INSTRUMENTED_NO_ZERO_CLAIM",
    "outer_writes": "NOT_INSTRUMENTED_NO_ZERO_CLAIM",
    "seal_control_publications_observed_after_run": 0,
    "used_as_contract_evidence": False,
    "required_contract_topology": "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS",
}
RECOVERY_PYTEST_NETWORK_GUARD_SHA256 = (
    "6151203525ff54b11acb93b13c3997ddf9c2df5c54e776ebddf4ab00fd1c7fa8"
)
RECOVERY_TEST_RELATIVES = (
    "tests/test_noaa_gfs_decode_recovery.py",
    "tests/test_noaa_gfs_decode_recovery_bootstrap.py",
    "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v2.py",
    "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v2.py",
    "tests/test_noaa_gfs_multiseason_raw_v2.py",
    "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py",
)
RECOVERY_COMPILED_RELATIVES = (
    "src/noaa_gfs_decode_recovery.py",
    "src/noaa_gfs_decode_recovery_bootstrap.py",
    "scripts/run_noaa_gfs_multiseason_decode_recovery_v2.py",
    "scripts/seal_noaa_gfs_multiseason_decode_recovery_v2.py",
    "scripts/audit_noaa_gfs_multiseason_raw_postrun_v2.py",
    *RECOVERY_TEST_RELATIVES,
)
RECOVERY_REQUIRED_TEST_NAMES = {
    "tests/test_noaa_gfs_decode_recovery_bootstrap.py": [
        "test_bootstrap_source_is_stdlib_only_and_has_no_decoder_import",
        "test_top_level_alias_is_picklable_and_clean_child_avoids_src_initializer",
        "test_child_import_contract_tamper_fails_closed",
        "test_child_rejects_same_alias_package_or_extension_candidate",
        "test_bootstrap_cache_and_bytecode_policy_are_fail_closed",
        "test_poisoned_timestamp_pyc_is_ignored_for_verified_source",
        "test_real_pilot_uses_exactly_seven_stdlib_bootstrap_processes",
        "test_production_worker_uses_stdlib_bootstrap_and_exact_source",
    ],
    "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v2.py": [
        "test_bootstrap_top_level_import_contract_is_exact_and_src_free",
        "test_v2_run_prefix_reaches_postpilot_alias_check_before_claim_and_writes_nothing",
        "test_v2_static_audit_uses_same_real_preclaim_prefix",
        "test_v2_identity_correction_is_sole_v1_chain_authority",
        "test_v2_identity_correction_tamper_fails",
        "test_required_test_names_nested_sealer_mapping_passes",
        "test_required_test_names_nested_mapping_tamper_fails",
        "test_bootstrap_import_root_rejects_casefolded_stdlib_shadow",
        "test_bootstrap_import_root_rejects_same_alias_competitor",
        "test_noncontract_monolithic_diagnostic_exact_semantics_pass",
        "test_noncontract_monolithic_diagnostic_tamper_fails",
        "test_locked_prestate_recloses_transaction_race_after_initial_audit",
        "test_plan_publication_race_never_overwrites",
        "test_commit_record_publication_race_never_overwrites",
        "test_original_progress_payloads_close_every_checkpoint",
        "test_independent_review_exact_schema_and_twelve_checks_pass",
        "test_independent_review_missing_extra_or_false_check_fails",
    ],
    "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v2.py": [
        "test_static_audit_writes_nothing",
        "test_seal_publishes_exactly_three_v2_controls",
        "test_v1_and_v2_control_temporary_siblings_fail_closed",
        "test_identity_correction_rejected_transient_wrapper_passes_exactly",
        "test_identity_correction_rejected_transient_wrapper_tamper_fails",
        "test_json_publisher_fails_on_race_without_overwrite",
        "test_failed_direct_file_seal_incident_binds_exact_alias_remediation",
        "test_noncontract_monolithic_diagnostic_is_recorded_but_never_selected",
        "test_direct_file_write_mode_fails_before_data_or_publication",
        "test_imported_programmatic_seal_fails_before_data_or_publication",
        "test_imported_programmatic_spawn_audit_fails_before_data_access",
        "test_direct_file_static_audit_remains_read_only_and_allowed",
        "test_canonical_module_entrypoint_real_seven_spawn_is_read_only",
    ],
}
RECOVERY_FROZEN_EXTERNAL_SHA256 = {
    "recovery_runner": "892b9e0fb26cb0f9be19c77bcaa04c656e24d2404f42b2102596914e3d1c31d1",
    "recovery_module": "6d1f05e36a1d60cdb73a6d8cbd6e3cc9bdfeac36871d75a6fc27a76952a8825c",
    "recovery_bootstrap": "09e350e19db0d278bbffa68dabac538d3cda4563c68afb5e00241d45e577e094",
    "recovery_test": "cf6d7a512f6cc997c1de7630c1d3e3b81e77a071af96a41c100ccf98b7ecea96",
    "recovery_bootstrap_test": "a918e192d6d9f172a112f6bea06613ab07524f96b83cab8505175890883928b3",
    "recovery_runner_test": "f4c248ddcd04d574846f96b0ef504ccfaa41c4fc97dece2b5b77ffae21767bc3",
}

FEATURE_PAIRS = (
    ("HPBL", "surface"),
    ("UGRD", "925 mb"),
    ("VGRD", "925 mb"),
    ("UGRD", "950 mb"),
    ("VGRD", "950 mb"),
    ("UGRD", "975 mb"),
    ("VGRD", "975 mb"),
    ("UGRD", "1000 mb"),
    ("VGRD", "1000 mb"),
)
FEATURE_COLUMNS = ("HPBL_surface",) + tuple(
    f"{variable}_{level.replace(' ', '')}"
    for variable, level in FEATURE_PAIRS[1:]
)
CENSUS_COLUMNS = (
    "target_operating_day_kst", "run_init_utc", "forecast_hour", "valid_time_utc",
    "cutoff_utc", "object_key", "object_size_bytes", "object_etag",
    "publication_last_modified_utc", "cutoff_margin_seconds", "idx_key",
    "idx_size_bytes", "idx_etag", "idx_last_modified_utc", "status",
    "source_archive", "archive_product", "retrieval_url_or_request_id",
    "official_metadata", "publication_evidence_type",
    "publication_evidence_reference", "family", "variable", "level",
    "grib_record_number", "forecast_descriptor", "range_start", "range_end",
    "range_bytes", "idx_relative_path", "idx_sha256",
)
RAW_MANIFEST_BASE_COLUMNS = (
    "source_archive", "archive_product", "target_operating_day_kst", "run_init_utc",
    "forecast_hour", "valid_time_utc", "retrieval_url_or_request_id", "retrieved_at",
    "raw_filename", "raw_sha256", "raw_size_bytes", "official_metadata",
    "publication_evidence_type", "publication_evidence_reference", "cutoff_utc",
    "cutoff_margin", "object_key", "object_etag", "object_size_bytes",
    "publication_last_modified_utc", "range_start", "range_end", "range_bytes",
    "family", "variable", "level", "idx_sha256", "http_status", "content_range",
    "request_range_start", "resumed_from_bytes", "status",
    "request_completion_evidence",
)
SITE_METADATA_COLUMNS = (
    "valid_time_utc", "target_operating_day_kst", "run_init_utc", "forecast_hour",
    "site_id", "group", "latitude", "longitude", "capacity_mw",
)
GROUP_METADATA_COLUMNS = (
    "valid_time_utc", "target_operating_day_kst", "run_init_utc", "forecast_hour",
    "group", "site_count", "capacity_mw",
)
OUTPUT_RELATIVE_PATHS = {
    "raw_parquet": "raw/RAW_RANGE_MANIFEST.parquet",
    "raw_csv": "raw/RAW_RANGE_MANIFEST.csv",
    "site": "decoded/site_values_target_free.parquet",
    "group": "decoded/group_values_target_free.parquet",
    "audit": "decoded/PHYSICAL_AND_COVERAGE_AUDIT.json",
    "lock": "decoded/DECODED_MATRIX_LOCK.json",
    "access": "raw/RAW_ACCESS_LEDGER.json",
    "manifest": "manifest_raw_v1.json",
}
V4_V2_CONTROL_RELATIVES = {
    "raw_cache_lock_v2": "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V2.json",
    "real_process_preflight_v2": "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2.json",
    "recovery_authorization_v2": "prereg/decode_recovery_authorization_v2.json",
    "independent_review_v2": "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V2.json",
    "independent_go_v2": "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V2.json",
}
V4_V2_CONTROL_IDENTITIES = {
    "raw_cache_lock_v2": {
        "path": V4_V2_CONTROL_RELATIVES["raw_cache_lock_v2"],
        "size_bytes": 7_545,
        "sha256": "ea235e8778090b8d488ad7c7aec530a50d969eec03095d25269e83b16b324a27",
    },
    "real_process_preflight_v2": {
        "path": V4_V2_CONTROL_RELATIVES["real_process_preflight_v2"],
        "size_bytes": 38_972,
        "sha256": "5c9b79845c03015f85647833b68603cbed187d3dc09f63c7d214a379b5940481",
    },
    "recovery_authorization_v2": {
        "path": V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"],
        "size_bytes": 17_653,
        "sha256": "2d47309b88d924668b522b8b65404c6c4cef25581e6b0f483b4f860cfad27eb7",
    },
    "independent_review_v2": {
        "path": V4_V2_CONTROL_RELATIVES["independent_review_v2"],
        "size_bytes": 8_903,
        "sha256": "b17635415b2b635ea771ce327833f166df76d40f02861daf2ed78c8a9f32b721",
    },
    "independent_go_v2": {
        "path": V4_V2_CONTROL_RELATIVES["independent_go_v2"],
        "size_bytes": 7_681,
        "sha256": "8b5f08cfb18cfaabfc9eddd5352d5bed1477f2ca18880f148b288d48a6fcdd76",
    },
}
V4_V2_SUPPORT_PATHS = {
    "recovery_runner": RECOVERY_RUNNER,
    "recovery_runner_test": RECOVERY_RUNNER_TEST,
    "recovery_sealer": RECOVERY_SEALER,
    "recovery_sealer_test": RECOVERY_SEALER_TEST,
    "recovery_module": RECOVERY_MODULE,
    "recovery_bootstrap": RECOVERY_BOOTSTRAP,
    "recovery_test": RECOVERY_TEST,
    "recovery_bootstrap_test": RECOVERY_BOOTSTRAP_TEST,
}
V4_V2_SUPPORT_SIZE_SHA256 = {
    "recovery_runner": (153_970, "892b9e0fb26cb0f9be19c77bcaa04c656e24d2404f42b2102596914e3d1c31d1"),
    "recovery_runner_test": (41_803, "f4c248ddcd04d574846f96b0ef504ccfaa41c4fc97dece2b5b77ffae21767bc3"),
    "recovery_sealer": (63_266, "5173cf3bbdd7fb0d35943c257b97c4984f36e2adac6452bbea1a155053bcdceb"),
    "recovery_sealer_test": (18_552, "78bf65181445f50aad50872ecbe31233fc5e22b4586c8706b68b586856eeceb1"),
    "recovery_module": (26_415, "6d1f05e36a1d60cdb73a6d8cbd6e3cc9bdfeac36871d75a6fc27a76952a8825c"),
    "recovery_bootstrap": (19_671, "09e350e19db0d278bbffa68dabac538d3cda4563c68afb5e00241d45e577e094"),
    "recovery_test": (10_234, "cf6d7a512f6cc997c1de7630c1d3e3b81e77a071af96a41c100ccf98b7ecea96"),
    "recovery_bootstrap_test": (14_575, "a918e192d6d9f172a112f6bea06613ab07524f96b83cab8505175890883928b3"),
}
V4_SUPERSEDED_V2_AUDITOR_IDENTITY = {
    "path": str(SUPERSEDED_V2_AUDITOR.resolve()),
    "size_bytes": 300_379,
    "sha256": "0b32c091098e28837c5b19d517683c16ff47cd3204cd5be074f0fdb222a76fb4",
}
V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY = {
    "path": str(SUPERSEDED_V2_AUDITOR_TEST.resolve()),
    "size_bytes": 133_607,
    "sha256": "0e2bd181cec9599ea0eebf9265d88e31aef3b34cc863365a449862bd957eb54e",
}
V4_ORIGINAL_V1_PROVENANCE = {
    "original_raw_authorization_v1": {
        "path": "prereg/raw_launch_authorization_v1.json",
        "size_bytes": 5_465,
        "sha256": "ff9a133e913a1b5a4a62028578b54eb4c5a43a067d6ff0da41357bfd2c889cb9",
    },
    "original_independent_prelaunch_audit_v1": {
        "path": "independent_redteam/TRACK_A_RAW_RUNNER_INDEPENDENT_PREFLIGHT_AUDIT_V1.json",
        "size_bytes": 11_439,
        "sha256": "4279935206cc245c0443d156bad114b96e6c7d7624dee670df9305a44b1009b5",
        "status": "PASS_TO_CREATE_FINAL_AUTH_ONLY",
    },
}
V4_TRANSACTION_IDENTITIES = {
    "plan": {
        "path": f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}__plan.json",
        "size_bytes": 7_247,
        "sha256": "0a093971f5b263b0fef9d37fdb4eaf0bb5116f00fe531ad8ec1f51c47f2ce2c7",
    },
    "commit": {
        "path": f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}__committed.json",
        "size_bytes": 1_997,
        "sha256": "cf091132ff812812d3095c2b51da41901e5ca3bc8f43b899721ea699fa781bcf",
    },
}
V4_COMPLETION_LOCK_IDENTITIES = {
    "decode_mutex_complete": {
        "path": f"decoded/recovery_history/{V4_RECOVERY_ATTEMPT_ID}__decode_mutex__complete.lock",
        "size_bytes": 206,
        "sha256": "0a121a5464a0437d7f7efcf972653f8610424858d96917b10410060c4409a6a2",
    },
    "raw_mutex_complete": {
        "path": f"decoded/recovery_history/{V4_RECOVERY_ATTEMPT_ID}__raw_mutex__complete.lock",
        "size_bytes": 206,
        "sha256": "0a121a5464a0437d7f7efcf972653f8610424858d96917b10410060c4409a6a2",
    },
}
V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY = {
    "path": f"decoded/recovery_history/{V4_RECOVERY_ATTEMPT_ID}__postcommit_input_audit.json",
    "size_bytes": 1_252,
    "sha256": "559b14d8effcede9259d2d0e9d520aee60fe0fa4ff810d0cdf39941bbc15332a",
}
V4_FINAL_PROGRESS_IDENTITY = {
    "path": f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__010368.json",
    "size_bytes": 221,
    "sha256": "d0d228ab900858919047c80c446454f627d23132f7037f3f557cc13cb3c2843f",
}
V4_CANONICAL_OUTPUT_IDENTITIES = {
    "raw_parquet": {"path": OUTPUT_RELATIVE_PATHS["raw_parquet"], "size_bytes": 3_812_043, "sha256": "11849682fd1c65ca20cfcb3002d3708beccf10ef862bb0f03d054c2c030790d5"},
    "raw_csv": {"path": OUTPUT_RELATIVE_PATHS["raw_csv"], "size_bytes": 13_527_324, "sha256": "52df75b66e17765f8a589ce979a694eaece9cfb013fe0d70de2ec63080090b1e"},
    "site": {"path": OUTPUT_RELATIVE_PATHS["site"], "size_bytes": 1_765_483, "sha256": "2da27c48cdce546cee5eccad8668491b7f50ac2edbd71b16340d1d0fd05b1e6b"},
    "group": {"path": OUTPUT_RELATIVE_PATHS["group"], "size_bytes": 317_008, "sha256": "3c017be4a6b445c8b7701556732630e96d16088081ccb8590c858bdba1aa909b"},
    "physical_audit": {"path": OUTPUT_RELATIVE_PATHS["audit"], "size_bytes": 13_802, "sha256": "6f4805e81ff04930e56a08ca452a68d1b336e7eea73b8d75fb89c975ae63049a"},
    "decoded_lock": {"path": OUTPUT_RELATIVE_PATHS["lock"], "size_bytes": 16_771, "sha256": "9c5ba145118bfbf6d5b9d328e244a099af19141b820ba02fef24475f34310959"},
    "access_ledger": {"path": OUTPUT_RELATIVE_PATHS["access"], "size_bytes": 18_127, "sha256": "4c2c0deea4e62ac4f88c2f9d7a3c269ca52da4e80b8378d3a8797abea15797a2"},
    "raw_manifest": {"path": OUTPUT_RELATIVE_PATHS["manifest"], "size_bytes": 15_985, "sha256": "3251a9a73e19a243c83529439a664d99b75d7d852d2b3b201cfe596701ea02c5"},
}
V4_AUTH_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "incident", "v2_false_reject_incident",
    "v3_path_mispublish_state", "superseded_v2_auditor",
    "superseded_v2_auditor_test", "superseded_v3_auditor",
    "superseded_v3_auditor_test", "superseded_v3_sealer",
    "superseded_v3_sealer_test", "v4_auditor", "v4_auditor_test",
    "v4_sealer", "v4_sealer_test",
    "v2_recovery_controls", "v2_recovery_support", "recovery_attempt_id",
    "recovery_transaction", "recovery_history", "recovery_progress",
    "canonical_outputs", "original_v1_provenance",
    "preaudit_zero_mutation_snapshot", "runtime_identity_sha256",
    "test_evidence", "required_command", "max_spawn_processes",
    "network_requests_allowed", "audit_files_written_allowed",
    "labels_read_allowed", "arrays_2024_read_allowed",
    "arrays_2025_read_allowed", "models_fit_allowed",
    "submission_csv_allowed", "full_offline_redecode_required",
    "stdout_only", "independent_review_required", "independent_go_required",
}
V4_REVIEW_KEYS = {
    "schema_version", "artifact_type", "status", "verdict", "created_utc",
    "audit_attempt_id", "authorization", "incident", "v2_false_reject_incident",
    "v3_path_mispublish_state", "superseded_v2_auditor",
    "superseded_v2_auditor_test", "superseded_v3_auditor",
    "superseded_v3_auditor_test", "superseded_v3_sealer",
    "superseded_v3_sealer_test", "v4_auditor", "v4_auditor_test",
    "v4_sealer", "v4_sealer_test",
    "v2_recovery_controls", "recovered_afterstate",
    "test_evidence_recheck", "independent_checks", "network_requests_allowed",
    "audit_files_written_allowed", "max_spawn_processes", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only", "auditor_execution_started",
    "independent_go_required",
}
V4_GO_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "authorization", "independent_review", "incident",
    "v2_false_reject_incident", "v3_path_mispublish_state",
    "superseded_v2_auditor", "superseded_v2_auditor_test",
    "superseded_v3_auditor", "superseded_v3_auditor_test",
    "superseded_v3_sealer", "superseded_v3_sealer_test",
    "v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test",
    "recovery_attempt_id", "recovered_afterstate",
    "required_command", "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only", "recovery_rerun_authorized",
}
V4_TEST_EVIDENCE_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc", "bound_identities",
    "source_compile", "pytest_isolation", "test_runs", "required_test_names",
    "required_test_names_present", "all_exit_codes_zero",
    "production_shape_regression", "real_seven_spawn_regression",
    "stdout_only_regression", "network_guard", "network_requests",
    "audit_files_written", "labels_read", "arrays_2024_read", "arrays_2025_read",
    "models_fit", "submission_csv_created",
}
V4_REQUIRED_TEST_NAMES = (
    "test_real_immutable_four_key_independent_prelaunch_identity_passes",
    "test_missing_independent_prelaunch_status_fails",
    "test_wrong_independent_prelaunch_status_fails",
    "test_null_or_non_string_independent_prelaunch_status_fails",
    "test_unknown_fifth_independent_prelaunch_identity_key_fails",
    "test_independent_prelaunch_path_size_or_sha_mismatch_fails",
    "test_full_recovered_production_shape_reaches_transaction_raw_decoded_and_offline_replay",
    "test_frozen_v2_false_reject_incident_and_source_identities_are_exact",
    "test_v3_path_mispublish_incident_and_frozen_chain_are_exact",
    "test_real_v3_authority_and_full_synthetic_core_proofs_are_compositional",
    "test_canonical_v3_go_presence_fails_before_data",
    "test_misplaced_v3_go_missing_identity_or_link_tamper_fails_before_data",
    "test_v3_chain_chronology_or_crosslink_tamper_fails_before_data",
    "test_unbound_matching_control_namespace_entry_fails_before_data",
    "test_v2_v3_v4_report_candidate_fails_before_data",
    "test_v4_authority_schema_path_status_crosslink_or_command_tamper_fails",
    "test_v4_postaudit_rechecks_mispublication_namespace_and_afterstate",
    "test_v4_auditor_is_stdout_only_and_writes_zero_files",
    "test_network_target_label_model_and_submission_routes_remain_absent",
    "test_v4_worker_is_importable_in_exactly_seven_spawn_processes",
    "test_v4_authority_and_go_precede_any_parquet_or_data_read",
)
V4_TEST_RUN_KEYS = {
    "v4_auditor_tests", "v4_sealer_tests", "frozen_v3_auditor_tests",
    "frozen_v3_sealer_tests", "frozen_v2_auditor_tests",
}
V4_TEST_RUN_RECORD_KEYS = {
    "command", "exit_code", "summary", "stdout_sha256", "stderr_sha256",
    "network_guard_installed", "cacheprovider_disabled",
}
V4_REVIEW_TEST_RECHECK_KEYS = {
    "authorization_test_evidence_canonical_sha256", "bound_identities_rehashed",
    "source_compile_exact", "required_test_names_exact", "all_exit_codes_zero",
    "production_shape_regression_exact", "real_seven_spawn_regression_exact",
    "stdout_only_zero_write_exact", "network_zero_target_free_exact",
}
V4_REVIEW_CHECKS = {
    "false_reject_incident_exact",
    "v3_path_mispublish_incident_exact",
    "v3_authorization_review_and_rejected_go_exact",
    "canonical_v3_go_absent_and_forbidden",
    "postrun_audit_control_namespace_exact",
    "frozen_v2_auditor_and_test_exact",
    "frozen_v3_auditor_test_sealer_identities_exact",
    "v4_code_test_sealer_identities_exact",
    "minimal_diff_single_role_four_key_fix_exact",
    "negative_status_unknown_key_and_identity_tamper_tests_pass",
    "v2_recovery_control_and_support_chain_exact",
    "original_v1_four_key_provenance_exact",
    "authorization_and_go_precede_any_parquet_or_data_read",
    "production_shape_regression_reaches_full_v2_audit_path",
    "real_seven_spawn_v4_worker_importability_pass",
    "recovered_afterstate_exact_and_closed",
    "full_offline_raw_to_decoded_replay_retained",
    "frozen_v2_auditor_suite_green",
    "executing_v4_vs_frozen_v2_and_v3_role_distinction_exact",
    "stdout_only_zero_write_contract_retained",
    "network_zero_target_free_no_model_submission_scope_retained",
}
FROZEN_V3_AUTH_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "incident", "superseded_v2_auditor",
    "superseded_v2_auditor_test", "v3_auditor", "v3_auditor_test",
    "v3_sealer", "v3_sealer_test", "v2_recovery_controls",
    "v2_recovery_support", "recovery_attempt_id", "recovery_transaction",
    "recovery_history", "recovery_progress", "canonical_outputs",
    "original_v1_provenance", "preaudit_zero_mutation_snapshot",
    "runtime_identity_sha256", "test_evidence", "required_command",
    "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only",
    "independent_review_required", "independent_go_required",
}
FROZEN_V3_REVIEW_KEYS = {
    "schema_version", "artifact_type", "status", "verdict", "created_utc",
    "audit_attempt_id", "authorization", "incident",
    "superseded_v2_auditor", "superseded_v2_auditor_test", "v3_auditor",
    "v3_auditor_test", "v3_sealer", "v3_sealer_test",
    "v2_recovery_controls", "recovered_afterstate", "test_evidence_recheck",
    "independent_checks", "network_requests_allowed",
    "audit_files_written_allowed", "max_spawn_processes", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed", "models_fit_allowed",
    "submission_csv_allowed", "full_offline_redecode_required", "stdout_only",
    "auditor_execution_started", "independent_go_required",
}
FROZEN_V3_GO_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "authorization", "independent_review", "incident",
    "superseded_v2_auditor", "superseded_v2_auditor_test", "v3_auditor",
    "v3_auditor_test", "v3_sealer", "v3_sealer_test", "recovery_attempt_id",
    "recovered_afterstate", "required_command", "max_spawn_processes",
    "network_requests_allowed", "audit_files_written_allowed",
    "labels_read_allowed", "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed", "full_offline_redecode_required",
    "stdout_only", "recovery_rerun_authorized",
}
V4_PATH_MISPUBLISH_STATE_KEYS = {
    "incident", "authorization", "independent_review", "misplaced_go",
    "canonical_go", "review_predates_misplaced_go",
    "misplaced_go_payload_canonical_sha256",
    "misplaced_go_payload_exact_v3_schema_and_crossbindings",
    "misplaced_go_authoritative", "v3_auditor_execution_authorized",
}
V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256 = (
    "311642243f2261bf8f50a07ac430f365c2e49a7971f58f478cce1cc026d91ce1"
)
V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256 = (
    "8e434429e6768e14dbfde2c2592609bcd8167e991fa5eaea65b5ba40e7d991cb"
)
V4_RECOVERED_AFTERSTATE_CANONICAL_BYTES = 24_234
_V4_CANONICAL_MAIN_ACTIVE = False
V5_RECOVERED_AFTERSTATE_COMMITMENT = {
    "canonical_sha256": V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
    "canonical_size_bytes": V4_RECOVERED_AFTERSTATE_CANONICAL_BYTES,
    "recovery_progress_file_count": 104,
    "recovery_progress_inventory_count": 104,
    "recovery_progress_inventory_canonical_sha256": (
        V4_PROGRESS_INVENTORY_CANONICAL_SHA256
    ),
    "canonical_output_count": 8,
    "transaction_total_recursive_files": 4,
    "recovery_history_file_count": 3,
    "raw_active_lock_present": False,
    "decoded_active_lock_present": False,
}
V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_BYTES = 470
V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256 = (
    "34be9415ded25118af85185f9bc6b15ca418720320d1e5c257d387e247858698"
)
V5_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256 = (
    "2c8e7d0be13c61c6b9fc77bc528e82478a4a759b3fc5e5488995e90be81f20e7"
)
V5_AUTH_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "incident", "incident_correction",
    "v4_authorization_base", "v4_review_transport_truncation_state",
    "superseded_v4_auditor", "superseded_v4_auditor_test",
    "superseded_v4_sealer", "superseded_v4_sealer_test",
    "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
    "recovery_attempt_id", "recovered_afterstate_commitment",
    "preaudit_zero_mutation_snapshot", "runtime_identity_sha256",
    "test_evidence", "required_command", "max_spawn_processes",
    "network_requests_allowed", "audit_files_written_allowed",
    "labels_read_allowed", "arrays_2024_read_allowed",
    "arrays_2025_read_allowed", "models_fit_allowed",
    "submission_csv_allowed", "full_offline_redecode_required",
    "stdout_only", "independent_review_required", "independent_go_required",
}
V5_REVIEW_KEYS = {
    "schema_version", "artifact_type", "status", "verdict", "created_utc",
    "audit_attempt_id", "authorization", "incident", "incident_correction",
    "v4_authorization_base", "v4_review_transport_truncation_state",
    "superseded_v4_auditor", "superseded_v4_auditor_test",
    "superseded_v4_sealer", "superseded_v4_sealer_test",
    "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
    "recovered_afterstate_commitment", "test_evidence_recheck",
    "independent_checks", "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only", "auditor_execution_started",
    "independent_go_required",
}
V5_GO_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "authorization", "independent_review", "incident",
    "incident_correction", "v4_authorization_base",
    "v4_review_transport_truncation_state", "superseded_v4_auditor",
    "superseded_v4_auditor_test", "superseded_v4_sealer",
    "superseded_v4_sealer_test", "v5_auditor", "v5_auditor_test",
    "v5_sealer", "v5_sealer_test", "recovery_attempt_id",
    "recovered_afterstate_commitment", "required_command", "max_spawn_processes",
    "network_requests_allowed", "audit_files_written_allowed",
    "labels_read_allowed", "arrays_2024_read_allowed",
    "arrays_2025_read_allowed", "models_fit_allowed",
    "submission_csv_allowed", "full_offline_redecode_required", "stdout_only",
    "recovery_rerun_authorized",
}
V5_AUTHORIZATION_BASE_KEYS = {
    "identity", "schema_version", "status", "audit_attempt_id",
    "recovery_attempt_id", "payload_canonical_sha256",
    "test_evidence_canonical_sha256", "recovered_afterstate_canonical_sha256",
}
V5_TRUNCATION_STATE_KEYS = {
    "authorization", "independent_review", "canonical_go",
    "actual_review_payload_canonical_sha256",
    "intended_review_payload_canonical_sha256",
    "actual_recovered_afterstate_canonical_sha256",
    "expected_recovered_afterstate_canonical_sha256",
    "actual_progress_inventory_count", "expected_progress_inventory_count",
    "intended_review_pretty_serialization", "v4_auditor_execution_started",
    "v4_auditor_execution_authorized",
}
V5_ZERO_SNAPSHOT_KEYS = {
    "incident_postfailure_state_canonical_sha256",
    "recovered_afterstate_commitment_canonical_sha256", "active_locks",
    "v5_control_temporary_files", "postrun_audit_output_files_present",
    "network_requests", "audit_files_written", "labels_read",
    "arrays_2024_read", "arrays_2025_read", "models_fit",
    "submission_csv_created",
}
V5_TEST_EVIDENCE_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc", "bound_identities",
    "source_compile", "pytest_isolation", "test_runs", "required_test_names",
    "required_test_names_present", "all_exit_codes_zero",
    "production_shape_regression", "real_seven_spawn_regression",
    "stdout_only_regression", "network_guard", "network_requests",
    "audit_files_written", "labels_read", "arrays_2024_read", "arrays_2025_read",
    "models_fit", "submission_csv_created",
}
V5_TEST_RUN_KEYS = {
    "v5_auditor_tests", "v5_sealer_tests", "frozen_v4_auditor_tests",
    "frozen_v4_sealer_tests",
}
V5_REQUIRED_TEST_NAMES = (
    "test_real_immutable_v4_authorization_base_passes",
    "test_v4_authorization_base_path_size_sha_schema_status_or_digest_tamper_fails",
    "test_v4_review_transport_truncation_incident_and_frozen_chain_are_exact",
    "test_v4_review_transport_truncation_correction_tamper_fails",
    "test_primary_transport_incident_without_correction_fails",
    "test_v4_review_raw_duplicate_and_transport_marker_are_detected_exactly",
    "test_v4_review_missing_progress_rows_and_wrong_effective_sha_are_exact",
    "test_intended_v4_review_reconstruction_matches_frozen_38918_d0e1",
    "test_canonical_v4_go_presence_fails_before_data",
    "test_malformed_v4_review_identity_link_or_payload_tamper_fails_before_data",
    "test_v4_chain_chronology_or_crosslink_tamper_fails_before_data",
    "test_compact_recovered_afterstate_commitment_matches_v4_authorization",
    "test_compact_commitment_field_or_inventory_digest_tamper_fails",
    "test_full_recovered_afterstate_is_not_embedded_in_v5_authority_controls",
    "test_real_v5_authority_and_full_synthetic_core_proofs_are_compositional",
    "test_full_recovered_production_shape_reaches_transaction_raw_decoded_and_offline_replay",
    "test_unbound_matching_control_namespace_entry_fails_before_data",
    "test_v2_v3_v4_v5_report_candidate_fails_before_data",
    "test_v5_authority_schema_path_status_crosslink_or_command_tamper_fails",
    "test_v5_postaudit_rechecks_v4_truncation_namespace_and_afterstate_commitment",
    "test_v5_auditor_is_stdout_only_and_writes_zero_files",
    "test_network_target_label_model_and_submission_routes_remain_absent",
    "test_v5_worker_is_importable_in_exactly_seven_spawn_processes",
    "test_v5_authority_and_go_precede_any_parquet_or_data_read",
    "test_v5_review_and_go_never_embed_full_progress_inventory",
    "test_duplicate_json_key_rejection_is_not_bypassed_by_standard_last_wins_parse",
    "test_incident_or_v4_review_cannot_authorize_v4_or_v5_execution",
)
V5_REVIEW_CHECKS = {
    "v4_review_transport_truncation_incident_and_correction_exact",
    "exact_one_inventory_digest_override_applied_before_semantic_use",
    "v4_authorization_base_exact_and_revalidated",
    "v4_malformed_review_identity_duplicate_and_marker_exact",
    "v4_intended_review_reconstruction_exact",
    "canonical_v4_go_absent_and_forbidden",
    "postrun_audit_control_namespace_exact",
    "frozen_v4_auditor_test_sealer_identities_exact",
    "v5_code_test_sealer_identities_exact",
    "v4_authority_inheritance_without_review_salvage_exact",
    "v2_v3_authority_transitively_revalidated",
    "original_v1_four_key_provenance_transitively_exact",
    "authorization_and_go_precede_any_parquet_or_data_read",
    "compact_recovered_afterstate_commitment_exact",
    "full_recovered_afterstate_reconstructed_in_memory",
    "recovered_afterstate_metadata_only_closure_exact",
    "production_shape_regression_reaches_full_v2_audit_path",
    "real_seven_spawn_v5_worker_importability_pass",
    "full_offline_raw_to_decoded_replay_retained",
    "v4_data_and_replay_logic_ast_identical",
    "frozen_v4_auditor_and_sealer_suites_green",
    "executing_v5_vs_frozen_v2_v3_v4_role_distinction_exact",
    "stdout_only_zero_write_contract_retained",
    "network_zero_target_free_no_model_submission_scope_retained",
    "full_104_inventory_not_duplicated_in_v5_controls",
    "negative_truncation_commitment_namespace_tamper_tests_pass",
}
V6_AUTH_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "incident", "v5_authority_base",
    "historical_failed_head_identity_contract", "superseded_v5_auditor",
    "superseded_v5_auditor_test", "superseded_v5_sealer",
    "superseded_v5_sealer_test", "v6_auditor", "v6_auditor_test",
    "v6_sealer", "v6_sealer_test", "recovery_attempt_id",
    "recovered_afterstate_commitment", "preaudit_zero_mutation_snapshot",
    "runtime_identity_sha256", "test_evidence", "required_command",
    "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only",
    "independent_review_required", "independent_go_required",
}
V6_REVIEW_KEYS = {
    "schema_version", "artifact_type", "status", "verdict", "created_utc",
    "audit_attempt_id", "authorization", "incident", "v5_authority_base",
    "historical_failed_head_identity_contract", "superseded_v5_auditor",
    "superseded_v5_auditor_test", "superseded_v5_sealer",
    "superseded_v5_sealer_test", "v6_auditor", "v6_auditor_test",
    "v6_sealer", "v6_sealer_test", "recovered_afterstate_commitment",
    "test_evidence_recheck", "independent_checks", "max_spawn_processes",
    "network_requests_allowed", "audit_files_written_allowed",
    "labels_read_allowed", "arrays_2024_read_allowed",
    "arrays_2025_read_allowed", "models_fit_allowed",
    "submission_csv_allowed", "full_offline_redecode_required", "stdout_only",
    "auditor_execution_started", "independent_go_required",
}
V6_GO_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "authorization", "independent_review", "incident",
    "v5_authority_base", "historical_failed_head_identity_contract",
    "superseded_v5_auditor", "superseded_v5_auditor_test",
    "superseded_v5_sealer", "superseded_v5_sealer_test", "v6_auditor",
    "v6_auditor_test", "v6_sealer", "v6_sealer_test",
    "recovery_attempt_id", "recovered_afterstate_commitment",
    "required_command", "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only", "recovery_rerun_authorized",
}
V6_AUTHORITY_BASE_KEYS = {
    "authorization", "independent_review", "independent_go",
    "audit_attempt_id", "recovery_attempt_id", "authorization_status",
    "review_status", "go_status", "recovered_afterstate_canonical_sha256",
}
V6_HISTORICAL_CONTRACT_KEYS = {
    "incident", "historical_incident_contract", "validation_mode",
    "validation_moved_before_parquet_and_data_reads",
    "current_role_path_dereference_forbidden",
}
V6_HISTORICAL_COMMITMENT_KEYS = {
    "failed_head_identities", "verified_zero_state_after_failure",
    "absent_control_paths",
}
V6_CANONICAL_COMMITMENT_KEYS = {"canonical_size_bytes", "canonical_sha256"}
V6_ZERO_SNAPSHOT_KEYS = {
    "incident_postfailure_state_canonical_sha256",
    "recovered_afterstate_commitment_canonical_sha256", "active_locks",
    "v6_control_temporary_files", "postrun_audit_output_files_present",
    "network_requests", "audit_files_written", "labels_read",
    "arrays_2024_read", "arrays_2025_read", "models_fit",
    "submission_csv_created",
}
V6_TEST_EVIDENCE_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc", "bound_identities",
    "source_compile", "pytest_isolation", "test_runs", "required_test_names",
    "required_test_names_present", "all_exit_codes_zero",
    "production_shape_regression", "real_seven_spawn_regression",
    "stdout_only_regression", "network_guard", "network_requests",
    "audit_files_written", "labels_read", "arrays_2024_read", "arrays_2025_read",
    "models_fit", "submission_csv_created",
}
V6_TEST_RUN_KEYS = {
    "v6_auditor_tests", "v6_sealer_tests", "frozen_v5_auditor_tests",
    "frozen_v5_sealer_tests",
}
V6_REQUIRED_TEST_NAMES = (
    "test_real_immutable_v5_authority_base_passes",
    "test_v5_authority_base_path_size_sha_schema_status_or_digest_tamper_fails",
    "test_v5_historical_identity_false_reject_incident_is_exact",
    "test_direct_file_failure_incident_9e74_historical_state_is_exact",
    "test_frozen_v5_reproduces_exact_recovery_runner_false_reject",
    "test_v6_real_9e74_historical_validator_passes",
    "test_historical_failed_head_path_size_or_sha_tamper_fails",
    "test_historical_v1_absent_control_path_tamper_fails",
    "test_current_v2_role_or_control_substitution_fails",
    "test_historical_predata_gate_precedes_schema_parquet_data_and_replay",
    "test_late_reuse_revalidates_historical_chain",
    "test_v5_production_data_and_replay_core_ast_is_identical",
    "test_compact_recovered_afterstate_commitment_matches_v5_authority",
    "test_compact_commitment_field_or_inventory_digest_tamper_fails",
    "test_full_recovered_afterstate_is_not_embedded_in_v6_authority_controls",
    "test_real_v6_authority_and_full_synthetic_core_proofs_are_compositional",
    "test_full_recovered_production_shape_reaches_transaction_raw_decoded_and_offline_replay",
    "test_unbound_matching_control_namespace_entry_fails_before_data",
    "test_v2_v3_v4_v5_v6_report_candidate_fails_before_data",
    "test_v6_authority_schema_path_status_crosslink_or_command_tamper_fails",
    "test_v6_postaudit_rechecks_historical_chain_namespace_and_afterstate_commitment",
    "test_v6_auditor_is_stdout_only_and_writes_zero_files",
    "test_network_target_label_model_and_submission_routes_remain_absent",
    "test_v6_worker_is_importable_in_exactly_seven_spawn_processes",
    "test_v6_authority_and_historical_gate_precede_any_parquet_or_data_read",
    "test_v6_review_and_go_never_embed_full_progress_inventory",
    "test_duplicate_json_key_rejection_is_not_bypassed_by_standard_last_wins_parse",
    "test_incident_cannot_authorize_v5_or_v6_execution",
)
V6_REVIEW_CHECKS = {
    "v5_historical_identity_false_reject_incident_exact",
    "direct_file_failure_incident_9e74_exact",
    "exact_seven_historical_failed_heads_frozen",
    "exact_five_v1_absent_control_paths_frozen",
    "historical_and_current_v2_roles_distinct",
    "current_role_or_control_substitution_forbidden",
    "historical_validation_precedes_schema_parquet_data_and_replay",
    "frozen_v5_authority_chain_exact_and_revalidated",
    "frozen_v5_auditor_test_sealer_identities_exact",
    "v6_code_test_sealer_identities_exact",
    "v4_and_earlier_authority_transitively_revalidated",
    "original_v1_four_key_provenance_transitively_exact",
    "compact_recovered_afterstate_commitment_exact",
    "full_recovered_afterstate_reconstructed_in_memory",
    "recovered_afterstate_metadata_only_closure_exact",
    "production_shape_regression_reaches_full_v2_audit_path",
    "real_seven_spawn_v6_worker_importability_pass",
    "full_offline_raw_to_decoded_replay_retained",
    "v5_production_data_and_replay_core_ast_identical",
    "frozen_v5_auditor_and_sealer_suites_green",
    "executing_v6_vs_frozen_v2_v3_v4_v5_role_distinction_exact",
    "stdout_only_zero_write_contract_retained",
    "network_zero_target_free_no_model_submission_scope_retained",
    "full_104_inventory_not_duplicated_in_v6_controls",
    "negative_historical_head_zero_state_namespace_tamper_tests_pass",
    "v5_false_reject_reproduced_and_v6_positive_passes",
}
V7_AUTH_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "incident", "v5_authority_base",
    "historical_failed_head_identity_contract", "superseded_v6_auditor",
    "superseded_v6_auditor_test", "superseded_v6_sealer",
    "superseded_v6_sealer_test", "v7_auditor", "v7_auditor_test",
    "v7_sealer", "v7_sealer_test", "recovery_attempt_id",
    "recovered_afterstate_commitment", "preaudit_zero_mutation_snapshot",
    "runtime_identity_sha256", "test_evidence", "required_command",
    "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only",
    "independent_review_required", "independent_go_required",
}
V7_REVIEW_KEYS = {
    "schema_version", "artifact_type", "status", "verdict", "created_utc",
    "audit_attempt_id", "authorization", "incident", "v5_authority_base",
    "historical_failed_head_identity_contract", "superseded_v6_auditor",
    "superseded_v6_auditor_test", "superseded_v6_sealer",
    "superseded_v6_sealer_test", "v7_auditor", "v7_auditor_test",
    "v7_sealer", "v7_sealer_test", "recovered_afterstate_commitment",
    "test_evidence_recheck", "independent_checks", "max_spawn_processes",
    "network_requests_allowed", "audit_files_written_allowed",
    "labels_read_allowed", "arrays_2024_read_allowed",
    "arrays_2025_read_allowed", "models_fit_allowed",
    "submission_csv_allowed", "full_offline_redecode_required", "stdout_only",
    "auditor_execution_started", "independent_go_required",
}
V7_GO_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc",
    "audit_attempt_id", "authorization", "independent_review", "incident",
    "v5_authority_base", "historical_failed_head_identity_contract",
    "superseded_v6_auditor", "superseded_v6_auditor_test",
    "superseded_v6_sealer", "superseded_v6_sealer_test", "v7_auditor",
    "v7_auditor_test", "v7_sealer", "v7_sealer_test",
    "recovery_attempt_id", "recovered_afterstate_commitment",
    "required_command", "max_spawn_processes", "network_requests_allowed",
    "audit_files_written_allowed", "labels_read_allowed",
    "arrays_2024_read_allowed", "arrays_2025_read_allowed",
    "models_fit_allowed", "submission_csv_allowed",
    "full_offline_redecode_required", "stdout_only", "recovery_rerun_authorized",
}
V7_ZERO_SNAPSHOT_KEYS = {
    "incident_postfailure_state_canonical_sha256",
    "recovered_afterstate_commitment_canonical_sha256", "active_locks",
    "v7_control_temporary_files", "postrun_audit_output_files_present",
    "network_requests", "audit_files_written", "labels_read",
    "arrays_2024_read", "arrays_2025_read", "models_fit",
    "submission_csv_created",
}
V7_TEST_EVIDENCE_KEYS = {
    "schema_version", "artifact_type", "status", "created_utc", "bound_identities",
    "source_compile", "pytest_isolation", "test_runs", "required_test_names",
    "required_test_names_present", "all_exit_codes_zero",
    "production_shape_regression", "real_seven_spawn_regression",
    "stdout_only_regression", "network_guard", "network_requests",
    "audit_files_written", "labels_read", "arrays_2024_read", "arrays_2025_read",
    "models_fit", "submission_csv_created",
}
V7_TEST_RUN_KEYS = {
    "v7_auditor_tests", "v7_sealer_tests", "frozen_v6_auditor_tests",
    "frozen_v6_sealer_tests",
}
V7_REQUIRED_TEST_NAMES = (
    "test_real_immutable_v5_authority_base_passes",
    "test_v5_authority_base_path_size_sha_schema_status_or_digest_tamper_fails",
    "test_v5_historical_identity_false_reject_incident_is_exact",
    "test_direct_file_failure_incident_9e74_historical_state_is_exact",
    "test_v6_auth_sealer_timestamp_precision_false_reject_incident_is_exact",
    "test_frozen_v5_reproduces_exact_recovery_runner_false_reject",
    "test_frozen_v6_reproduces_exact_sealer_timestamp_precision_false_reject",
    "test_v7_rfc3339_100ns_parser_accepts_zero_through_seven_fractional_digits",
    "test_v7_rfc3339_100ns_parser_preserves_exact_seven_digit_tick_ordering",
    "test_v7_rfc3339_100ns_parser_rejects_eight_digits_offsets_non_utc_and_malformed",
    "test_v7_actual_9e74_collect_build_and_self_validator_passes",
    "test_v7_actual_722f_incident_and_raw_timestamp_bindings_pass",
    "test_timestamp_raw_value_or_chronology_substitution_fails",
    "test_historical_failed_head_path_size_or_sha_tamper_fails",
    "test_historical_v1_absent_control_path_tamper_fails",
    "test_current_v2_role_or_control_substitution_fails",
    "test_timestamp_and_historical_predata_gate_precedes_schema_parquet_data_and_replay",
    "test_late_reuse_revalidates_timestamp_incident_and_historical_chain",
    "test_v6_and_v5_production_data_and_replay_core_ast_is_identical",
    "test_compact_recovered_afterstate_commitment_matches_v5_authority",
    "test_compact_commitment_field_or_inventory_digest_tamper_fails",
    "test_full_recovered_afterstate_is_not_embedded_in_v7_authority_controls",
    "test_real_v7_authority_and_full_synthetic_core_proofs_are_compositional",
    "test_full_recovered_production_shape_reaches_transaction_raw_decoded_and_offline_replay",
    "test_unbound_matching_control_namespace_entry_fails_before_data",
    "test_v2_v3_v4_v5_v6_v7_report_candidate_fails_before_data",
    "test_v7_authority_schema_path_status_crosslink_or_command_tamper_fails",
    "test_v7_postaudit_rechecks_timestamp_incident_historical_chain_namespace_and_afterstate_commitment",
    "test_v7_auditor_is_stdout_only_and_writes_zero_files",
    "test_network_target_label_model_and_submission_routes_remain_absent",
    "test_v7_worker_is_importable_in_exactly_seven_spawn_processes",
    "test_v7_review_and_go_never_embed_full_progress_inventory",
    "test_duplicate_json_key_rejection_is_not_bypassed_by_standard_last_wins_parse",
    "test_incident_cannot_authorize_v6_or_v7_execution",
)
V7_REVIEW_CHECKS = {
    "v5_historical_identity_false_reject_incident_exact",
    "direct_file_failure_incident_9e74_exact",
    "exact_seven_historical_failed_heads_frozen",
    "exact_five_v1_absent_control_paths_frozen",
    "historical_and_current_v2_roles_distinct",
    "current_role_or_control_substitution_forbidden",
    "timestamp_and_historical_validation_precedes_schema_parquet_data_and_replay",
    "frozen_v5_authority_chain_exact_and_revalidated",
    "frozen_v6_auditor_test_sealer_identities_exact",
    "v7_code_test_sealer_identities_exact",
    "v4_and_earlier_authority_transitively_revalidated",
    "original_v1_four_key_provenance_transitively_exact",
    "compact_recovered_afterstate_commitment_exact",
    "full_recovered_afterstate_reconstructed_in_memory",
    "recovered_afterstate_metadata_only_closure_exact",
    "production_shape_regression_reaches_full_v2_audit_path",
    "real_seven_spawn_v7_worker_importability_pass",
    "full_offline_raw_to_decoded_replay_retained",
    "v6_and_v5_production_data_and_replay_core_ast_identical",
    "frozen_v6_auditor_and_sealer_suites_green",
    "executing_v7_vs_frozen_v2_v3_v4_v5_v6_role_distinction_exact",
    "stdout_only_zero_write_contract_retained",
    "network_zero_target_free_no_model_submission_scope_retained",
    "full_104_inventory_not_duplicated_in_v7_controls",
    "negative_timestamp_historical_head_zero_state_namespace_tamper_tests_pass",
    "v5_false_reject_reproduced_and_v7_historical_positive_passes",
    "v6_auth_sealer_timestamp_precision_false_reject_incident_exact",
    "timestamp_incident_frozen_v6_failure_boundary_exact",
    "historical_raw_seven_digit_timestamp_and_chronology_exact",
    "rfc3339_100ns_zero_through_seven_fractional_digits_exact",
    "rfc3339_100ns_eight_digit_offset_non_utc_and_malformed_rejected",
    "frozen_v6_timestamp_false_reject_reproduced_and_v7_actual_root_positive_passes",
}
_V7_CANONICAL_MAIN_ACTIVE = False
_V6_CANONICAL_MAIN_ACTIVE = False
_V5_CANONICAL_MAIN_ACTIVE = False
V4_PYTEST_NETWORK_GUARD_SOURCE = """
import socket
import sys

def _deny_network(*_args, **_kwargs):
    raise RuntimeError("network forbidden in immutable V4 postrun-auditor test subprocess")

def _deny_network_audit(event, _args):
    if event in {
        "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
        "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname",
        "socket.getnameinfo", "socket.sendmsg", "socket.sendto",
    }:
        raise RuntimeError("network audit event forbidden in immutable V4 postrun-auditor test subprocess")

sys.addaudithook(_deny_network_audit)
socket.create_connection = _deny_network
socket.getaddrinfo = _deny_network
socket.gethostbyaddr = _deny_network
socket.gethostbyname = _deny_network
socket.getnameinfo = _deny_network
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
""".strip()
V5_PYTEST_NETWORK_GUARD_SOURCE = """
import socket
import sys

def _deny_network(*_args, **_kwargs):
    raise RuntimeError("network forbidden in immutable V5 postrun-auditor test subprocess")

def _deny_network_audit(event, _args):
    if event in {
        "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
        "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname",
        "socket.getnameinfo", "socket.sendmsg", "socket.sendto",
    }:
        raise RuntimeError("network audit event forbidden in immutable V5 postrun-auditor test subprocess")

sys.addaudithook(_deny_network_audit)
socket.create_connection = _deny_network
socket.getaddrinfo = _deny_network
socket.gethostbyaddr = _deny_network
socket.gethostbyname = _deny_network
socket.getnameinfo = _deny_network
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
""".strip()
V6_PYTEST_NETWORK_GUARD_SOURCE = """
import socket
import sys

def _deny_network(*_args, **_kwargs):
    raise RuntimeError("network forbidden in immutable V6 postrun-auditor test subprocess")

def _deny_network_audit(event, _args):
    if event in {
        "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
        "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname",
        "socket.getnameinfo", "socket.sendmsg", "socket.sendto",
    }:
        raise RuntimeError("network audit event forbidden in immutable V6 postrun-auditor test subprocess")

sys.addaudithook(_deny_network_audit)
socket.create_connection = _deny_network
socket.getaddrinfo = _deny_network
socket.gethostbyaddr = _deny_network
socket.gethostbyname = _deny_network
socket.getnameinfo = _deny_network
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
""".strip()
EVENT_NAME = re.compile(
    r"^(?P<event_id>[0-9a-f]{32})__attempt_(?P<attempt>[0-9]{3,5})_"
    r"(?P<kind>start|complete|error)\.json$"
)
PROGRESS_NAME = re.compile(
    r"^(?P<kind>raw_ranges|decoded_messages)__(?P<attempt>.+)__"
    r"(?P<count>[0-9]{6})\.json$"
)
DECODE_MAX_WORKERS = 7
DECODE_ABSOLUTE_TOLERANCE = 1e-12
_DECODE_WORKER_RUNTIME_IDENTITY: dict[str, Any] | None = None


@dataclass(frozen=True)
class GroupContract:
    name: str
    site_count: int
    capacity_mw: float


@dataclass(frozen=True)
class AuditContract:
    range_rows: int
    range_bytes: int
    object_rows: int
    site_count: int
    groups: tuple[GroupContract, ...]
    raw_attempt_cap: int = 15_200
    census_worst_case_attempts: int = 4_800
    combined_attempt_cap: int = 20_000
    census_successful_requests: int = 1_200
    operating_years: tuple[int, ...] | None = (2022, 2023)
    operating_days_of_month: tuple[int, ...] | None = (5, 20)
    forecast_hours: tuple[int, ...] | None = tuple(range(28, 52))

    @property
    def site_rows(self) -> int:
        return self.object_rows * self.site_count

    @property
    def group_rows(self) -> int:
        return self.object_rows * len(self.groups)


PRODUCTION_CONTRACT = AuditContract(
    range_rows=10_368,
    range_bytes=10_296_112_890,
    object_rows=1_152,
    site_count=17,
    groups=(
        GroupContract("kpx_group_1", 6, 21.6),
        GroupContract("kpx_group_2", 6, 21.6),
        GroupContract("kpx_group_3", 5, 21.0),
    ),
)


def recovery_progress_counts(contract: AuditContract) -> list[int]:
    counts = list(range(100, contract.range_rows, 100))
    if contract != PRODUCTION_CONTRACT and contract.range_rows < 100:
        counts.append(max(1, contract.range_rows - 4))
    counts.append(contract.range_rows)
    return sorted(set(counts))


class AuditFailure(RuntimeError):
    """A completed-looking producer tree violates a frozen invariant."""


class AuditNotReady(RuntimeError):
    """The producer tree cannot be audited because it is still active."""

    def __init__(self, message: str, *, active_lock: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.active_lock = dict(active_lock)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_segment(path: Path, start: int, length: int) -> str:
    _require(start >= 0 and length >= 0, f"invalid segment for {path}")
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining:
            block = stream.read(min(8 << 20, remaining))
            _require(bool(block), f"segment ends beyond file: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _authoritative_v1_support_external() -> dict[str, dict[str, Any]]:
    """Convert correction-relative frozen records to rehashed auth path form."""

    result: dict[str, dict[str, Any]] = {}
    for role, frozen in RECOVERY_V1_SUPPORT_IDENTITIES.items():
        lexical = REPO / str(frozen["path"])
        require_no_symlink_chain(
            lexical, REPO, label=f"authoritative V1 support {role}"
        )
        _require(
            lexical.is_file() and not _linklike(lexical),
            f"authoritative V1 support is absent/link: {role}",
        )
        path = lexical.resolve()
        observed = identity(path)
        _require(
            observed["size_bytes"] == frozen["size_bytes"]
            and observed["sha256"] == frozen["sha256"],
            f"authoritative V1 support frozen identity mismatch: {role}",
        )
        result[role] = observed
    return result


def _validate_v2_support_path_forms(
    authorization_support: Any,
    correction_support: Any,
) -> dict[str, dict[str, Any]]:
    """Keep absolute execution records distinct from relative history records."""

    absolute = _authoritative_v1_support_external()
    _require(
        authorization_support == absolute,
        "V2 authorization/plan support must use absolute external identities",
    )
    _require(
        correction_support == RECOVERY_V1_SUPPORT_IDENTITIES,
        "V1 identity correction support must remain repo-relative",
    )
    return absolute


def load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser detail is immaterial
        raise AuditFailure(f"malformed JSON: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def parquet_schema_columns(path: Path, *, label: str) -> tuple[str, ...]:
    """Inspect only Parquet metadata; never materialize a column value."""

    _require(
        not _linklike(path),
        f"symlink forbidden at Parquet schema boundary ({label}); junctions are links",
    )
    _require(path.is_file(), f"required Parquet is absent ({label}): {path}")
    try:
        import pyarrow.parquet as pq

        names = tuple(str(name) for name in pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:
        raise AuditFailure(f"malformed Parquet schema ({label}): {exc}") from exc
    _require(len(names) == len(set(names)), f"duplicate Parquet column name ({label})")
    return names


def require_exact_schema(
    observed: Sequence[str], allowed: Iterable[str], *, label: str
) -> tuple[str, ...]:
    expected = tuple(allowed)
    _require(
        len(observed) == len(expected) and set(observed) == set(expected),
        f"{label} violates exact zero-target schema boundary",
    )
    return tuple(observed)


def require_no_symlink_chain(path: Path, root: Path, *, label: str) -> None:
    """Reject a symlink at a controlled file or any parent below root."""

    root = root.resolve()
    current = path.absolute()
    try:
        current.relative_to(root)
    except ValueError as exc:
        raise AuditFailure(f"{label} is outside its controlled root") from exc
    chain: list[Path] = []
    while True:
        chain.append(current)
        if current == root:
            break
        current = current.parent
    for member in reversed(chain):
        _require(
            not _linklike(member),
            f"symlink forbidden in {label}: {member}; junctions are links",
        )


def preflight_zero_target_parquet_schemas(
    root: Path, contract: AuditContract
) -> dict[str, tuple[str, ...]]:
    """Reject unknown/label columns before pandas or Arrow reads any arrays."""

    census_path = root / "census" / "field_range_census.parquet"
    require_no_symlink_chain(census_path, root, label="field census path")
    census_columns = parquet_schema_columns(census_path, label="field census")
    require_exact_schema(census_columns, CENSUS_COLUMNS, label="field census")
    raw_path = root / OUTPUT_RELATIVE_PATHS["raw_parquet"]
    require_no_symlink_chain(raw_path, root, label="raw manifest path")
    raw_columns = parquet_schema_columns(raw_path, label="raw range manifest")
    allowed_raw_sets = (
        set(RAW_MANIFEST_BASE_COLUMNS),
        set(RAW_MANIFEST_BASE_COLUMNS) | {"resume_prefix_evidence"},
    )
    _require(
        len(raw_columns) == len(set(raw_columns))
        and set(raw_columns) in allowed_raw_sets,
        "raw range manifest violates exact zero-target schema boundary",
    )
    site_path = root / OUTPUT_RELATIVE_PATHS["site"]
    require_no_symlink_chain(site_path, root, label="decoded site path")
    site_columns = parquet_schema_columns(site_path, label="decoded site matrix")
    require_exact_schema(
        site_columns,
        SITE_METADATA_COLUMNS + FEATURE_COLUMNS,
        label="decoded site matrix",
    )
    group_path = root / OUTPUT_RELATIVE_PATHS["group"]
    require_no_symlink_chain(group_path, root, label="decoded group path")
    group_columns = parquet_schema_columns(group_path, label="decoded group matrix")
    require_exact_schema(
        group_columns,
        GROUP_METADATA_COLUMNS + FEATURE_COLUMNS,
        label="decoded group matrix",
    )
    _require(contract.site_count > 0, "invalid schema-preflight site contract")
    return {
        "census": census_columns,
        "raw": raw_columns,
        "site": site_columns,
        "group": group_columns,
    }


def path_under(root: Path, relative: str) -> Path:
    _require(bool(relative), "empty relative artifact path")
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"expected a root-relative path: {relative}")
    _require(
        ".." not in candidate.parts,
        f"parent traversal is forbidden in artifact path: {relative}",
    )
    resolved_root = root.resolve()
    joined = resolved_root / candidate
    require_no_symlink_chain(joined, resolved_root, label="root-relative artifact path")
    resolved = joined.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AuditFailure(f"artifact path escapes producer root: {relative}") from exc
    return resolved


def verify_identity(
    record: Mapping[str, Any],
    expected_path: Path,
    *,
    root: Path | None,
    label: str,
    allowed_extra_fields: Iterable[str] = (),
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{label} identity is not an object")
    _require(
        set(record)
        == {"path", "size_bytes", "sha256"} | set(allowed_extra_fields),
        f"{label} identity schema mismatch",
    )
    recorded_path = Path(str(record.get("path", "")))
    expected_lexical = Path(os.path.abspath(expected_path))
    if recorded_path.is_absolute():
        recorded_lexical = Path(os.path.abspath(recorded_path))
        _require(
            os.path.normcase(str(recorded_lexical))
            == os.path.normcase(str(expected_lexical)),
            f"{label} path identity mismatch",
        )
    else:
        _require(root is not None, f"{label} has a relative path without a root")
        root_lexical = Path(os.path.abspath(root))
        try:
            expected_relative = expected_lexical.relative_to(root_lexical).as_posix()
        except ValueError as exc:
            raise AuditFailure(f"{label} expected path is outside its root") from exc
        _require(
            recorded_path.as_posix() == expected_relative,
            f"{label} path identity mismatch",
        )
    if root is not None:
        require_no_symlink_chain(
            expected_lexical, root, label=f"{label} expected path"
        )
    _require(
        not _linklike(expected_lexical),
        f"{label} identity is a forbidden link",
    )
    resolved_expected = expected_lexical.resolve()
    _require(
        resolved_expected.is_file(),
        f"{label} file is absent: {resolved_expected}",
    )
    actual = identity(
        resolved_expected,
        root.resolve()
        if root is not None and not recorded_path.is_absolute()
        else None,
    )
    _require(int(record.get("size_bytes", -1)) == actual["size_bytes"], f"{label} size mismatch")
    _require(record.get("sha256") == actual["sha256"], f"{label} SHA-256 mismatch")
    return actual


def require_zero_facts(
    payload: Mapping[str, Any], required: Mapping[str, Any], *, label: str
) -> None:
    for key, expected in required.items():
        _require(key in payload, f"{label} lacks zero-access fact {key}")
        _require(payload[key] == expected, f"{label} violates zero-access fact {key}")


def require_no_unknown_keys(
    payload: Mapping[str, Any], allowed: Iterable[str], *, label: str
) -> None:
    unknown = set(payload) - set(allowed)
    _require(not unknown, f"{label} contains unrecognized fields: {sorted(unknown)}")


def event_id_for(row: Mapping[str, Any], start: int, end: int) -> str:
    return hashlib.sha256(
        (
            f"{row['object_key']}|{row['variable']}|{row['level']}|{start}|{end}"
        ).encode("utf-8")
    ).hexdigest()[:32]


def raw_relative_path(row: Mapping[str, Any]) -> str:
    run_date = str(row["run_init_utc"])[:10].replace("-", "")
    feature = f"{row['variable']}_{str(row['level']).replace(' ', '')}"
    return (
        f"raw/ranges/gfs.{run_date}/12/f{int(row['forecast_hour']):03d}/"
        f"{feature}.grib2"
    )


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def _parse_exact_utc_z(value: Any, *, label: str) -> datetime:
    _require(
        isinstance(value, str)
        and value.endswith("Z")
        and value.strip() == value,
        f"{label} is not an exact UTC-Z timestamp",
    )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AuditFailure(f"{label} is not a valid timestamp") from exc
    _require(
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{label} is not UTC",
    )
    return parsed


def normalize_nested_value(value: Any) -> Any:
    """Normalize pandas/Arrow scalar containers for exact JSON-sidecar comparison."""

    if isinstance(value, Mapping):
        normalized = {
            str(key): normalize_nested_value(item) for key, item in value.items()
        }
        return {key: item for key, item in normalized.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [normalize_nested_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if converted is not value:
            return normalize_nested_value(converted)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return normalize_nested_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _audit_active_lock(root: Path) -> None:
    active_paths = (
        root / "raw" / "RAW_LAUNCH_ACTIVE.lock",
        root / "decoded" / "DECODE_RECOVERY_ACTIVE.lock",
    )
    for active in active_paths:
        if not active.exists() and not _linklike(active):
            continue
        relative = active.relative_to(root).as_posix()
        if _linklike(active):
            raise AuditNotReady(
                "active-lock path is a link; postrun audit conditionally skipped",
                active_lock={"path": relative, "symlink": True},
            )
        details: dict[str, Any] = {
            "path": relative,
            "size_bytes": active.stat().st_size if active.is_file() else None,
        }
        if active.is_file():
            details["sha256"] = sha256_file(active)
            try:
                parsed = json.loads(active.read_text(encoding="utf-8"))
            except Exception:
                parsed = {"malformed": True}
            if isinstance(parsed, Mapping):
                for key in (
                    "artifact_type",
                    "attempt_id",
                    "pid",
                    "created_utc",
                    "malformed",
                ):
                    if key in parsed:
                        details[key] = parsed[key]
        raise AuditNotReady(
            "producer/recovery has an active lock; postrun audit conditionally skipped",
            active_lock=details,
        )


def _audit_provenance(
    root: Path, runner_path: Path, contract: AuditContract
) -> dict[str, Any]:
    runner_test_path = (
        REPO / "tests" / "test_noaa_gfs_multiseason_raw_v2.py"
        if contract == PRODUCTION_CONTRACT
        else runner_path.with_name("frozen_runner_test.py")
    )
    _require(not _linklike(runner_test_path), "runner-test path link is forbidden")
    runner_test_path = runner_test_path.resolve()
    authorization_path = root / "prereg" / "raw_launch_authorization_v1.json"
    go_path = root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    preflight_path = root / "manifest_raw_preflight_v5.json"
    preflight_amendment_path = (
        root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V5.json"
    )
    independent_path = (
        root
        / "independent_redteam"
        / "TRACK_A_RAW_RUNNER_INDEPENDENT_PREFLIGHT_AUDIT_V1.json"
    )
    census_path = root / "census" / "field_range_census.parquet"
    census_manifest_path = root / "manifest_census_v1.json"
    coordinate_path = root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json"
    cumulative_path = root / "audit" / "CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json"

    for provenance_path in (
        authorization_path,
        go_path,
        preflight_path,
        preflight_amendment_path,
        independent_path,
        census_manifest_path,
        cumulative_path,
        coordinate_path,
    ):
        require_no_symlink_chain(
            provenance_path,
            root,
            label="provenance control path",
        )

    authorization = load_json(authorization_path)
    go = load_json(go_path)
    preflight = load_json(preflight_path)
    amendment = load_json(preflight_amendment_path)
    independent = load_json(independent_path)
    census_manifest = load_json(census_manifest_path)
    cumulative = load_json(cumulative_path)
    coordinate = load_json(coordinate_path)

    # Reject unknown roles before dereferencing any embedded identity.  This is
    # deliberately not recursive: an injected {path,size,sha} can never make
    # this target-free auditor open an arbitrary label/2024/2025 file.
    require_no_unknown_keys(
        authorization,
        (
            "2024_arrays_read", "2025_arrays_read", "artifact_type",
            "authorization_created_utc", "bounded_predictor_manifest",
            "census_cumulative_access", "census_manifest",
            "census_plus_raw_max_http_attempts", "census_worst_case_http_attempts",
            "coordinate_lock", "effective_authorization_draft_amendment",
            "effective_preflight_amendment", "effective_preflight_manifest",
            "expected_range_bytes", "expected_range_rows", "field_range_census",
            "independent_census_audit", "independent_prelaunch_audit",
            "independent_target_free_preregister", "independent_windows_test_result",
            "labels_read", "models_fit", "post_download_free_disk_reserve_bytes",
            "py_compile_result", "raw_actual_http_attempt_budget", "runner",
            "runner_test", "runtime_identity", "runtime_identity_sha256",
            "schema_version", "status", "submission_csv_created",
        ),
        label="authorization",
    )
    require_no_unknown_keys(
        go,
        (
            "2024_arrays_read", "2025_arrays_read", "artifact_type",
            "bound_authorization_sha256", "bound_authorization_size_bytes",
            "bound_census_manifest_sha256", "bound_coordinate_lock_sha256",
            "bound_effective_preflight_manifest_sha256",
            "bound_independent_prelaunch_audit_sha256", "bound_runner_sha256",
            "bound_runtime_identity_sha256", "census_plus_raw_max_http_attempts",
            "census_worst_case_http_attempts", "created_utc", "expected_range_bytes",
            "expected_range_rows", "labels_read", "max_actual_http_attempts",
            "models_fit", "network_scope", "schema_version", "status",
            "submission_csv_created",
        ),
        label="independent GO",
    )
    require_no_unknown_keys(
        preflight,
        (
            "amendment", "artifact_type", "authorization_created",
            "authorization_draft_amendment", "created_utc", "independent_go_present",
            "independent_windows_test_result", "parent_manifest", "py_compile_result",
            "raw_network_requests", "runner", "runner_test", "schema_version",
            "seal_code", "seal_test",
        ),
        label="effective preflight manifest",
    )
    require_no_unknown_keys(
        amendment,
        (
            "2024_arrays_read", "2025_arrays_read", "artifact_type",
            "authorization_created", "independent_go_present", "independent_windows_test",
            "independent_windows_test_result", "labels_read", "models_fit",
            "parent_manifest", "py_compile", "py_compile_result", "raw_network_bytes",
            "raw_network_requests", "runner", "runner_test", "schema_version", "status",
            "submission_csv_created", "v4_artifacts_immutable",
        ),
        label="effective preflight amendment",
    )
    require_no_unknown_keys(
        independent,
        (
            "2024_arrays_read", "2025_arrays_read", "artifact_type", "audit_scope",
            "attempt_budget_audit", "code_and_test_identity", "code_review", "created_utc",
            "effective_preflight_chain", "field_census_independent_recheck",
            "final_authorization_requirements", "free_space_audit",
            "independent_execution_evidence", "provenance_rehash", "runtime_identity",
            "runtime_identity_sha256", "schema_version", "status", "verdict",
            "zero_output_inventory_before_audit_seal",
        ),
        label="independent prelaunch audit",
    )
    require_no_unknown_keys(
        census_manifest,
        (
            "access_ledger", "artifact_type", "created_utc", "day_summary",
            "field_range_census", "field_summary", "labels_read", "models_fit",
            "object_census", "parent_preregister_manifest", "progress_checkpoints",
            "raw_download_plan", "raw_downloaded_bytes", "raw_network_requests",
            "reproduction_code", "schema_version", "submission_csv_created", "summary",
            "test_code",
        ),
        label="census manifest",
    )
    require_no_unknown_keys(
        cumulative,
        (
            "2024_arrays_read", "2025_arrays_read", "artifact_type", "effect_on_science",
            "initial_population_invocation", "labels_read", "phase_total",
            "raw_payload_read", "successful_cache_reuse_invocation", "why_needed",
        ),
        label="cumulative census access",
    )
    require_no_unknown_keys(
        coordinate,
        (
            "artifact_type", "audit", "authoritative_source", "comparison_config",
            "coordinate_change_after_values_forbidden", "frozen_before_index_census_or_decode",
            "labels_read", "sites",
        ),
        label="coordinate lock",
    )

    rehashed_identity_paths: set[Path] = set()

    _require(
        authorization.get("artifact_type")
        == "NOAA_GFS_MULTISEASON_RAW_LAUNCH_AUTHORIZATION",
        "authorization artifact type mismatch",
    )
    _require(
        authorization.get("status")
        == "AUTHORIZED_RAW_RANGE_LAUNCH_ONLY_AFTER_INDEPENDENT_GO",
        "authorization status mismatch",
    )
    _require(
        authorization.get("py_compile_result") == "PASS"
        and "passed" in str(authorization.get("independent_windows_test_result", "")),
        "authorization code/test execution evidence mismatch",
    )
    require_zero_facts(
        authorization,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="authorization",
    )
    for key, expected in {
        "expected_range_rows": contract.range_rows,
        "expected_range_bytes": contract.range_bytes,
        "raw_actual_http_attempt_budget": contract.raw_attempt_cap,
        "census_worst_case_http_attempts": contract.census_worst_case_attempts,
        "census_plus_raw_max_http_attempts": contract.combined_attempt_cap,
    }.items():
        _require(int(authorization.get(key, -1)) == expected, f"authorization {key} mismatch")
    _require(
        int(authorization.get("post_download_free_disk_reserve_bytes", -1))
        == 200_000_000_000
        if contract == PRODUCTION_CONTRACT
        else True,
        "authorization 200 GB reserve mismatch",
    )
    _require(
        contract.raw_attempt_cap + contract.census_worst_case_attempts
        <= contract.combined_attempt_cap,
        "audit contract attempt arithmetic is invalid",
    )

    runner_actual = verify_identity(
        authorization.get("runner", {}), runner_path, root=None, label="authorized runner"
    )
    runner_test_actual = verify_identity(
        authorization.get("runner_test", {}),
        runner_test_path,
        root=None,
        label="authorized runner test",
    )
    verify_identity(
        authorization.get("field_range_census", {}),
        census_path,
        root=root,
        label="authorized field census",
    )
    verify_identity(
        authorization.get("census_manifest", {}),
        census_manifest_path,
        root=root,
        label="authorized census manifest",
    )
    verify_identity(
        authorization.get("coordinate_lock", {}),
        coordinate_path,
        root=root,
        label="authorized coordinate lock",
    )
    verify_identity(
        authorization.get("census_cumulative_access", {}),
        cumulative_path,
        root=root,
        label="authorized cumulative census access",
        allowed_extra_fields=(
            "successful_http_requests",
            "worst_case_http_attempts",
            "attempts_per_logical_request_hard_max",
        ),
    )
    preflight_actual = verify_identity(
        authorization.get("effective_preflight_manifest", {}),
        preflight_path,
        root=root,
        label="effective preflight manifest",
    )
    amendment_actual = verify_identity(
        authorization.get("effective_preflight_amendment", {}),
        preflight_amendment_path,
        root=root,
        label="effective preflight amendment",
    )
    independent_actual = verify_identity(
        authorization.get("independent_prelaunch_audit", {}),
        independent_path,
        root=root,
        label="independent prelaunch audit",
        allowed_extra_fields=("status",),
    )
    _require(
        authorization.get("independent_prelaunch_audit", {}).get("status")
        == "PASS_TO_CREATE_FINAL_AUTH_ONLY",
        "independent prelaunch audit status mismatch",
    )

    _require(
        preflight.get("artifact_type") == "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST"
        and int(preflight.get("schema_version", -1)) == 5,
        "effective preflight header mismatch",
    )
    _require(
        preflight.get("py_compile_result") == "PASS"
        and preflight.get("independent_windows_test_result") == "25 passed",
        "effective preflight code/test result mismatch",
    )
    verify_identity(preflight.get("runner", {}), runner_path, root=None, label="preflight runner")
    verify_identity(
        preflight.get("runner_test", {}),
        runner_test_path,
        root=None,
        label="preflight runner test",
    )
    verify_identity(
        preflight.get("amendment", {}),
        preflight_amendment_path,
        root=root,
        label="preflight amendment link",
    )
    _require(
        amendment.get("artifact_type")
        == "RAW_RUNNER_STATIC_PREFLIGHT_APPEND_ONLY_AMENDMENT"
        and int(amendment.get("schema_version", -1)) == 5,
        "preflight amendment header mismatch",
    )
    _require(
        amendment.get("py_compile_result") == "PASS"
        and amendment.get("independent_windows_test_result") == "25 passed",
        "preflight amendment code/test result mismatch",
    )
    verify_identity(amendment.get("runner", {}), runner_path, root=None, label="amendment runner")
    verify_identity(
        amendment.get("runner_test", {}),
        runner_test_path,
        root=None,
        label="amendment runner test",
    )
    require_zero_facts(
        amendment,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
            "raw_network_requests": 0,
            "raw_network_bytes": 0,
        },
        label="preflight amendment",
    )

    _require(
        independent.get("artifact_type")
        == "TRACK_A_RAW_RUNNER_INDEPENDENT_PREFLIGHT_AUDIT"
        and int(independent.get("schema_version", -1)) == 1
        and independent.get("status") == "PASS_TO_CREATE_FINAL_AUTH_ONLY",
        "independent prelaunch audit header/status mismatch",
    )
    code_review = independent.get("code_review")
    required_code_review = {
        "actual_http_attempt_reserved_before_urlopen": True,
        "attempt_15200_valid_and_attempt_15201_forbidden": True,
        "bounded_parallel_fail_fast_wave_max_workers": 8,
        "canonical_outputs_transactional_after_decode_and_physical_audit": True,
        "content_range_and_archive_header_identity_fail_closed": True,
        "crash_consistent_prefix_suffix_and_complete_part_recovery": True,
        "direct_sequential_exact_event_paths_no_per_range_directory_scan": True,
        "event_attempt_path_payload_and_global_inventory_fail_closed": True,
        "exact_single_grib_grid_run_valid_variable_and_level_checks": True,
        "final_and_projected_200gb_free_space_gates": True,
        "global_event_inventory_scanned_once": True,
        "remaining_concrete_static_or_windows_runtime_blockers": [],
        "windows_handle_types_and_writable_fsync_paths": True,
    }
    _require(
        isinstance(code_review, Mapping)
        and (
            dict(code_review) == required_code_review
            if contract == PRODUCTION_CONTRACT
            else all(code_review.get(key) == value for key, value in required_code_review.items())
        ),
        "independent prelaunch code-review PASS evidence mismatch",
    )
    execution = independent.get("independent_execution_evidence")
    _require(
        isinstance(execution, Mapping)
        and execution.get("py_compile_result") == "PASS"
        and "31 passed" in str(execution.get("pytest_result", ""))
        and int(execution.get("network_requests", -1)) == 0,
        "independent prelaunch execution evidence mismatch",
    )
    if contract == PRODUCTION_CONTRACT:
        provenance_rehash = independent.get("provenance_rehash")
        _require(
            isinstance(provenance_rehash, Mapping)
            and provenance_rehash.get("all_exact") is True
            and int(provenance_rehash.get("v4_source_and_provenance_chain_failures", -1)) == 0,
            "independent provenance-rehash PASS evidence mismatch",
        )
        attempt_audit = independent.get("attempt_budget_audit")
        _require(
            isinstance(attempt_audit, Mapping)
            and attempt_audit.get("identity_and_arithmetic_pass") is True
            and int(attempt_audit.get("census_attempts_per_logical_request_hard_max", -1)) == 4
            and int(attempt_audit.get("census_successful_logical_requests", -1)) == contract.census_successful_requests
            and int(attempt_audit.get("census_worst_case_http_attempts", -1)) == contract.census_worst_case_attempts
            and int(attempt_audit.get("raw_actual_http_attempt_budget", -1)) == contract.raw_attempt_cap
            and int(attempt_audit.get("raw_required_successful_ranges", -1)) == contract.range_rows
            and int(attempt_audit.get("census_plus_raw_max_http_attempts", -1)) == contract.combined_attempt_cap,
            "independent attempt-budget PASS evidence mismatch",
        )
        free_space = independent.get("free_space_audit")
        _require(
            isinstance(free_space, Mapping)
            and free_space.get("start_projection_pass") is True
            and int(free_space.get("expected_range_bytes", -1)) == contract.range_bytes
            and int(free_space.get("required_post_download_free_bytes", -1)) == 200_000_000_000
            and int(free_space.get("projected_free_after_expected_ranges_bytes", -1)) >= 200_000_000_000,
            "independent free-space PASS evidence mismatch",
        )
        census_recheck = independent.get("field_census_independent_recheck")
        _require(
            isinstance(census_recheck, Mapping)
            and census_recheck.get("all_range_bytes_positive") is True
            and census_recheck.get("all_range_bytes_within_object") is True
            and census_recheck.get("all_status_census_verified") is True
            and int(census_recheck.get("range_rows", -1)) == contract.range_rows
            and int(census_recheck.get("range_bytes", -1)) == contract.range_bytes
            and int(census_recheck.get("object_count", -1)) == contract.object_rows
            and census_recheck.get("families")
            == {
                "PBL_HEIGHT": contract.object_rows,
                "LOW_LEVEL_ISOBARIC_WIND_PROFILE": contract.object_rows * 8,
            },
            "independent field-census recheck mismatch",
        )
    verify_identity(
        independent.get("code_and_test_identity", {}).get("runner", {}),
        runner_path,
        root=None,
        label="independent-audit runner",
    )
    code_and_test = independent.get("code_and_test_identity")
    _require(
        isinstance(code_and_test, Mapping)
        and set(code_and_test) == {"runner", "runner_test"},
        "independent prelaunch code/test identity set mismatch",
    )
    verify_identity(
        code_and_test.get("runner_test", {}),
        runner_test_path,
        root=None,
        label="independent-audit runner test",
    )
    independent_manifest_v5 = (
        independent.get("effective_preflight_chain", {}).get("manifest_v5", {})
    )
    verify_identity(
        independent_manifest_v5,
        preflight_path,
        root=root,
        label="independent-audit effective preflight",
    )

    def bind_known_role(
        record: Any,
        expected_path: Path,
        *,
        label: str,
        required_in_production: bool = True,
    ) -> None:
        if record is None:
            _require(
                not (required_in_production and contract == PRODUCTION_CONTRACT),
                f"production provenance role is absent: {label}",
            )
            return
        _require(isinstance(record, Mapping), f"provenance role is malformed: {label}")
        _require(
            set(record) == {"path", "size_bytes", "sha256"},
            f"provenance identity schema mismatch: {label}",
        )
        recorded_path = Path(str(record.get("path", "")))
        verify_identity(
            record,
            expected_path,
            root=None if recorded_path.is_absolute() else root,
            label=label,
        )
        rehashed_identity_paths.add(expected_path.resolve())

    if contract == PRODUCTION_CONTRACT:
        provenance_rehash = independent["provenance_rehash"]
        require_no_unknown_keys(
            provenance_rehash,
            (
                "all_exact",
                "critical_identities",
                "v4_source_and_provenance_chain_entries_rehashed",
                "v4_source_and_provenance_chain_failures",
            ),
            label="independent provenance rehash",
        )
        critical = provenance_rehash.get("critical_identities")
        critical_paths = {
            "census_cumulative_access": cumulative_path,
            "census_manifest": census_manifest_path,
            "coordinate_lock": coordinate_path,
            "independent_census_audit": root / "independent_redteam" / "TRACK_A_CENSUS_INDEPENDENT_AUDIT_V1.json",
            "independent_target_free_preregister": root / "independent_redteam" / "TRACK_A_TARGET_FREE_AND_EXPANSION_PREREG.json",
            "raw_download_plan": root / "census" / "RAW_DOWNLOAD_PLAN_EXACT.json",
        }
        _require(
            isinstance(critical, Mapping) and set(critical) == set(critical_paths),
            "independent critical identity role set mismatch",
        )
        for role, expected_path in critical_paths.items():
            bind_known_role(
                critical[role],
                expected_path,
                label=f"independent critical identity {role}",
            )

    bind_known_role(
        authorization.get("bounded_predictor_manifest"),
        root / "independent_bounded_predictor" / "MANIFEST_V1.json",
        label="authorized bounded predictor manifest",
    )
    bind_known_role(
        authorization.get("effective_authorization_draft_amendment"),
        root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_AMENDMENT_V2.json",
        label="authorized effective draft amendment",
    )
    bind_known_role(
        authorization.get("independent_census_audit"),
        root / "independent_redteam" / "TRACK_A_CENSUS_INDEPENDENT_AUDIT_V1.json",
        label="authorized independent census audit",
    )
    bind_known_role(
        authorization.get("independent_target_free_preregister"),
        root / "independent_redteam" / "TRACK_A_TARGET_FREE_AND_EXPANSION_PREREG.json",
        label="authorized independent target-free preregister",
    )
    bind_known_role(
        preflight.get("parent_manifest"),
        root / "manifest_raw_preflight_v4.json",
        label="preflight parent manifest v4",
    )
    bind_known_role(
        preflight.get("authorization_draft_amendment"),
        root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_AMENDMENT_V2.json",
        label="preflight authorization draft amendment",
    )
    bind_known_role(
        preflight.get("seal_code"),
        REPO / "scripts" / "seal_noaa_gfs_multiseason_raw_preflight_amendment_v5.py",
        label="preflight v5 seal code",
    )
    bind_known_role(
        preflight.get("seal_test"),
        REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v5.py",
        label="preflight v5 seal test",
    )
    bind_known_role(
        amendment.get("parent_manifest"),
        root / "manifest_raw_preflight_v4.json",
        label="amendment parent manifest v4",
    )

    chain = independent.get("effective_preflight_chain", {})
    _require(isinstance(chain, Mapping), "independent effective preflight chain malformed")
    chain_paths = {
        "authorization_draft_amendment_v2": root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_AMENDMENT_V2.json",
        "authorization_draft_v1": root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_V1.json",
        "manifest_v4": root / "manifest_raw_preflight_v4.json",
        "manifest_v5": preflight_path,
        "preflight_amendment_v4": root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V4.json",
        "preflight_amendment_v5": preflight_amendment_path,
        "v4_sealer": REPO / "scripts" / "seal_noaa_gfs_multiseason_raw_preflight_amendment_v4.py",
        "v4_sealer_test": REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v4.py",
        "v5_sealer": REPO / "scripts" / "seal_noaa_gfs_multiseason_raw_preflight_amendment_v5.py",
        "v5_sealer_test": REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v5.py",
    }
    if contract == PRODUCTION_CONTRACT:
        _require(set(chain) == set(chain_paths), "independent effective preflight chain role set mismatch")
    for role, expected_path in chain_paths.items():
        bind_known_role(
            chain.get(role),
            expected_path,
            label=f"independent preflight chain {role}",
            required_in_production=True,
        )

    census_bindings = {
        "access_ledger": root / "census" / "CENSUS_ACCESS_LEDGER.json",
        "day_summary": root / "census" / "DAY_CENSUS_SUMMARY.csv",
        "field_range_census": census_path,
        "field_summary": root / "census" / "FIELD_CENSUS_SUMMARY.csv",
        "object_census": root / "census" / "object_census.parquet",
        "parent_preregister_manifest": root / "manifest_preregister_v2.json",
        "raw_download_plan": root / "census" / "RAW_DOWNLOAD_PLAN_EXACT.json",
        "reproduction_code": REPO / "scripts" / "census_noaa_gfs_multiseason_v2.py",
        "summary": root / "census" / "CENSUS_SUMMARY.md",
        "test_code": REPO / "tests" / "test_noaa_gfs_multiseason_census_v2.py",
    }
    for role, expected_path in census_bindings.items():
        bind_known_role(
            census_manifest.get(role),
            expected_path,
            label=f"census manifest {role}",
        )
    expected_census_progress = [
        root / "census" / "progress" / f"idx_files_{count:06d}.json"
        for count in tuple(range(100, 1_101, 100)) + (1_152,)
    ] + [root / "census" / "progress" / "list_runs_000048.json"]
    census_progress_records = census_manifest.get("progress_checkpoints")
    if contract == PRODUCTION_CONTRACT:
        _require(
            isinstance(census_progress_records, list)
            and len(census_progress_records) == len(expected_census_progress),
            "census progress identity inventory mismatch",
        )
        for record, expected_path in zip(
            census_progress_records, expected_census_progress
        ):
            bind_known_role(
                record,
                expected_path,
                label=f"census progress {expected_path.name}",
            )

    bind_known_role(
        coordinate.get("authoritative_source"),
        Path("data/local/open/info.xlsx"),
        label="coordinate authoritative source",
    )
    bind_known_role(
        coordinate.get("comparison_config"),
        REPO / "configs" / "copernicus_dem_directional_exposure_paired_increment_preregister_v1.json",
        label="coordinate comparison config",
    )

    cumulative_progress = (
        cumulative.get("initial_population_invocation", {})
        if isinstance(cumulative.get("initial_population_invocation"), Mapping)
        else {}
    ).get("durable_checkpoint_identities")
    if contract == PRODUCTION_CONTRACT:
        _require(
            isinstance(cumulative_progress, list)
            and len(cumulative_progress) == len(expected_census_progress),
            "cumulative census checkpoint identity inventory mismatch",
        )
        for record, expected_path in zip(cumulative_progress, expected_census_progress):
            bind_known_role(
                record,
                expected_path,
                label=f"cumulative census progress {expected_path.name}",
            )

    authorization_actual = identity(authorization_path, root)
    go_actual = identity(go_path, root)
    _require(
        go.get("artifact_type") == "TRACK_A_INDEPENDENT_RAW_RANGE_LAUNCH_GO"
        and go.get("status") == "GO_RAW_RANGE_LAUNCH",
        "independent GO header/status mismatch",
    )
    if contract == PRODUCTION_CONTRACT:
        _require(
            go.get("network_scope")
            == "ONLY_FROZEN_NOAA_GFS_BYTE_RANGES_IN_FIELD_CENSUS",
            "independent GO network scope mismatch",
        )
    require_zero_facts(
        go,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="independent GO",
    )
    go_bindings = {
        "bound_authorization_sha256": authorization_actual["sha256"],
        "bound_authorization_size_bytes": authorization_actual["size_bytes"],
        "bound_runner_sha256": runner_actual["sha256"],
        "bound_census_manifest_sha256": identity(census_manifest_path, root)["sha256"],
        "bound_coordinate_lock_sha256": identity(coordinate_path, root)["sha256"],
        "bound_effective_preflight_manifest_sha256": preflight_actual["sha256"],
        "bound_independent_prelaunch_audit_sha256": independent_actual["sha256"],
        "expected_range_rows": contract.range_rows,
        "expected_range_bytes": contract.range_bytes,
        "max_actual_http_attempts": contract.raw_attempt_cap,
        "census_worst_case_http_attempts": contract.census_worst_case_attempts,
        "census_plus_raw_max_http_attempts": contract.combined_attempt_cap,
    }
    for key, expected in go_bindings.items():
        _require(go.get(key) == expected, f"independent GO binding mismatch: {key}")

    runtime = authorization.get("runtime_identity")
    _require(isinstance(runtime, Mapping), "authorization runtime identity is absent")
    require_no_unknown_keys(
        runtime,
        (
            "packages",
            "platform",
            "python_executable",
            "python_executable_sha256",
            "python_executable_size_bytes",
            "python_version",
        ),
        label="runtime identity",
    )
    runtime_sha = canonical_payload_sha256(runtime)
    _require(
        authorization.get("runtime_identity_sha256") == runtime_sha,
        "authorization runtime canonical digest mismatch",
    )
    _require(
        go.get("bound_runtime_identity_sha256") == runtime_sha,
        "GO runtime canonical digest mismatch",
    )
    executable = Path(str(runtime.get("python_executable", ""))).resolve()
    if contract == PRODUCTION_CONTRACT:
        _require(
            executable == Path(sys.executable).resolve(),
            "production runtime Python path is not the active authorized executable",
        )
    _require(executable.is_file(), "authorized Python executable is absent")
    _require(
        executable.stat().st_size == int(runtime.get("python_executable_size_bytes", -1))
        and sha256_file(executable) == runtime.get("python_executable_sha256"),
        "authorized Python executable identity changed",
    )
    rehashed_identity_paths.add(executable)
    packages = runtime.get("packages")
    _require(isinstance(packages, Mapping), "runtime package identity map is absent")
    if contract == PRODUCTION_CONTRACT:
        _require(
            set(packages) == {"eccodes", "numpy", "pandas", "pyarrow"},
            "production runtime package identity set mismatch",
        )
    for package, record in packages.items():
        _require(isinstance(record, Mapping), f"runtime package record malformed: {package}")
        require_no_unknown_keys(
            record,
            ("module_file", "module_file_sha256", "module_file_size_bytes", "version"),
            label=f"runtime package identity {package}",
        )
        module_path = Path(str(record.get("module_file", ""))).resolve()
        if contract == PRODUCTION_CONTRACT:
            actual_module_path = Path(
                str(importlib.import_module(str(package)).__file__)
            ).resolve()
            _require(
                module_path == actual_module_path,
                f"production runtime package path differs from active module: {package}",
            )
        _require(module_path.is_file(), f"runtime package module absent: {package}")
        _require(
            module_path.stat().st_size == int(record.get("module_file_size_bytes", -1))
            and sha256_file(module_path) == record.get("module_file_sha256"),
            f"runtime package identity changed: {package}",
        )
        rehashed_identity_paths.add(module_path)

    rehashed_identity_paths.update(
        {
            SUPERSEDED_V2_AUDITOR.resolve(),
            (REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v1.py").resolve(),
            runner_path.resolve(),
            runner_test_path.resolve(),
            census_path.resolve(),
            census_manifest_path.resolve(),
            coordinate_path.resolve(),
            cumulative_path.resolve(),
            preflight_path.resolve(),
            preflight_amendment_path.resolve(),
            independent_path.resolve(),
            authorization_path.resolve(),
            go_path.resolve(),
        }
    )
    embedded_identity_rehashes = len(rehashed_identity_paths)

    external_identity_paths: list[Path] = []
    resolved_root = root.resolve()
    for path in sorted(rehashed_identity_paths):
        try:
            path.relative_to(resolved_root)
        except ValueError:
            external_identity_paths.append(path)
    external_identity_bytes = sum(path.stat().st_size for path in external_identity_paths)

    cumulative_binding = authorization["census_cumulative_access"]
    for key, expected in {
        "successful_http_requests": contract.census_successful_requests,
        "worst_case_http_attempts": contract.census_worst_case_attempts,
    }.items():
        _require(int(cumulative_binding.get(key, -1)) == expected, f"census binding {key} mismatch")
    _require(
        cumulative.get("artifact_type")
        == "CENSUS_CUMULATIVE_ACCESS_APPEND_ONLY_AMENDMENT",
        "cumulative census access artifact type mismatch",
    )
    require_zero_facts(
        cumulative,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "raw_payload_read": False,
        },
        label="cumulative census access",
    )
    _require(
        int(cumulative.get("phase_total", {}).get("successful_http_requests", -1))
        == contract.census_successful_requests,
        "cumulative census successful request total mismatch",
    )
    _require(
        int(cumulative.get("phase_total", {}).get("raw_range_requests", -1)) == 0,
        "census phase improperly includes raw range requests",
    )
    _require(
        int(cumulative.get("phase_total", {}).get("raw_range_bytes", -1)) == 0,
        "census phase improperly includes raw range bytes",
    )
    initial_population = cumulative.get("initial_population_invocation")
    _require(
        isinstance(initial_population, Mapping)
        and int(initial_population.get("successful_http_requests", -1))
        == contract.census_successful_requests,
        "census initial-population successful request total mismatch",
    )
    _require(
        int(cumulative_binding.get("attempts_per_logical_request_hard_max", -1)) == 4
        and int(cumulative_binding.get("worst_case_http_attempts", -1))
        == contract.census_worst_case_attempts,
        "authorized census hard-attempt arithmetic mismatch",
    )
    _require(
        census_manifest.get("artifact_type") == "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST",
        "census manifest artifact type mismatch",
    )
    require_zero_facts(
        census_manifest,
        {"labels_read": False, "models_fit": 0, "submission_csv_created": False},
        label="census manifest",
    )
    _require(
        coordinate.get("artifact_type") == "AUTHORITATIVE_TURBINE_COORDINATE_LOCK"
        and coordinate.get("labels_read") is False,
        "coordinate lock header/target-free fact mismatch",
    )

    return {
        "authorization": authorization,
        "authorization_identity": authorization_actual,
        "go": go,
        "go_identity": go_actual,
        "preflight_identity": preflight_actual,
        "preflight_amendment_identity": amendment_actual,
        "independent_prelaunch_identity": dict(
            authorization["independent_prelaunch_audit"]
        ),
        "independent_prelaunch_identity_base": independent_actual,
        "runner_identity": runner_actual,
        "runner_test_identity": runner_test_actual,
        "runtime_identity": dict(runtime),
        "runtime_identity_sha256": runtime_sha,
        "embedded_identity_rehashes": embedded_identity_rehashes,
        "external_identity_paths": [
            str(path.resolve()) for path in external_identity_paths
        ],
        "external_identity_unique_file_count": len(external_identity_paths),
        "external_identity_unique_file_bytes": external_identity_bytes,
        "coordinate": coordinate,
        "coordinate_identity": identity(coordinate_path, root),
        "census_manifest_identity": identity(census_manifest_path, root),
        "field_census_path": census_path,
    }


def _audit_original_transaction_and_launch(root: Path) -> dict[str, Any]:
    outputs = {name: root / relative for name, relative in OUTPUT_RELATIVE_PATHS.items()}
    for name, path in outputs.items():
        _require(not _linklike(path), f"canonical output link is forbidden ({name})")
        _require(path.is_file(), f"canonical output is absent ({name}): {path}")

    transaction_root = root / "raw" / "output_transactions"
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "output transaction root is absent or a link",
    )
    transaction_entries = list(transaction_root.rglob("*"))
    symlinks = [
        path.relative_to(root).as_posix()
        for path in transaction_entries
        if _linklike(path)
    ]
    _require(not symlinks, f"symlink forbidden in output transaction inventory: {symlinks[:5]}")
    _require(
        all(path.is_file() or path.is_dir() for path in transaction_entries),
        "special filesystem object forbidden in output transaction inventory",
    )
    plans = sorted(transaction_root.glob("*__plan.json"))
    commits = sorted(transaction_root.glob("*__committed.json"))
    _require(len(plans) == 1, "expected exactly one canonical output transaction plan")
    _require(len(commits) == 1, "expected exactly one canonical output transaction commit")
    plan_path = plans[0]
    commit_path = commits[0]
    plan = load_json(plan_path)
    commit = load_json(commit_path)
    _require(
        set(plan)
        == {
            "artifact_type", "schema_version", "attempt_id", "created_utc",
            "items", "canonical_outputs_absent_before_plan",
            "decode_and_physical_audit_completed_before_plan",
        },
        "transaction plan payload schema mismatch",
    )
    _require(
        set(commit)
        == {
            "artifact_type", "schema_version", "attempt_id", "plan", "outputs",
            "all_outputs_reopened_and_rehashed",
        },
        "transaction commit payload schema mismatch",
    )
    attempt_id = str(plan.get("attempt_id", ""))
    _require(
        plan_path.name == f"{attempt_id}__plan.json"
        and commit_path.name == f"{attempt_id}__committed.json",
        "transaction filename/attempt identity mismatch",
    )
    _require(
        plan.get("artifact_type") == "RAW_OUTPUT_TRANSACTION_PLAN"
        and int(plan.get("schema_version", -1)) == 1
        and plan.get("canonical_outputs_absent_before_plan") is True
        and plan.get("decode_and_physical_audit_completed_before_plan") is True,
        "transaction plan header/gates mismatch",
    )
    items = plan.get("items")
    _require(isinstance(items, list), "transaction plan items are not a list")
    _require(
        all(
            isinstance(item, Mapping)
            and set(item)
            == {
                "name", "staged_path", "destination_path", "size_bytes", "sha256"
            }
            for item in items
        ),
        "transaction plan item schema mismatch",
    )
    by_name = {str(item.get("name")): item for item in items if isinstance(item, Mapping)}
    _require(len(items) == len(by_name) == len(outputs), "transaction plan item cardinality mismatch")
    _require(set(by_name) == set(outputs), "transaction plan output name set mismatch")
    output_identities: dict[str, dict[str, Any]] = {}
    allowed_transaction_files = {plan_path.resolve(), commit_path.resolve()}
    allowed_transaction_directories = {transaction_root.resolve()}
    planned_staged_files_present = 0
    for name, destination in outputs.items():
        item = by_name[name]
        expected_destination = destination.relative_to(root).as_posix()
        expected_staged = (
            f"raw/output_transactions/{attempt_id}/staged/{expected_destination}"
        )
        _require(item.get("destination_path") == expected_destination, f"transaction destination mismatch: {name}")
        _require(item.get("staged_path") == expected_staged, f"transaction staged path mismatch: {name}")
        actual = identity(destination, root)
        _require(
            int(item.get("size_bytes", -1)) == actual["size_bytes"]
            and item.get("sha256") == actual["sha256"],
            f"canonical output differs from transaction plan: {name}",
        )
        staged = path_under(root, expected_staged)
        allowed_transaction_files.add(staged.resolve())
        parent = staged.parent.resolve()
        while True:
            allowed_transaction_directories.add(parent)
            if parent == transaction_root.resolve():
                break
            _require(
                transaction_root.resolve() in parent.parents,
                f"planned staged parent escapes transaction root: {name}",
            )
            parent = parent.parent
        if staged.exists():
            _require(staged.is_file(), f"transaction staged remnant is not a file: {name}")
            _require(
                staged.stat().st_size == actual["size_bytes"]
                and sha256_file(staged) == actual["sha256"],
                f"transaction staged remnant differs: {name}",
            )
            planned_staged_files_present += 1
        output_identities[name] = actual

    _require(
        commit.get("artifact_type") == "RAW_OUTPUT_TRANSACTION_COMMIT"
        and int(commit.get("schema_version", -1)) == 1
        and commit.get("attempt_id") == attempt_id
        and commit.get("all_outputs_reopened_and_rehashed") is True,
        "transaction commit header mismatch",
    )
    verify_identity(commit.get("plan", {}), plan_path, root=root, label="transaction commit plan")
    committed_outputs = commit.get("outputs")
    _require(isinstance(committed_outputs, Mapping), "transaction commit output map is absent")
    _require(set(committed_outputs) == set(outputs), "transaction commit output name set mismatch")
    for name, destination in outputs.items():
        verify_identity(
            committed_outputs[name], destination, root=root, label=f"committed output {name}"
        )

    unexpected_files = [
        path.relative_to(root).as_posix()
        for path in transaction_entries
        if path.is_file() and path.resolve() not in allowed_transaction_files
    ]
    _require(
        not unexpected_files,
        f"unplanned output transaction file remains: {unexpected_files[:5]}",
    )
    unexpected_directories = [
        path.relative_to(root).as_posix()
        for path in transaction_entries
        if path.is_dir() and path.resolve() not in allowed_transaction_directories
    ]
    _require(
        not unexpected_directories,
        f"unplanned output transaction directory remains: {unexpected_directories[:5]}",
    )

    decoded_root = root / "decoded"
    _require(
        decoded_root.is_dir() and not _linklike(decoded_root),
        "canonical decoded directory is absent or a link",
    )
    decoded_entries = list(decoded_root.rglob("*"))
    decoded_symlinks = [
        path.relative_to(root).as_posix()
        for path in decoded_entries
        if _linklike(path)
    ]
    _require(
        not decoded_symlinks,
        f"symlink forbidden in decoded output inventory: {decoded_symlinks[:5]}",
    )
    _require(
        all(path.is_file() or path.is_dir() for path in decoded_entries),
        "special filesystem object forbidden in decoded output inventory",
    )
    decoded_directories = [
        path.relative_to(root).as_posix()
        for path in decoded_entries
        if path.is_dir()
    ]
    _require(
        not decoded_directories,
        f"unexpected directory in decoded output inventory: {decoded_directories[:5]}",
    )
    expected_decoded_files = {
        outputs[name].resolve() for name in ("site", "group", "audit", "lock")
    }
    actual_decoded_files = {
        path.resolve() for path in decoded_entries if path.is_file()
    }
    unexpected_decoded_files = actual_decoded_files - expected_decoded_files
    _require(
        not unexpected_decoded_files,
        "unrecognized file in decoded output inventory: "
        f"{[path.relative_to(root).as_posix() for path in sorted(unexpected_decoded_files)[:5]]}",
    )
    _require(
        actual_decoded_files == expected_decoded_files,
        "canonical decoded filesystem inventory is incomplete",
    )

    manifest = load_json(outputs["manifest"])
    _require(manifest.get("launch_attempt_id") == attempt_id, "manifest/transaction attempt mismatch")
    history_root = root / "raw" / "launch_history"
    require_no_symlink_chain(history_root, root, label="launch history path")
    matching_complete = history_root / f"{attempt_id}__complete.lock"
    _require(
        matching_complete.is_file() and not _linklike(matching_complete),
        "matching producer complete lock is absent or a link",
    )
    complete_payload = load_json(matching_complete)
    _require(complete_payload.get("attempt_id") == attempt_id, "complete lock payload mismatch")
    _require(history_root.is_dir(), "launch history directory is absent")
    history_entries = list(history_root.rglob("*"))
    history_symlinks = [
        path.relative_to(root).as_posix()
        for path in history_entries
        if _linklike(path)
    ]
    _require(
        not history_symlinks,
        f"symlink forbidden in launch history inventory: {history_symlinks[:5]}",
    )
    history_directories = [
        path.relative_to(root).as_posix()
        for path in history_entries
        if path.is_dir()
    ]
    _require(
        not history_directories,
        f"unexpected directory in launch history inventory: {history_directories[:5]}",
    )
    _require(
        all(path.is_file() for path in history_entries),
        "launch history contains a special filesystem object",
    )
    history_files = sorted(path for path in history_entries if path.is_file())
    _require(matching_complete in history_files, "complete lock is outside launch history inventory")
    history_attempt_ids: set[str] = set()
    for history_path in history_files:
        payload = load_json(history_path)
        _require(
            set(payload) == {"attempt_id", "pid", "created_utc"},
            f"launch history payload schema mismatch: {history_path.name}",
        )
        suffixes = ("__complete.lock", "__failed.lock", "__stale_dead_pid.lock")
        suffix = next((value for value in suffixes if history_path.name.endswith(value)), None)
        _require(suffix is not None, f"unexpected launch history filename: {history_path.name}")
        assert suffix is not None
        stem_attempt = history_path.name[: -len(suffix)]
        _require(payload.get("attempt_id") == stem_attempt, f"launch history filename/payload mismatch: {history_path.name}")
        _require(
            stem_attempt not in history_attempt_ids,
            f"duplicate launch-history status for attempt: {stem_attempt}",
        )
        history_attempt_ids.add(stem_attempt)

    raw_root = root / "raw"
    _require(raw_root.is_dir() and not _linklike(raw_root), "raw root is absent or a link")
    expected_raw_top = {
        "RAW_RANGE_MANIFEST.parquet",
        "RAW_RANGE_MANIFEST.csv",
        "RAW_ACCESS_LEDGER.json",
        "ranges",
        "request_events",
        "progress",
        "launch_history",
        "output_transactions",
    }
    raw_top_entries = list(raw_root.iterdir())
    raw_top_symlinks = [path.name for path in raw_top_entries if _linklike(path)]
    _require(
        not raw_top_symlinks,
        f"symlink forbidden in raw top-level inventory: {raw_top_symlinks[:5]}",
    )
    _require(
        {path.name for path in raw_top_entries} == expected_raw_top,
        "raw top-level filesystem inventory mismatch",
    )
    for name in ("RAW_RANGE_MANIFEST.parquet", "RAW_RANGE_MANIFEST.csv", "RAW_ACCESS_LEDGER.json"):
        _require((raw_root / name).is_file(), f"raw top-level canonical file absent: {name}")
    for name in ("ranges", "request_events", "progress", "launch_history", "output_transactions"):
        _require((raw_root / name).is_dir(), f"raw top-level canonical directory absent: {name}")

    leftovers = sorted(
        path.relative_to(root).as_posix()
        for pattern in ("*.part", "*.meta.part.json", "*.writepart")
        for path in (root / "raw").rglob(pattern)
    )
    _require(not leftovers, f"unfinished raw/transaction staging files remain: {leftovers[:5]}")
    return {
        "outputs": outputs,
        "output_identities": output_identities,
        "plan": plan,
        "plan_identity": identity(plan_path, root),
        "commit": commit,
        "commit_identity": identity(commit_path, root),
        "attempt_id": attempt_id,
        "complete_lock_identity": identity(matching_complete, root),
        "launch_history_lock_count": len(history_files),
        "transaction_inventory_recursively_closed": True,
        "planned_staged_files_present": planned_staged_files_present,
        "manifest": manifest,
        "recovered": False,
    }


def _linklike(path: Path) -> bool:
    return path.is_symlink() or bool(
        hasattr(path, "is_junction") and path.is_junction()
    )


def _audit_recovery_transaction_and_launch(
    root: Path, contract: AuditContract, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Close the separately authorized network-zero recovery transaction."""

    outputs = {name: root / relative for name, relative in OUTPUT_RELATIVE_PATHS.items()}
    for name, path in outputs.items():
        _require(not _linklike(path), f"canonical recovered output link is forbidden ({name})")
        _require(path.is_file(), f"canonical recovered output is absent ({name}): {path}")

    _require(
        manifest.get("artifact_type")
        == "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST_RECOVERED_V2",
        "V2 recovered manifest artifact type mismatch",
    )
    transaction_root = root / "raw" / "output_transactions"
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "recovery output transaction root is absent or a link",
    )
    transaction_entries = list(transaction_root.rglob("*"))
    link_entries = [
        path.relative_to(root).as_posix()
        for path in transaction_entries
        if _linklike(path)
    ]
    _require(
        not link_entries,
        f"link forbidden in recovery transaction inventory: {link_entries[:5]}",
    )
    _require(
        all(path.is_file() or path.is_dir() for path in transaction_entries),
        "special filesystem object forbidden in recovery transaction inventory",
    )
    plans = sorted(transaction_root.glob("*__plan.json"))
    commits = sorted(transaction_root.glob("*__committed.json"))
    _require(len(plans) == 1, "expected exactly one recovery transaction plan")
    _require(len(commits) == 1, "expected exactly one recovery transaction commit")
    plan_path = plans[0]
    commit_path = commits[0]
    plan = load_json(plan_path)
    commit = load_json(commit_path)
    _require(
        set(plan)
        == {
            "artifact_type",
            "schema_version",
            "attempt_id",
            "created_utc",
            "recovery_authorization",
            "independent_recovery_go",
            "decode_failure_incident",
            "flawed_launch_incident",
            "launch_identity_correction",
            "superseded_v1_chain",
            "superseded_v1_support",
            "items",
            "canonical_outputs_absent_before_plan",
            "all_10368_messages_decoded_before_plan",
            "physical_audit_completed_before_plan",
            "network_requests",
        },
        "recovery transaction plan payload schema mismatch",
    )
    _require(
        set(commit)
        == {
            "artifact_type",
            "schema_version",
            "attempt_id",
            "plan",
            "outputs",
            "atomic_method",
            "all_outputs_reopened_and_rehashed",
            "network_requests",
        },
        "recovery transaction commit payload schema mismatch",
    )
    attempt_id = str(plan.get("attempt_id", ""))
    _require(
        attempt_id.startswith("decode_recovery_v2__")
        and all(character.isalnum() or character in "_-" for character in attempt_id),
        "invalid recovery attempt id",
    )
    _require(
        plan_path.name == f"{attempt_id}__plan.json"
        and commit_path.name == f"{attempt_id}__committed.json",
        "recovery transaction filename/attempt mismatch",
    )
    _require(
        plan.get("artifact_type") == "DECODE_RECOVERY_OUTPUT_TRANSACTION_PLAN_V2"
        and int(plan.get("schema_version", -1)) == 2
        and plan.get("canonical_outputs_absent_before_plan") is True
        and plan.get("all_10368_messages_decoded_before_plan") is True
        and plan.get("physical_audit_completed_before_plan") is True
        and int(plan.get("network_requests", -1)) == 0,
        "recovery transaction plan gates mismatch",
    )
    _require(
        manifest.get("launch_attempt_id") == attempt_id,
        "recovered manifest/transaction attempt mismatch",
    )

    authorization_path = root / RECOVERY_AUTH_RELATIVE
    go_path = root / RECOVERY_GO_RELATIVE
    incident_path = root / RECOVERY_INCIDENT_RELATIVE
    flawed_launch_incident_path = root / RECOVERY_FLAWED_LAUNCH_INCIDENT_RELATIVE
    launch_identity_correction_path = (
        root / RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE
    )
    for control_path, label in (
        (authorization_path, "recovery authorization"),
        (go_path, "recovery GO"),
        (incident_path, "recovery incident"),
        (flawed_launch_incident_path, "preserved flawed V1 launch incident"),
        (launch_identity_correction_path, "authoritative V1 launch correction"),
    ):
        require_no_symlink_chain(control_path, root, label=label)
        _require(not _linklike(control_path), f"{label} link is forbidden")
    authorization = load_json(authorization_path)
    incident = load_json(incident_path)
    flawed_launch_incident = load_json(flawed_launch_incident_path)
    launch_identity_correction = load_json(launch_identity_correction_path)
    authorization_identity = verify_identity(
        plan.get("recovery_authorization", {}),
        authorization_path,
        root=root,
        label="recovery-plan authorization",
    )
    go_identity = verify_identity(
        plan.get("independent_recovery_go", {}),
        go_path,
        root=root,
        label="recovery-plan independent GO",
    )
    incident_identity = verify_identity(
        plan.get("decode_failure_incident", {}),
        incident_path,
        root=root,
        label="recovery-plan incident",
    )
    flawed_launch_incident_identity = verify_identity(
        plan.get("flawed_launch_incident", {}),
        flawed_launch_incident_path,
        root=root,
        label="recovery-plan preserved flawed V1 launch incident",
    )
    launch_identity_correction_identity = verify_identity(
        plan.get("launch_identity_correction", {}),
        launch_identity_correction_path,
        root=root,
        label="recovery-plan authoritative V1 launch correction",
    )
    failed_attempt = str(incident.get("failed_attempt_id", ""))
    if contract == PRODUCTION_CONTRACT:
        _require(
            failed_attempt == RECOVERY_FAILED_ATTEMPT
            and incident_identity["sha256"] == RECOVERY_INCIDENT_SHA256,
            "production recovery incident constant mismatch",
        )
    _require(
        bool(failed_attempt) and failed_attempt != attempt_id,
        "recovery attempt does not differ from failed producer attempt",
    )
    _require(
        authorization.get("recovery_attempt_id") == attempt_id,
        "authorization/recovery transaction attempt mismatch",
    )
    _require(
        authorization.get("incident") == incident_identity,
        "authorization/recovery incident mismatch",
    )
    _require(
        authorization.get("flawed_launch_incident")
        == flawed_launch_incident_identity,
        "authorization/preserved flawed launch incident mismatch",
    )
    _require(
        authorization.get("launch_identity_correction")
        == launch_identity_correction_identity,
        "authorization/authoritative launch correction mismatch",
    )
    if contract == PRODUCTION_CONTRACT:
        _require(
            flawed_launch_incident_identity["sha256"]
            == RECOVERY_FLAWED_LAUNCH_INCIDENT_SHA256
            and launch_identity_correction_identity["sha256"]
            == RECOVERY_LAUNCH_IDENTITY_CORRECTION_SHA256,
            "production launch incident/correction digest mismatch",
        )
    authoritative_v1_support_external = _validate_v2_support_path_forms(
        authorization.get("superseded_v1_support"),
        launch_identity_correction.get("authoritative_v1_support"),
    )
    _require(
        plan.get("superseded_v1_chain") == RECOVERY_V1_CHAIN_IDENTITIES
        and authorization.get("superseded_v1_chain")
        == RECOVERY_V1_CHAIN_IDENTITIES
        and plan.get("superseded_v1_support")
        == authoritative_v1_support_external
        and authorization.get("superseded_v1_support")
        == authoritative_v1_support_external,
        "V2 recovery transaction did not bind the authoritative V1 chain/support",
    )

    items = plan.get("items")
    _require(isinstance(items, list), "recovery transaction items are not a list")
    _require(
        all(
            isinstance(item, Mapping)
            and set(item)
            == {"name", "staged_path", "destination_path", "size_bytes", "sha256"}
            for item in items
        ),
        "recovery transaction item schema mismatch",
    )
    by_name = {str(item.get("name")): item for item in items if isinstance(item, Mapping)}
    _require(
        len(items) == len(by_name) == len(outputs) and set(by_name) == set(outputs),
        "recovery transaction output name/cardinality mismatch",
    )
    output_identities: dict[str, dict[str, Any]] = {}
    allowed_files = {plan_path.resolve(), commit_path.resolve()}
    allowed_directories = {transaction_root.resolve()}
    for name, destination in outputs.items():
        item = by_name[name]
        destination_relative = destination.relative_to(root).as_posix()
        staged_relative = (
            f"raw/output_transactions/{attempt_id}/staged/{destination_relative}"
        )
        _require(
            item.get("destination_path") == destination_relative
            and item.get("staged_path") == staged_relative,
            f"recovery transaction path mismatch: {name}",
        )
        actual = identity(destination, root)
        _require(
            int(item.get("size_bytes", -1)) == actual["size_bytes"]
            and item.get("sha256") == actual["sha256"],
            f"canonical recovered output differs from plan: {name}",
        )
        staged_path = path_under(root, staged_relative)
        _require(
            not staged_path.exists() and not _linklike(staged_path),
            f"recovery staged output remains after commit: {name}",
        )
        parent = staged_path.parent.resolve()
        while True:
            allowed_directories.add(parent)
            if parent == transaction_root.resolve():
                break
            _require(
                transaction_root.resolve() in parent.parents,
                f"recovery staged parent escapes transaction root: {name}",
            )
            parent = parent.parent
        output_identities[name] = actual

    _require(
        commit.get("artifact_type") == "DECODE_RECOVERY_OUTPUT_TRANSACTION_COMMIT_V2"
        and int(commit.get("schema_version", -1)) == 2
        and commit.get("attempt_id") == attempt_id
        and commit.get("atomic_method")
        == "os.link_create_if_absent_then_unlink_staging"
        and commit.get("all_outputs_reopened_and_rehashed") is True
        and int(commit.get("network_requests", -1)) == 0,
        "recovery transaction commit header/gates mismatch",
    )
    verify_identity(
        commit.get("plan", {}), plan_path, root=root, label="recovery commit plan"
    )
    committed_outputs = commit.get("outputs")
    _require(isinstance(committed_outputs, Mapping), "recovery commit outputs absent")
    _require(set(committed_outputs) == set(outputs), "recovery commit output set mismatch")
    for name, destination in outputs.items():
        verify_identity(
            committed_outputs[name],
            destination,
            root=root,
            label=f"recovery committed output {name}",
        )

    remnants = authorization.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list) and len(remnants) == 2,
        "recovery authorization remnant inventory mismatch",
    )
    incident_remnants = incident.get("documented_preplan_staging_remnants")
    _require(
        incident_remnants == remnants,
        "incident/authorization preplan remnant inventory mismatch",
    )
    expected_remnant_relatives = {
        f"raw/output_transactions/{failed_attempt}/staged/raw/RAW_RANGE_MANIFEST.csv",
        f"raw/output_transactions/{failed_attempt}/staged/raw/RAW_RANGE_MANIFEST.parquet",
    }
    remnant_by_path = {
        str(record.get("path")): record
        for record in remnants
        if isinstance(record, Mapping)
    }
    _require(
        len(remnant_by_path) == 2
        and set(remnant_by_path) == expected_remnant_relatives,
        "preplan remnant path set mismatch",
    )
    for relative in sorted(expected_remnant_relatives):
        remnant_path = path_under(root, relative)
        verify_identity(
            remnant_by_path[relative],
            remnant_path,
            root=root,
            label="incident-bound preplan remnant",
        )
        allowed_files.add(remnant_path.resolve())
        parent = remnant_path.parent.resolve()
        while True:
            allowed_directories.add(parent)
            if parent == transaction_root.resolve():
                break
            _require(
                transaction_root.resolve() in parent.parents,
                "incident-bound remnant parent escapes transaction root",
            )
            parent = parent.parent

    actual_files = {path.resolve() for path in transaction_entries if path.is_file()}
    actual_directories = {path.resolve() for path in transaction_entries if path.is_dir()}
    actual_directories_with_root = actual_directories | {transaction_root.resolve()}
    unexpected_files = actual_files - allowed_files
    unexpected_directories = actual_directories - allowed_directories
    _require(
        not unexpected_files,
        "unplanned recovery transaction file remains: "
        f"{[path.relative_to(root).as_posix() for path in sorted(unexpected_files)[:5]]}",
    )
    _require(
        not unexpected_directories,
        "unplanned recovery transaction directory remains: "
        f"{[path.relative_to(root).as_posix() for path in sorted(unexpected_directories)[:5]]}",
    )
    _require(
        actual_files == allowed_files
        and actual_directories_with_root == allowed_directories,
        "recovery transaction recursive inventory is incomplete",
    )

    decoded_root = root / "decoded"
    _require(
        decoded_root.is_dir() and not _linklike(decoded_root),
        "recovered decoded root is absent or a link",
    )
    decoded_entries = list(decoded_root.rglob("*"))
    decoded_links = [
        path.relative_to(root).as_posix()
        for path in decoded_entries
        if _linklike(path)
    ]
    _require(
        not decoded_links,
        f"link forbidden in recovered decoded inventory: {decoded_links[:5]}",
    )
    _require(
        all(path.is_file() or path.is_dir() for path in decoded_entries),
        "special filesystem object forbidden in recovered decoded inventory",
    )
    progress_root = decoded_root / "recovery_progress"
    history_root = decoded_root / "recovery_history"
    expected_decoded_dirs = {progress_root.resolve(), history_root.resolve()}
    actual_decoded_dirs = {path.resolve() for path in decoded_entries if path.is_dir()}
    _require(
        actual_decoded_dirs == expected_decoded_dirs,
        "recovered decoded directory inventory mismatch",
    )
    progress_counts = recovery_progress_counts(contract)
    expected_progress_paths = {
        progress_root / f"decoded_messages__{attempt_id}__{count:06d}.json"
        for count in progress_counts
    }
    for count, progress_path in zip(progress_counts, sorted(expected_progress_paths)):
        del count  # filename/payload validation below is independent of sort order
        _require(progress_path.is_file(), "recovery progress checkpoint is absent")
    for progress_path in sorted(expected_progress_paths):
        match = re.fullmatch(
            rf"decoded_messages__{re.escape(attempt_id)}__(?P<count>[0-9]{{6}})\.json",
            progress_path.name,
        )
        _require(match is not None, "recovery progress filename mismatch")
        assert match is not None
        checkpoint_count = int(match.group("count"))
        checkpoint = load_json(progress_path)
        _require(
            set(checkpoint)
            == {
                "artifact_type",
                "attempt_id",
                "completed_messages",
                "network_requests",
                "process_model",
                "max_processes",
            },
            f"recovery progress schema mismatch: {progress_path.name}",
        )
        _require(
            checkpoint.get("artifact_type") == "DECODE_RECOVERY_PROGRESS_V2"
            and checkpoint.get("attempt_id") == attempt_id
            and int(checkpoint.get("completed_messages", -1)) == checkpoint_count
            and int(checkpoint.get("network_requests", -1)) == 0
            and checkpoint.get("process_model") == "spawn"
            and int(checkpoint.get("max_processes", -1)) == DECODE_MAX_WORKERS,
            f"recovery progress value mismatch: {progress_path.name}",
        )

    raw_mutex = history_root / f"{attempt_id}__raw_mutex__complete.lock"
    decode_mutex = history_root / f"{attempt_id}__decode_mutex__complete.lock"
    postcommit_path = history_root / f"{attempt_id}__postcommit_input_audit.json"
    history_paths = {raw_mutex, decode_mutex, postcommit_path}
    for history_path in history_paths:
        _require(history_path.is_file(), f"recovery history file absent: {history_path.name}")
    _require(
        raw_mutex.read_bytes() == decode_mutex.read_bytes(),
        "recovery mutex completion locks are not byte-identical",
    )
    mutex = load_json(raw_mutex)
    _require(
        set(mutex)
        == {"artifact_type", "attempt_id", "pid", "created_utc", "network_requests"}
        and mutex.get("artifact_type") == "DECODE_RECOVERY_ACTIVE_LOCK_V2"
        and mutex.get("attempt_id") == attempt_id
        and int(mutex.get("pid", -1)) > 0
        and int(mutex.get("network_requests", -1)) == 0,
        "recovery completion-lock payload mismatch",
    )
    postcommit = load_json(postcommit_path)
    _require(
        set(postcommit)
        == {
            "artifact_type",
            "attempt_id",
            "initial",
            "preplan",
            "postcommit",
            "all_equal",
            "network_requests",
        },
        "recovery postcommit input-audit schema mismatch",
    )
    _require(
        postcommit.get("artifact_type") == "DECODE_RECOVERY_POSTCOMMIT_INPUT_AUDIT_V2"
        and postcommit.get("attempt_id") == attempt_id
        and postcommit.get("all_equal") is True
        and int(postcommit.get("network_requests", -1)) == 0
        and postcommit.get("initial") == postcommit.get("preplan")
        == postcommit.get("postcommit"),
        "recovery fixed-input snapshots differ",
    )
    expected_decoded_files = {
        outputs[name].resolve() for name in ("site", "group", "audit", "lock")
    } | {path.resolve() for path in expected_progress_paths | history_paths}
    actual_decoded_files = {path.resolve() for path in decoded_entries if path.is_file()}
    _require(
        actual_decoded_files == expected_decoded_files,
        "recovered decoded file inventory mismatch",
    )

    launch_root = root / "raw" / "launch_history"
    _require(launch_root.is_dir() and not _linklike(launch_root), "launch history invalid")
    launch_entries = list(launch_root.rglob("*"))
    _require(
        not any(_linklike(path) for path in launch_entries),
        "link forbidden in recovered launch history",
    )
    _require(
        all(path.is_file() for path in launch_entries),
        "unexpected directory/special object in recovered launch history",
    )
    failed_lock = launch_root / f"{failed_attempt}__failed.lock"
    launch_files = {path.resolve() for path in launch_entries if path.is_file()}
    _require(
        launch_files == {failed_lock.resolve()} and failed_lock.is_file(),
        "recovered launch history must preserve only the original failed lock",
    )
    failed_payload = load_json(failed_lock)
    _require(
        set(failed_payload) == {"attempt_id", "pid", "created_utc"}
        and failed_payload.get("attempt_id") == failed_attempt,
        "original failed-lock payload mismatch",
    )
    verify_identity(
        authorization.get("original_failed_lock", {}),
        failed_lock,
        root=root,
        label="authorized original failed lock",
    )

    raw_root = root / "raw"
    expected_raw_top = {
        "RAW_RANGE_MANIFEST.parquet",
        "RAW_RANGE_MANIFEST.csv",
        "RAW_ACCESS_LEDGER.json",
        "ranges",
        "request_events",
        "progress",
        "launch_history",
        "output_transactions",
    }
    raw_top_entries = list(raw_root.iterdir())
    _require(
        not any(_linklike(path) for path in raw_top_entries)
        and {path.name for path in raw_top_entries} == expected_raw_top,
        "recovered raw top-level filesystem inventory mismatch",
    )
    for name in ("RAW_RANGE_MANIFEST.parquet", "RAW_RANGE_MANIFEST.csv", "RAW_ACCESS_LEDGER.json"):
        _require((raw_root / name).is_file(), f"recovered raw canonical file absent: {name}")
    for name in ("ranges", "request_events", "progress", "launch_history", "output_transactions"):
        _require((raw_root / name).is_dir(), f"recovered raw directory absent: {name}")
    leftovers = sorted(
        path.relative_to(root).as_posix()
        for pattern in ("*.part", "*.meta.part.json", "*.writepart", "*.tmp.*")
        for path in raw_root.rglob(pattern)
    )
    _require(not leftovers, f"unfinished recovery staging files remain: {leftovers[:5]}")
    return {
        "outputs": outputs,
        "output_identities": output_identities,
        "plan": plan,
        "plan_identity": identity(plan_path, root),
        "commit": commit,
        "commit_identity": identity(commit_path, root),
        "attempt_id": attempt_id,
        "complete_lock_identity": identity(raw_mutex, root),
        "launch_history_lock_count": len(launch_files),
        "transaction_inventory_recursively_closed": True,
        "planned_staged_files_present": 0,
        "manifest": dict(manifest),
        "recovered": True,
        "recovery_authorization": authorization,
        "recovery_authorization_identity": authorization_identity,
        "recovery_go_identity": go_identity,
        "recovery_incident": incident,
        "recovery_incident_identity": incident_identity,
        "flawed_launch_incident": flawed_launch_incident,
        "flawed_launch_incident_identity": flawed_launch_incident_identity,
        "launch_identity_correction": launch_identity_correction,
        "launch_identity_correction_identity": launch_identity_correction_identity,
        "failed_attempt_id": failed_attempt,
        "recovery_progress_paths": sorted(expected_progress_paths),
        "recovery_history_paths": sorted(history_paths),
        "recovery_postcommit_audit": postcommit,
        "recovery_postcommit_audit_identity": identity(postcommit_path, root),
        "incident_bound_preplan_remnant_count": len(remnants),
    }


def _audit_transaction_and_launch(
    root: Path, contract: AuditContract
) -> dict[str, Any]:
    manifest_path = root / OUTPUT_RELATIVE_PATHS["manifest"]
    require_no_symlink_chain(manifest_path, root, label="producer manifest path")
    _require(not _linklike(manifest_path), "producer manifest link is forbidden")
    manifest = load_json(manifest_path)
    artifact_type = manifest.get("artifact_type")
    if artifact_type == "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST":
        return _audit_original_transaction_and_launch(root)
    if artifact_type == "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST_RECOVERED_V2":
        return _audit_recovery_transaction_and_launch(root, contract, manifest)
    raise AuditFailure(f"unrecognized producer manifest artifact type: {artifact_type!r}")


def _audit_census(path: Path, contract: AuditContract) -> tuple[Any, list[dict[str, Any]]]:
    import pandas as pd

    require_exact_schema(
        parquet_schema_columns(path, label="field census"),
        CENSUS_COLUMNS,
        label="field census",
    )
    frame = pd.read_parquet(path, columns=list(CENSUS_COLUMNS))
    _require(len(frame) == contract.range_rows, "field census range row count mismatch")
    _require(int(frame["range_bytes"].sum()) == contract.range_bytes, "field census byte total mismatch")
    _require((frame["status"] == "CENSUS_VERIFIED").all(), "unverified field census row")
    _require((frame["source_archive"] == "NOAA_NODD_S3").all(), "non-NODD source in census")
    _require((frame["range_bytes"] > 0).all(), "non-positive range in census")
    _require((frame["range_end"] - frame["range_start"] + 1 == frame["range_bytes"]).all(), "census inclusive range arithmetic mismatch")
    _require((frame["range_start"] >= 0).all(), "negative census range start")
    _require((frame["range_end"] < frame["object_size_bytes"]).all(), "census range exceeds object")
    _require((frame["cutoff_margin_seconds"] > 0).all(), "non-causal census publication cutoff")
    cutoff = pd.to_datetime(frame["cutoff_utc"], utc=True, errors="raise")
    publication = pd.to_datetime(
        frame["publication_last_modified_utc"], utc=True, errors="raise"
    )
    observed_margin = (cutoff - publication).dt.total_seconds()
    _require(
        (observed_margin == frame["cutoff_margin_seconds"].astype(float)).all(),
        "census cutoff margin does not equal frozen publication timing",
    )
    _require(
        frame["retrieval_url_or_request_id"].str.startswith(
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        ).all(),
        "census URL is outside frozen official NOAA NODD S3 host",
    )
    keys = ["object_key", "variable", "level"]
    _require(not frame.duplicated(keys).any(), "duplicate field census key")
    _require(frame["object_key"].nunique() == contract.object_rows, "census object count mismatch")
    expected_pairs = set(FEATURE_PAIRS)
    for _object_key, members in frame.groupby("object_key", sort=False):
        observed = set(zip(members["variable"], members["level"]))
        _require(observed == expected_pairs, "object does not contain exactly the frozen nine fields")
    family_counts = frame.groupby("family").size().to_dict()
    _require(
        family_counts
        == {
            "PBL_HEIGHT": contract.object_rows,
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": contract.object_rows * 8,
        },
        "frozen family census counts mismatch",
    )
    dates = pd.to_datetime(frame["target_operating_day_kst"], errors="raise")
    if contract.operating_years is not None:
        _require(set(dates.dt.year) == set(contract.operating_years), "operating-year scope mismatch")
    if contract.operating_days_of_month is not None:
        _require(set(dates.dt.day) == set(contract.operating_days_of_month), "operating-day sample lock mismatch")
    if contract.forecast_hours is not None:
        _require(set(int(v) for v in frame["forecast_hour"]) == set(contract.forecast_hours), "forecast-hour scope mismatch")
    _require(
        not frame["target_operating_day_kst"].str.startswith(("2024", "2025")).any(),
        "field census includes operating-2024/2025 rows",
    )
    return frame, frame.to_dict("records")


def _audit_raw_ranges(
    root: Path,
    census_rows: list[dict[str, Any]],
    raw_manifest_path: Path,
    raw_csv_path: Path,
    contract: AuditContract,
) -> dict[str, Any]:
    import pandas as pd

    raw_schema = parquet_schema_columns(raw_manifest_path, label="raw range manifest")
    _require(
        set(raw_schema)
        in (
            set(RAW_MANIFEST_BASE_COLUMNS),
            set(RAW_MANIFEST_BASE_COLUMNS) | {"resume_prefix_evidence"},
        ),
        "raw range manifest violates exact producer schema",
    )
    manifest_frame = pd.read_parquet(raw_manifest_path, columns=list(raw_schema))
    _require(len(manifest_frame) == contract.range_rows, "raw manifest parquet row count mismatch")
    manifest_rows = manifest_frame.to_dict("records")
    census_by_key = {
        (str(row["object_key"]), str(row["variable"]), str(row["level"])): row
        for row in census_rows
    }
    manifest_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_info: dict[tuple[str, str, str], dict[str, Any]] = {}
    expected_raw_paths: set[Path] = set()
    expected_sidecars: set[Path] = set()
    expected_prefixes: set[Path] = set()
    total_bytes = 0
    flat_same = (
        "source_archive", "archive_product", "target_operating_day_kst", "run_init_utc",
        "forecast_hour", "valid_time_utc", "retrieval_url_or_request_id", "official_metadata",
        "publication_evidence_type", "publication_evidence_reference", "cutoff_utc",
        "object_key", "object_etag", "object_size_bytes", "publication_last_modified_utc",
        "range_start", "range_end", "range_bytes", "family", "variable", "level", "idx_sha256",
    )
    integer_fields = {
        "forecast_hour", "object_size_bytes", "range_start", "range_end", "range_bytes",
        "raw_size_bytes", "http_status", "request_range_start", "resumed_from_bytes",
    }
    for parquet_row in manifest_rows:
        key = (
            str(parquet_row.get("object_key")),
            str(parquet_row.get("variable")),
            str(parquet_row.get("level")),
        )
        _require(key in census_by_key, f"raw manifest row is outside field census: {key}")
        _require(key not in manifest_by_key, f"duplicate raw manifest key: {key}")
        census = census_by_key[key]
        relative = raw_relative_path(census)
        _require(parquet_row.get("raw_filename") == relative, f"raw filename mismatch: {key}")
        raw_path = path_under(root, relative)
        sidecar_path = raw_path.with_suffix(".grib2.meta.json")
        require_no_symlink_chain(raw_path, root, label="raw range path")
        require_no_symlink_chain(sidecar_path, root, label="raw sidecar path")
        _require(raw_path.is_file() and sidecar_path.is_file(), f"raw/sidecar pair absent: {relative}")
        _require(not _linklike(raw_path) and not _linklike(sidecar_path), f"link forbidden in raw closure: {relative}")
        sidecar = load_json(sidecar_path)
        allowed_sidecar_keys = set(RAW_MANIFEST_BASE_COLUMNS)
        if int(sidecar.get("resumed_from_bytes", 0)) > 0:
            allowed_sidecar_keys.add("resume_prefix_evidence")
        _require(
            set(sidecar) == allowed_sidecar_keys,
            f"raw sidecar violates exact producer schema: {key}",
        )
        request_evidence_schema = sidecar.get("request_completion_evidence")
        _require(
            isinstance(request_evidence_schema, Mapping)
            and set(request_evidence_schema)
            == {
                "semantically_identical_completion_event_count",
                "selected_start_event",
                "selected_complete_event",
            },
            f"raw sidecar request evidence schema mismatch: {key}",
        )
        for identity_name in ("selected_start_event", "selected_complete_event"):
            _require(
                isinstance(request_evidence_schema[identity_name], Mapping)
                and set(request_evidence_schema[identity_name])
                == {"path", "size_bytes", "sha256"},
                f"raw sidecar request identity schema mismatch: {key}/{identity_name}",
            )
        if "resume_prefix_evidence" in sidecar:
            prefix_schema = sidecar["resume_prefix_evidence"]
            _require(
                isinstance(prefix_schema, Mapping)
                and set(prefix_schema)
                == {
                    "object_key", "object_etag", "publication_last_modified_utc",
                    "range_start", "range_end", "prefix_size_bytes", "prefix_sha256",
                    "retrieved_at", "start_event", "complete_event",
                },
                f"raw sidecar prefix evidence schema mismatch: {key}",
            )
            for identity_name in ("start_event", "complete_event"):
                _require(
                    isinstance(prefix_schema[identity_name], Mapping)
                    and set(prefix_schema[identity_name])
                    == {"path", "size_bytes", "sha256"},
                    f"raw sidecar prefix identity schema mismatch: {key}/{identity_name}",
                )
        for field in flat_same:
            expected = census[field]
            observed = sidecar.get(field)
            if field in integer_fields:
                _require(int(observed) == int(expected), f"sidecar/census mismatch {field}: {key}")
            else:
                _require(observed == expected, f"sidecar/census mismatch {field}: {key}")
        _require(int(sidecar.get("cutoff_margin", -1)) == int(census["cutoff_margin_seconds"]), f"sidecar cutoff margin mismatch: {key}")
        _require(sidecar.get("status") == "VERIFIED", f"raw sidecar status mismatch: {key}")
        _require(int(sidecar.get("http_status", -1)) == 206, f"raw HTTP status mismatch: {key}")
        expected_size = int(census["range_bytes"])
        _require(raw_path.stat().st_size == expected_size, f"raw size mismatch: {relative}")
        with raw_path.open("rb") as stream:
            _require(stream.read(4) == b"GRIB", f"raw GRIB magic mismatch: {relative}")
            stream.seek(-4, 2)
            _require(stream.read(4) == b"7777", f"raw GRIB terminator mismatch: {relative}")
        raw_sha = sha256_file(raw_path)
        _require(
            sidecar.get("raw_sha256") == raw_sha
            and int(sidecar.get("raw_size_bytes", -1)) == expected_size,
            f"raw bytes differ from sidecar: {relative}",
        )
        resumed = int(sidecar.get("resumed_from_bytes", -1))
        request_start = int(sidecar.get("request_range_start", -1))
        _require(0 <= resumed < expected_size, f"invalid resumed byte count: {key}")
        _require(request_start == int(census["range_start"]) + resumed, f"request start/resume mismatch: {key}")
        _require(
            sidecar.get("content_range")
            == f"bytes {request_start}-{int(census['range_end'])}/{int(census['object_size_bytes'])}",
            f"sidecar Content-Range mismatch: {key}",
        )
        if resumed:
            _require(isinstance(sidecar.get("resume_prefix_evidence"), Mapping), f"resumed sidecar lacks prefix evidence: {key}")
            expected_prefixes.add(raw_path.with_suffix(".grib2.prefix.json"))
        else:
            _require("resume_prefix_evidence" not in sidecar, f"zero-resume sidecar has prefix evidence: {key}")
        evidence = sidecar.get("request_completion_evidence")
        _require(isinstance(evidence, Mapping), f"sidecar lacks request completion evidence: {key}")
        for field, value in sidecar.items():
            if isinstance(value, Mapping):
                continue
            parquet_value = parquet_row.get(field)
            if field in integer_fields or isinstance(value, int):
                _require(int(parquet_value) == int(value), f"raw parquet/sidecar mismatch {field}: {key}")
            elif isinstance(value, float):
                _require(_same_number(parquet_value, value), f"raw parquet/sidecar mismatch {field}: {key}")
            else:
                _require(parquet_value == value, f"raw parquet/sidecar mismatch {field}: {key}")
        for nested_field in (
            "request_completion_evidence",
            "resume_prefix_evidence",
        ):
            observed_nested = normalize_nested_value(parquet_row.get(nested_field))
            expected_nested = normalize_nested_value(sidecar.get(nested_field))
            _require(
                observed_nested == expected_nested,
                f"raw parquet/sidecar nested mismatch {nested_field}: {key}",
            )
        manifest_by_key[key] = parquet_row
        raw_info[key] = {
            "row": census,
            "meta": sidecar,
            "path": raw_path,
            "sha256": raw_sha,
            "size_bytes": expected_size,
        }
        expected_raw_paths.add(raw_path.resolve())
        expected_sidecars.add(sidecar_path.resolve())
        total_bytes += expected_size
    _require(set(manifest_by_key) == set(census_by_key), "raw manifest/census key closure mismatch")
    _require(total_bytes == contract.range_bytes, "rehash raw byte total mismatch")

    range_root = root / "raw" / "ranges"
    _require(range_root.is_dir(), "raw range directory is absent")
    expected_range_files = expected_raw_paths | expected_sidecars | expected_prefixes
    allowed_range_directories = {range_root.resolve()}
    for expected_path in expected_range_files:
        parent = expected_path.parent.resolve()
        while True:
            allowed_range_directories.add(parent)
            if parent == range_root.resolve():
                break
            _require(
                range_root.resolve() in parent.parents,
                "expected raw range path escapes range root",
            )
            parent = parent.parent
    range_entries = list(range_root.rglob("*"))
    range_symlinks = [
        path.relative_to(root).as_posix()
        for path in range_entries
        if _linklike(path)
    ]
    _require(
        not range_symlinks,
        f"symlink forbidden in raw range inventory: {range_symlinks[:5]}",
    )
    _require(
        all(path.is_file() or path.is_dir() for path in range_entries),
        "special filesystem object forbidden in raw range inventory",
    )
    actual_range_files = {
        path.resolve() for path in range_entries if path.is_file()
    }
    unexpected_range_files = sorted(actual_range_files - expected_range_files)
    missing_range_files = sorted(expected_range_files - actual_range_files)
    _require(
        not unexpected_range_files,
        "unrecognized file in raw range inventory: "
        f"{[path.relative_to(root).as_posix() for path in unexpected_range_files[:5]]}",
    )
    _require(
        not missing_range_files,
        "expected file absent from raw range inventory: "
        f"{[path.relative_to(root).as_posix() for path in missing_range_files[:5]]}",
    )
    unexpected_range_directories = [
        path.relative_to(root).as_posix()
        for path in range_entries
        if path.is_dir() and path.resolve() not in allowed_range_directories
    ]
    _require(
        not unexpected_range_directories,
        f"unrecognized directory in raw range inventory: {unexpected_range_directories[:5]}",
    )

    with raw_csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            tuple(reader.fieldnames or ()) == raw_schema,
            "raw manifest CSV violates exact producer schema/order",
        )
        csv_rows = list(reader)
    _require(len(csv_rows) == contract.range_rows, "raw manifest CSV row count mismatch")
    csv_by_path = {row.get("raw_filename", ""): row for row in csv_rows}
    _require(len(csv_by_path) == len(csv_rows), "duplicate raw filename in CSV manifest")
    _require(set(csv_by_path) == {info["meta"]["raw_filename"] for info in raw_info.values()}, "raw CSV path closure mismatch")
    for info in raw_info.values():
        meta = info["meta"]
        csv_row = csv_by_path[meta["raw_filename"]]
        for field in raw_schema:
            observed = csv_row.get(field)
            expected = meta.get(field)
            if expected is None:
                _require(observed == "", f"raw CSV mismatch {field}: {meta['raw_filename']}")
            elif isinstance(expected, Mapping):
                try:
                    parsed = ast.literal_eval(str(observed))
                except (SyntaxError, ValueError) as exc:
                    raise AuditFailure(
                        f"raw CSV nested field is malformed {field}: {meta['raw_filename']}"
                    ) from exc
                _require(
                    normalize_nested_value(parsed) == normalize_nested_value(expected),
                    f"raw CSV mismatch {field}: {meta['raw_filename']}",
                )
            elif isinstance(expected, bool):
                _require(observed == str(expected), f"raw CSV mismatch {field}: {meta['raw_filename']}")
            elif isinstance(expected, int):
                _require(int(str(observed)) == expected, f"raw CSV mismatch {field}: {meta['raw_filename']}")
            elif isinstance(expected, float):
                _require(_same_number(observed, expected), f"raw CSV mismatch {field}: {meta['raw_filename']}")
            else:
                _require(observed == str(expected), f"raw CSV mismatch {field}: {meta['raw_filename']}")
    raw_info_in_census_order = {
        (
            str(row["object_key"]),
            str(row["variable"]),
            str(row["level"]),
        ): raw_info[
            (
                str(row["object_key"]),
                str(row["variable"]),
                str(row["level"]),
            )
        ]
        for row in census_rows
    }
    return {
        "raw_info": raw_info_in_census_order,
        "raw_rows": contract.range_rows,
        "raw_bytes": total_bytes,
    }


def _audit_request_events(
    root: Path,
    raw_info: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    event_root = root / "raw" / "request_events"
    _require(event_root.is_dir(), "request event directory is absent")
    event_entries = list(event_root.rglob("*"))
    event_symlinks = [
        path.relative_to(root).as_posix()
        for path in event_entries
        if _linklike(path)
    ]
    _require(
        not event_symlinks,
        f"symlink forbidden in request event inventory: {event_symlinks[:5]}",
    )
    _require(
        all(path.is_file() for path in event_entries),
        "request event inventory contains a directory/special object",
    )
    event_directories = [
        path.relative_to(root).as_posix()
        for path in event_entries
        if path.is_dir()
    ]
    _require(
        not event_directories,
        f"unexpected directory in request event inventory: {event_directories[:5]}",
    )
    paths = sorted(path for path in event_entries if path.is_file())
    _require(paths, "request event inventory is empty")
    records: dict[tuple[str, int], dict[str, tuple[Path, dict[str, Any]]]] = {}
    rows_by_url: dict[str, list[tuple[tuple[str, str, str], Mapping[str, Any]]]] = {}
    for key, info in raw_info.items():
        rows_by_url.setdefault(str(info["row"]["retrieval_url_or_request_id"]), []).append((key, info))
    for path in paths:
        match = EVENT_NAME.fullmatch(path.name)
        _require(match is not None, f"unexpected request event filename: {path.name}")
        assert match is not None
        event_id = match.group("event_id")
        attempt = int(match.group("attempt"))
        kind = match.group("kind")
        payload = load_json(path)
        _require(payload.get("event_id") == event_id, f"event ID payload/path mismatch: {path.name}")
        _require(int(payload.get("attempt", -1)) == attempt, f"event attempt payload/path mismatch: {path.name}")
        slot = records.setdefault((event_id, attempt), {})
        _require(kind not in slot, f"duplicate request event kind: {path.name}")
        slot[kind] = (path, payload)

    global_numbers: list[int] = []
    completed = 0
    errors = 0
    indeterminate = 0
    completion_by_path: dict[Path, tuple[tuple[str, str, str], dict[str, Any], dict[str, Any]]] = {}
    start_by_path: dict[Path, tuple[tuple[str, str, str], dict[str, Any]]] = {}
    attempts_by_event: dict[str, list[int]] = {}
    attempt_inventory: list[dict[str, Any]] = []
    for (event_id, attempt), slot in sorted(records.items()):
        _require(
            set(slot) in ({"start"}, {"start", "complete"}, {"start", "error"}),
            f"request attempt has an invalid terminal-outcome inventory: {event_id}/{attempt}",
        )
        start_path, start = slot["start"]
        _require(
            set(start)
            == {
                "event", "event_id", "attempt", "global_raw_attempt_number",
                "url", "range_start", "range_end", "created_utc",
            },
            f"request start payload schema mismatch: {start_path.name}",
        )
        _require(start.get("event") == "RANGE_REQUEST_START", f"request start event type mismatch: {start_path.name}")
        number = int(start.get("global_raw_attempt_number", -1))
        _require(number > 0, f"invalid global request attempt number: {start_path.name}")
        global_numbers.append(number)
        url = str(start.get("url", ""))
        start_byte = int(start.get("range_start", -1))
        end_byte = int(start.get("range_end", -1))
        candidates = [
            (key, info)
            for key, info in rows_by_url.get(url, [])
            if int(info["row"]["range_start"]) <= start_byte <= end_byte <= int(info["row"]["range_end"])
        ]
        _require(len(candidates) == 1, f"request event does not map to one frozen range: {start_path.name}")
        key, info = candidates[0]
        row = info["row"]
        _require(event_id_for(row, start_byte, end_byte) == event_id, f"request event hash identity mismatch: {start_path.name}")
        start_by_path[start_path.resolve()] = (key, start)
        attempts_by_event.setdefault(event_id, []).append(attempt)
        outcome_kind = (
            "complete" if "complete" in slot else "error" if "error" in slot else None
        )
        if outcome_kind is None:
            # A hard process death can occur after the durable START but before
            # either terminal event.  The producer conservatively counts that
            # attempt on restart.  It is therefore valid provenance, not a free
            # or uncounted retry, provided a later exact completion closes the
            # raw range.
            indeterminate += 1
            attempt_inventory.append(
                {
                    "global_raw_attempt_number": number,
                    "event_id": event_id,
                    "local_attempt": attempt,
                    "outcome": "START_ONLY",
                    "complete_payload_bytes": 0,
                    "raw_key": list(key),
                    "start_created_utc": start["created_utc"],
                }
            )
            continue
        outcome_path, outcome = slot[outcome_kind]
        expected_outcome_fields = (
            {
                "event", "event_id", "attempt", "global_raw_attempt_number",
                "http_status", "range_start", "range_end", "content_range",
                "etag", "last_modified_utc", "bytes", "payload_sha256",
                "retrieved_at",
            }
            if outcome_kind == "complete"
            else {
                "event", "event_id", "attempt", "global_raw_attempt_number",
                "error_type", "error", "recorded_utc",
            }
        )
        _require(
            set(outcome) == expected_outcome_fields,
            f"request outcome payload schema mismatch: {outcome_path.name}",
        )
        expected_event = "RANGE_REQUEST_COMPLETE" if outcome_kind == "complete" else "RANGE_REQUEST_ERROR"
        _require(outcome.get("event") == expected_event, f"request outcome event type mismatch: {outcome_path.name}")
        _require(outcome.get("global_raw_attempt_number") == number, f"request start/outcome global number mismatch: {outcome_path.name}")
        if outcome_kind == "error":
            errors += 1
            _require(bool(outcome.get("error_type")), f"request error type absent: {outcome_path.name}")
            attempt_inventory.append(
                {
                    "global_raw_attempt_number": number,
                    "event_id": event_id,
                    "local_attempt": attempt,
                    "outcome": "ERROR",
                    "complete_payload_bytes": 0,
                    "raw_key": list(key),
                    "start_created_utc": start["created_utc"],
                }
            )
            continue
        completed += 1
        byte_count = end_byte - start_byte + 1
        _require(
            int(outcome.get("http_status", -1)) == 206
            and int(outcome.get("range_start", -1)) == start_byte
            and int(outcome.get("range_end", -1)) == end_byte
            and int(outcome.get("bytes", -1)) == byte_count,
            f"completed request range/status mismatch: {outcome_path.name}",
        )
        _require(
            outcome.get("content_range")
            == f"bytes {start_byte}-{end_byte}/{int(row['object_size_bytes'])}"
            and outcome.get("etag") == row["object_etag"]
            and outcome.get("last_modified_utc") == row["publication_last_modified_utc"],
            f"completed request frozen object identity mismatch: {outcome_path.name}",
        )
        offset = start_byte - int(row["range_start"])
        if offset == 0 and byte_count == int(info["size_bytes"]):
            payload_sha = str(info["sha256"])
        else:
            payload_sha = sha256_segment(Path(info["path"]), offset, byte_count)
        _require(outcome.get("payload_sha256") == payload_sha, f"completed request payload SHA mismatch: {outcome_path.name}")
        attempt_inventory.append(
            {
                "global_raw_attempt_number": number,
                "event_id": event_id,
                "local_attempt": attempt,
                "outcome": "COMPLETE",
                "complete_payload_bytes": byte_count,
                "raw_key": list(key),
                "start_created_utc": start["created_utc"],
            }
        )
        completion_by_path[outcome_path.resolve()] = (key, start, outcome)

    for event_id, attempts in attempts_by_event.items():
        ordered = sorted(attempts)
        _require(ordered == list(range(1, max(ordered) + 1)), f"non-contiguous local attempts: {event_id}")
        outcome_sequence = [
            "COMPLETE"
            if "complete" in records[(event_id, attempt)]
            else "ERROR"
            if "error" in records[(event_id, attempt)]
            else "START_ONLY"
            for attempt in ordered
        ]
        _require(
            outcome_sequence[-1] == "COMPLETE",
            f"request event sequence is not closed by a final completion: {event_id}",
        )
        local_global_numbers = [
            int(records[(event_id, attempt)]["start"][1]["global_raw_attempt_number"])
            for attempt in ordered
        ]
        _require(
            all(
                later > earlier
                for earlier, later in zip(
                    local_global_numbers, local_global_numbers[1:]
                )
            ),
            f"global attempt numbers do not increase with local attempts: {event_id}",
        )
    starts = len(global_numbers)
    _require(len(set(global_numbers)) == starts, "duplicate global raw HTTP-attempt number")
    _require(set(global_numbers) == set(range(1, starts + 1)), "non-contiguous global raw HTTP-attempt inventory")
    _require(starts <= contract.raw_attempt_cap, "raw actual HTTP-attempt cap exceeded")
    _require(completed >= contract.range_rows, "fewer completed requests than frozen raw ranges")
    _require(
        completed + errors + indeterminate == starts,
        "request terminal outcome arithmetic mismatch",
    )

    for key, info in raw_info.items():
        row = info["row"]
        meta = info["meta"]
        evidence = meta["request_completion_evidence"]
        selected_start_record = evidence.get("selected_start_event", {})
        selected_complete_record = evidence.get("selected_complete_event", {})
        selected_start_path = path_under(root, str(selected_start_record.get("path", "")))
        selected_complete_path = path_under(root, str(selected_complete_record.get("path", "")))
        verify_identity(selected_start_record, selected_start_path, root=root, label=f"selected request start {key}")
        verify_identity(selected_complete_record, selected_complete_path, root=root, label=f"selected request complete {key}")
        _require(selected_start_path.resolve() in start_by_path, f"selected start is outside event inventory: {key}")
        _require(selected_complete_path.resolve() in completion_by_path, f"selected completion is outside event inventory: {key}")
        complete_key, start, complete = completion_by_path[selected_complete_path.resolve()]
        _require(complete_key == key, f"selected completion maps to another census row: {key}")
        selected_start_key, selected_start = start_by_path[selected_start_path.resolve()]
        _require(selected_start_key == key and selected_start["event_id"] == complete["event_id"] and selected_start["attempt"] == complete["attempt"], f"selected request start/complete pair mismatch: {key}")
        request_start = int(meta["request_range_start"])
        _require(int(start["range_start"]) == request_start and int(start["range_end"]) == int(row["range_end"]), f"selected request does not cover final suffix: {key}")
        semantic_matches = [
            candidate_complete
            for completion_key, candidate_start, candidate_complete in completion_by_path.values()
            if completion_key == key
            and int(candidate_start["range_start"]) == request_start
            and int(candidate_start["range_end"]) == int(row["range_end"])
            and candidate_complete.get("payload_sha256") == complete.get("payload_sha256")
        ]
        semantic_count = len(semantic_matches)
        _require(int(evidence.get("semantically_identical_completion_event_count", -1)) == semantic_count, f"semantic completion-event count mismatch: {key}")
        _require(
            meta.get("retrieved_at")
            in {candidate.get("retrieved_at") for candidate in semantic_matches},
            f"sidecar retrieval time lacks a semantic completion event: {key}",
        )

        resumed = int(meta["resumed_from_bytes"])
        if resumed:
            prefix_path = Path(info["path"]).with_suffix(".grib2.prefix.json")
            prefix = load_json(prefix_path)
            _require(prefix == meta["resume_prefix_evidence"], f"prefix sidecar/meta evidence mismatch: {key}")
            _require(
                int(prefix.get("prefix_size_bytes", -1)) == resumed
                and int(prefix.get("range_start", -1)) == int(row["range_start"])
                and int(prefix.get("range_end", -1)) == int(row["range_start"]) + resumed - 1
                and prefix.get("prefix_sha256") == sha256_segment(Path(info["path"]), 0, resumed)
                and prefix.get("object_key") == row["object_key"]
                and prefix.get("object_etag") == row["object_etag"]
                and prefix.get("publication_last_modified_utc") == row["publication_last_modified_utc"],
                f"resume prefix evidence mismatch: {key}",
            )
            for kind, inventory in (("start_event", start_by_path), ("complete_event", completion_by_path)):
                record = prefix.get(kind, {})
                event_path = path_under(root, str(record.get("path", "")))
                verify_identity(record, event_path, root=root, label=f"prefix {kind} {key}")
                _require(event_path.resolve() in inventory, f"prefix {kind} outside event inventory: {key}")
            prefix_start_key, prefix_start = start_by_path[
                path_under(root, str(prefix["start_event"]["path"])).resolve()
            ]
            prefix_complete_key, _, prefix_complete = completion_by_path[
                path_under(root, str(prefix["complete_event"]["path"])).resolve()
            ]
            _require(
                prefix_start_key == prefix_complete_key == key
                and prefix_start["event_id"] == prefix_complete["event_id"]
                and prefix_start["attempt"] == prefix_complete["attempt"]
                and int(prefix_start["range_start"]) == int(row["range_start"])
                and int(prefix_start["range_end"])
                == int(row["range_start"]) + resumed - 1,
                f"resume prefix start/completion pair mismatch: {key}",
            )
            _require(
                prefix_complete.get("retrieved_at") == prefix.get("retrieved_at"),
                f"resume prefix completion/retrieval time mismatch: {key}",
            )

    identities = [identity(path, root) for path in paths]
    return {
        "starts": starts,
        "completions": completed,
        "errors": errors,
        "indeterminate_starts": indeterminate,
        "event_identities": identities,
        "max_global_attempt_number": max(global_numbers),
        "attempt_inventory": sorted(
            attempt_inventory,
            key=lambda record: int(record["global_raw_attempt_number"]),
        ),
    }


def _audit_decoded(
    root: Path,
    census_frame: Any,
    coordinate: Mapping[str, Any],
    outputs: Mapping[str, Path],
    contract: AuditContract,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    site_columns = SITE_METADATA_COLUMNS + FEATURE_COLUMNS
    group_columns = GROUP_METADATA_COLUMNS + FEATURE_COLUMNS
    require_exact_schema(
        parquet_schema_columns(outputs["site"], label="decoded site matrix"),
        site_columns,
        label="decoded site matrix",
    )
    require_exact_schema(
        parquet_schema_columns(outputs["group"], label="decoded group matrix"),
        group_columns,
        label="decoded group matrix",
    )
    site = pd.read_parquet(outputs["site"], columns=list(site_columns))
    group = pd.read_parquet(outputs["group"], columns=list(group_columns))
    _require(len(site) == contract.site_rows, "decoded site row count mismatch")
    _require(len(group) == contract.group_rows, "decoded group row count mismatch")
    _require(not site.duplicated(["valid_time_utc", "site_id"]).any(), "duplicate decoded site key")
    _require(not group.duplicated(["valid_time_utc", "group"]).any(), "duplicate decoded group key")

    time_columns = ["valid_time_utc", "target_operating_day_kst", "run_init_utc", "forecast_hour"]
    census_times = census_frame[time_columns].drop_duplicates()
    _require(len(census_times) == contract.object_rows, "census time/object key mismatch")
    expected_time_records = {
        tuple(row[column] for column in time_columns)
        for row in census_times.to_dict("records")
    }
    site_time_records = {
        (str(row.valid_time_utc), str(row.target_operating_day_kst), str(row.run_init_utc), int(row.forecast_hour))
        for row in site.itertuples(index=False)
    }
    group_time_records = {
        (str(row.valid_time_utc), str(row.target_operating_day_kst), str(row.run_init_utc), int(row.forecast_hour))
        for row in group.itertuples(index=False)
    }
    expected_time_records = {
        (str(valid), str(day), str(run), int(hour)) for valid, day, run, hour in expected_time_records
    }
    _require(site_time_records == expected_time_records, "decoded site/census time closure mismatch")
    _require(group_time_records == expected_time_records, "decoded group/census time closure mismatch")

    sites = coordinate.get("sites")
    _require(isinstance(sites, list) and len(sites) == contract.site_count, "coordinate site count mismatch")
    coordinate_by_id = {int(row["site_id"]): row for row in sites}
    _require(len(coordinate_by_id) == contract.site_count, "duplicate coordinate site ID")
    _require(set(int(value) for value in site["site_id"]) == set(coordinate_by_id), "decoded site ID closure mismatch")
    for row in site.itertuples(index=False):
        expected = coordinate_by_id[int(row.site_id)]
        _require(str(row.group) == expected["group"], "decoded site group differs from coordinate lock")
        for field in ("latitude", "longitude", "capacity_mw"):
            _require(_same_number(getattr(row, field), expected[field]), f"decoded site {field} differs from coordinate lock")

    group_contract = {item.name: item for item in contract.groups}
    _require(set(str(value) for value in group["group"]) == set(group_contract), "decoded group name closure mismatch")
    indexed_sites = site.set_index(["valid_time_utc", "group"])
    for row in group.itertuples(index=False):
        expected = group_contract[str(row.group)]
        _require(int(row.site_count) == expected.site_count, "decoded group site-count mismatch")
        _require(_same_number(row.capacity_mw, expected.capacity_mw), "decoded group capacity mismatch")
        members = indexed_sites.loc[(row.valid_time_utc, row.group)]
        if isinstance(members, pd.Series):
            members = members.to_frame().T
        _require(len(members) == expected.site_count, "decoded group membership count mismatch")
        total_capacity = float(members["capacity_mw"].astype(float).sum())
        _require(_same_number(total_capacity, expected.capacity_mw), "site-derived group capacity mismatch")
        for feature in FEATURE_COLUMNS:
            values = members[feature].to_numpy(dtype=float)
            capacities = members["capacity_mw"].to_numpy(dtype=float)
            independently_aggregated = (
                float(np.sum(values * capacities) / total_capacity)
                if np.isfinite(values).all()
                else float("nan")
            )
            _require(
                _same_number(getattr(row, feature), independently_aggregated, tolerance=1e-10),
                f"decoded group capacity-weighted aggregation mismatch: {row.group}/{feature}",
            )

    hpbl = site["HPBL_surface"].to_numpy(dtype=float)
    vertical = site[list(FEATURE_COLUMNS[1:])].to_numpy(dtype=float)
    pbl_finite_fraction = float(np.isfinite(hpbl).mean())
    wind_finite_fraction = float(np.isfinite(vertical).mean())
    pbl_pass = bool(np.isfinite(hpbl).all() and np.all((hpbl >= 0.0) & (hpbl <= 10_000.0)))
    wind_pass = bool(np.isfinite(vertical).all() and np.all(np.abs(vertical) <= 150.0))
    pbl_min = float(np.nanmin(hpbl))
    pbl_max = float(np.nanmax(hpbl))
    wind_max_abs = float(np.nanmax(np.abs(vertical)))

    physical = load_json(outputs["audit"])
    physical_base_keys = {
        "expected_site_rows", "observed_site_rows", "expected_group_rows",
        "observed_group_rows", "duplicate_site_keys", "duplicate_group_keys",
        "families", "independent_family_salvage_applied_exactly", "labels_read",
        "models_fit", "final_free_disk_before_transaction_bytes",
        "final_200gb_reserve_pass",
    }
    recovered_physical = (
        physical.get("artifact_type")
        == "TRACK_A_PHYSICAL_AND_COVERAGE_AUDIT_RECOVERED_V2"
    )
    expected_physical_keys = (
        physical_base_keys
        | {
            "artifact_type",
            "recovery_provenance",
            "decode_process_model",
            "decode_worker_pids",
            "decode_process_count",
        }
        if recovered_physical
        else physical_base_keys
    )
    _require(
        set(physical) == expected_physical_keys,
        "physical/coverage audit payload schema mismatch",
    )
    if recovered_physical:
        worker_pids = physical.get("decode_worker_pids")
        _require(
            physical.get("decode_process_model")
            == "WINDOWS_SPAWN_PROCESS_ISOLATION_SERIAL_ECCODES_PER_PROCESS"
            and isinstance(worker_pids, list)
            and len(worker_pids) == DECODE_MAX_WORKERS
            and len({int(pid) for pid in worker_pids}) == DECODE_MAX_WORKERS
            and all(int(pid) > 0 for pid in worker_pids)
            and int(physical.get("decode_process_count", -1))
            == DECODE_MAX_WORKERS
            and isinstance(physical.get("recovery_provenance"), Mapping),
            "recovered physical audit process/provenance evidence mismatch",
        )
    require_zero_facts(physical, {"labels_read": False, "models_fit": 0}, label="physical/coverage audit")
    for key, expected in {
        "expected_site_rows": contract.site_rows,
        "observed_site_rows": contract.site_rows,
        "expected_group_rows": contract.group_rows,
        "observed_group_rows": contract.group_rows,
        "duplicate_site_keys": 0,
        "duplicate_group_keys": 0,
    }.items():
        _require(int(physical.get(key, -1)) == expected, f"physical audit {key} mismatch")
    _require(physical.get("independent_family_salvage_applied_exactly") is True, "independent family salvage flag mismatch")
    families = physical.get("families")
    _require(isinstance(families, Mapping) and set(families) == {"PBL_HEIGHT", "LOW_LEVEL_ISOBARIC_WIND_PROFILE"}, "physical family map mismatch")
    pbl_record = families["PBL_HEIGHT"]
    wind_record = families["LOW_LEVEL_ISOBARIC_WIND_PROFILE"]
    _require(
        isinstance(pbl_record, Mapping)
        and set(pbl_record)
        == {"finite_fraction", "minimum", "maximum", "physical_gate_pass", "status"}
        and isinstance(wind_record, Mapping)
        and set(wind_record)
        == {
            "finite_fraction", "maximum_absolute_component_mps",
            "physical_gate_pass", "status",
        },
        "physical family payload schema mismatch",
    )
    _require(_same_number(pbl_record.get("finite_fraction"), pbl_finite_fraction), "PBL finite fraction mismatch")
    _require(_same_number(pbl_record.get("minimum"), pbl_min), "PBL minimum mismatch")
    _require(_same_number(pbl_record.get("maximum"), pbl_max), "PBL maximum mismatch")
    _require(pbl_record.get("physical_gate_pass") is pbl_pass, "PBL physical gate mismatch")
    _require(pbl_record.get("status") == ("PASS" if pbl_pass else "REJECT_PBL_HEIGHT_ONLY"), "PBL status mismatch")
    _require(_same_number(wind_record.get("finite_fraction"), wind_finite_fraction), "wind finite fraction mismatch")
    _require(_same_number(wind_record.get("maximum_absolute_component_mps"), wind_max_abs), "wind maximum component mismatch")
    _require(wind_record.get("physical_gate_pass") is wind_pass, "wind physical gate mismatch")
    _require(wind_record.get("status") == ("PASS" if wind_pass else "REJECT_LOW_LEVEL_ISOBARIC_WIND_PROFILE_ONLY"), "wind status mismatch")
    _require(physical.get("final_200gb_reserve_pass") is True, "producer final disk reserve gate was not passed")
    _require(int(physical.get("final_free_disk_before_transaction_bytes", -1)) >= 200_000_000_000, "recorded final disk reserve is below 200 GB")
    return {
        "site_rows": len(site),
        "group_rows": len(group),
        "families": {
            "PBL_HEIGHT": {"pass": pbl_pass, "finite_fraction": pbl_finite_fraction, "minimum": pbl_min, "maximum": pbl_max},
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": {"pass": wind_pass, "finite_fraction": wind_finite_fraction, "maximum_absolute_component_mps": wind_max_abs},
        },
        "downstream_target_free_estimator_may_start": pbl_pass or wind_pass,
        "physical_payload": physical,
        "recovered": recovered_physical,
    }


def offline_bilinear_regular_ll(
    values: Any,
    latitude: float,
    longitude: float,
    *,
    ni: int,
    nj: int,
    first_latitude: float,
    first_longitude: float,
    di: float,
    dj: float,
    missing_value: float,
) -> float:
    """Independent regular-lat/lon interpolation used only by the auditor."""

    import numpy as np

    grid = np.asarray(values, dtype=float).reshape((nj, ni))
    fractional_x = ((float(longitude) - first_longitude) % 360.0) / di
    fractional_y = (first_latitude - float(latitude)) / dj
    _require(0.0 <= fractional_y <= nj - 1, "audit site latitude is outside GRIB grid")
    x_floor = math.floor(fractional_x)
    y_floor = math.floor(fractional_y)
    i0 = int(x_floor) % ni
    i1 = (i0 + 1) % ni
    j0 = min(int(y_floor), nj - 1)
    j1 = min(j0 + 1, nj - 1)
    wx = fractional_x - x_floor
    wy = fractional_y - y_floor
    corners = np.asarray(
        (grid[j0, i0], grid[j0, i1], grid[j1, i0], grid[j1, i1]),
        dtype=float,
    )
    if (
        not np.isfinite(corners).all()
        or np.any(corners == float(missing_value))
        or np.any(np.abs(corners) > 1e19)
    ):
        return float("nan")
    north = float(corners[0] * (1.0 - wx) + corners[1] * wx)
    south = float(corners[2] * (1.0 - wx) + corners[3] * wx)
    return float(north * (1.0 - wy) + south * wy)


def decode_offline_grib_message(
    info: Mapping[str, Any], sites: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Decode one already-local GRIB message; no producer import or network path."""

    import eccodes
    import numpy as np

    path = Path(info["path"])
    row = info["row"]
    with path.open("rb") as stream:
        handle = eccodes.codes_grib_new_from_file(stream)
        _require(handle is not None, f"offline ecCodes could not open one message: {path}")
        assert handle is not None
        try:
            key_names = (
                "edition", "discipline", "parameterCategory", "parameterNumber",
                "shortName", "typeOfLevel", "level", "dataDate", "dataTime",
                "forecastTime", "stepUnits", "indicatorOfUnitOfTimeRange",
                "validityDate", "validityTime", "gridType",
                "gridDefinitionTemplateNumber", "Ni", "Nj", "numberOfDataPoints",
                "latitudeOfFirstGridPointInDegrees",
                "longitudeOfFirstGridPointInDegrees",
                "latitudeOfLastGridPointInDegrees",
                "longitudeOfLastGridPointInDegrees",
                "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees",
                "iScansNegatively", "jScansPositively", "jPointsAreConsecutive",
                "alternativeRowScanning", "missingValue",
            )
            keys = {name: eccodes.codes_get(handle, name) for name in key_names}
            values = np.asarray(eccodes.codes_get_array(handle, "values"), dtype=float)
        finally:
            eccodes.codes_release(handle)
        second = eccodes.codes_grib_new_from_file(stream)
        if second is not None:
            try:
                raise AuditFailure(f"raw range contains more than one GRIB message: {path}")
            finally:
                eccodes.codes_release(second)

    exact_grid = {
        "edition": 2,
        "gridType": "regular_ll",
        "gridDefinitionTemplateNumber": 0,
        "Ni": 1440,
        "Nj": 721,
        "numberOfDataPoints": 1_038_240,
        "iScansNegatively": 0,
        "jScansPositively": 0,
        "jPointsAreConsecutive": 0,
        "alternativeRowScanning": 0,
        "stepUnits": 1,
        "indicatorOfUnitOfTimeRange": 1,
    }
    for key, expected in exact_grid.items():
        _require(keys[key] == expected, f"offline GRIB exact metadata mismatch {key}: {path}")
    for key, expected in {
        "latitudeOfFirstGridPointInDegrees": 90.0,
        "longitudeOfFirstGridPointInDegrees": 0.0,
        "latitudeOfLastGridPointInDegrees": -90.0,
        "longitudeOfLastGridPointInDegrees": 359.75,
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
    }.items():
        _require(float(keys[key]) == expected, f"offline GRIB grid geometry mismatch {key}: {path}")
    _require(values.size == 1_038_240, f"offline GRIB value count mismatch: {path}")
    _require(int(keys["forecastTime"]) == int(row["forecast_hour"]), f"offline GRIB forecast hour mismatch: {path}")
    run = datetime.fromisoformat(str(row["run_init_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    valid = datetime.fromisoformat(str(row["valid_time_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    _require(
        (valid - run).total_seconds() == int(row["forecast_hour"]) * 3600,
        f"offline GRIB census run/valid/forecast arithmetic mismatch: {path}",
    )
    _require(
        int(keys["dataDate"]) == int(run.strftime("%Y%m%d"))
        and int(keys["dataTime"]) == int(run.strftime("%H%M"))
        and int(keys["validityDate"]) == int(valid.strftime("%Y%m%d"))
        and int(keys["validityTime"]) == int(valid.strftime("%H%M")),
        f"offline GRIB run/valid time mismatch: {path}",
    )
    variable = str(row["variable"])
    expected_parameter = {
        "HPBL": ("unknown", 0, 3, 196),
        "UGRD": ("u", 0, 2, 2),
        "VGRD": ("v", 0, 2, 3),
    }.get(variable)
    _require(expected_parameter is not None, f"offline GRIB unexpected variable: {variable}")
    assert expected_parameter is not None
    _require(
        (
            str(keys["shortName"]), int(keys["discipline"]),
            int(keys["parameterCategory"]), int(keys["parameterNumber"]),
        )
        == expected_parameter,
        f"offline GRIB variable/parameter mismatch: {path}",
    )
    level = str(row["level"])
    expected_type = "surface" if level == "surface" else "isobaricInhPa"
    expected_level = 0 if level == "surface" else int(level.split()[0])
    _require(
        keys["typeOfLevel"] == expected_type and int(keys["level"]) == expected_level,
        f"offline GRIB level mismatch: {path}",
    )
    site_values = [
        offline_bilinear_regular_ll(
            values,
            float(site["latitude"]),
            float(site["longitude"]),
            ni=int(keys["Ni"]),
            nj=int(keys["Nj"]),
            first_latitude=float(keys["latitudeOfFirstGridPointInDegrees"]),
            first_longitude=float(keys["longitudeOfFirstGridPointInDegrees"]),
            di=float(keys["iDirectionIncrementInDegrees"]),
            dj=float(keys["jDirectionIncrementInDegrees"]),
            missing_value=float(keys["missingValue"]),
        )
        for site in sites
    ]
    return {
        "feature": f"{variable}_{level.replace(' ', '')}",
        "site_values": site_values,
        "metadata_verified": True,
        "single_message_verified": True,
    }


def decode_offline_grib_process_worker(
    info: Mapping[str, Any], sites: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Spawn-safe worker: each process initializes and uses ecCodes serially."""

    global _DECODE_WORKER_RUNTIME_IDENTITY
    if _DECODE_WORKER_RUNTIME_IDENTITY is None:
        packages: dict[str, Any] = {}
        for distribution_name in ("eccodes", "numpy", "pandas", "pyarrow"):
            module_path = Path(
                str(importlib.import_module(distribution_name).__file__)
            ).resolve()
            packages[distribution_name] = {
                "version": importlib.metadata.version(distribution_name),
                "module_file": str(module_path),
                "module_file_size_bytes": module_path.stat().st_size,
                "module_file_sha256": sha256_file(module_path),
            }
        executable = Path(sys.executable).resolve()
        _DECODE_WORKER_RUNTIME_IDENTITY = {
            "python_version": platform.python_version(),
            "python_executable": str(executable),
            "python_executable_size_bytes": executable.stat().st_size,
            "python_executable_sha256": sha256_file(executable),
            "platform": platform.platform(),
            "packages": packages,
        }
    decoded = decode_offline_grib_message(info, sites)
    decoded["decoder_process_id"] = os.getpid()
    decoded["decoder_runtime_identity_sha256"] = canonical_payload_sha256(
        _DECODE_WORKER_RUNTIME_IDENTITY
    )
    return decoded


def v4_spawn_probe_barrier_initializer(barrier: Any) -> None:
    """Test-evidence initializer proving all seven spawned workers rendezvous."""

    barrier.wait(timeout=120)


def v5_spawn_probe_barrier_initializer(barrier: Any) -> None:
    """V5 evidence initializer proving all seven spawned workers rendezvous."""

    barrier.wait(timeout=120)


def v6_spawn_probe_barrier_initializer(barrier: Any) -> None:
    """V6 evidence initializer proving all seven spawned workers rendezvous."""

    barrier.wait(timeout=120)


def v7_spawn_probe_barrier_initializer(barrier: Any) -> None:
    """V7 evidence initializer proving all seven spawned workers rendezvous."""

    barrier.wait(timeout=120)


def _compare_redecoded_value(
    actual: Any,
    expected: Any,
    *,
    label: str,
    counters: dict[str, int],
) -> None:
    actual_float = float(actual)
    expected_float = float(expected)
    if math.isnan(actual_float) or math.isnan(expected_float):
        _require(
            math.isnan(actual_float) and math.isnan(expected_float),
            f"offline replay finite/missing mismatch: {label}",
        )
        counters["nan_exact"] += 1
        return
    _require(
        math.isfinite(actual_float) and math.isfinite(expected_float),
        f"offline replay non-finite value: {label}",
    )
    if struct.pack(">d", actual_float) == struct.pack(">d", expected_float):
        counters["finite_bit_exact"] += 1
        return
    _require(
        abs(actual_float - expected_float) <= DECODE_ABSOLUTE_TOLERANCE,
        f"offline replay exceeds absolute tolerance: {label}",
    )
    counters["finite_within_tolerance"] += 1


def _audit_full_offline_redecode(
    raw_info: Mapping[tuple[str, str, str], Mapping[str, Any]],
    coordinate: Mapping[str, Any],
    outputs: Mapping[str, Path],
    contract: AuditContract,
    *,
    decoder: Any,
    synthetic_decoder_override: bool,
    expected_runtime_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Stream every message through an independent <=7-worker offline replay."""

    import numpy as np
    import pandas as pd

    sites = coordinate["sites"]
    canonical_site = pd.read_parquet(
        outputs["site"], columns=list(SITE_METADATA_COLUMNS + FEATURE_COLUMNS)
    )
    canonical_group = pd.read_parquet(
        outputs["group"], columns=list(GROUP_METADATA_COLUMNS + FEATURE_COLUMNS)
    )
    site_index = {
        (str(row["valid_time_utc"]), int(row["site_id"])): row
        for row in canonical_site.to_dict("records")
    }
    group_index = {
        (str(row["valid_time_utc"]), str(row["group"])): row
        for row in canonical_group.to_dict("records")
    }
    items = sorted(raw_info.values(), key=lambda info: Path(info["path"]).as_posix())
    counters = {
        "finite_bit_exact": 0,
        "finite_within_tolerance": 0,
        "nan_exact": 0,
    }
    message_count = 0
    site_comparisons = 0
    group_comparisons = 0
    decoder_process_ids: set[int] = set()
    decoder_runtime_digests: set[str] = set()

    def consume(info: Mapping[str, Any], decoded: Mapping[str, Any]) -> None:
        nonlocal message_count, site_comparisons, group_comparisons
        row = info["row"]
        expected_feature = f"{row['variable']}_{str(row['level']).replace(' ', '')}"
        _require(decoded.get("feature") == expected_feature, "offline replay feature identity mismatch")
        _require(decoded.get("metadata_verified") is True, "offline replay metadata gate absent")
        _require(decoded.get("single_message_verified") is True, "offline replay single-message gate absent")
        if "decoder_process_id" in decoded:
            decoder_process_ids.add(int(decoded["decoder_process_id"]))
        if "decoder_runtime_identity_sha256" in decoded:
            decoder_runtime_digests.add(
                str(decoded["decoder_runtime_identity_sha256"])
            )
        values = list(decoded.get("site_values", []))
        _require(len(values) == contract.site_count, "offline replay site-value count mismatch")
        valid_time = str(row["valid_time_utc"])
        for site, value in zip(sites, values):
            site_id = int(site["site_id"])
            expected_row = site_index.get((valid_time, site_id))
            _require(expected_row is not None, "offline replay site key absent from canonical matrix")
            _compare_redecoded_value(
                value,
                expected_row[expected_feature],
                label=f"{valid_time}/{site_id}/{expected_feature}",
                counters=counters,
            )
            site_comparisons += 1
        for group_contract in contract.groups:
            members = [
                (float(value), float(site["capacity_mw"]))
                for site, value in zip(sites, values)
                if str(site["group"]) == group_contract.name
            ]
            _require(len(members) == group_contract.site_count, "offline replay group membership mismatch")
            member_values = np.asarray([value for value, _ in members], dtype=float)
            capacities = np.asarray([capacity for _, capacity in members], dtype=float)
            replay_group_value = (
                float(np.sum(member_values * capacities) / float(np.sum(capacities)))
                if np.isfinite(member_values).all()
                else float("nan")
            )
            expected_group = group_index.get((valid_time, group_contract.name))
            _require(expected_group is not None, "offline replay group key absent from canonical matrix")
            _compare_redecoded_value(
                replay_group_value,
                expected_group[expected_feature],
                label=f"{valid_time}/{group_contract.name}/{expected_feature}",
                counters=counters,
            )
            group_comparisons += 1
        message_count += 1

    if synthetic_decoder_override:
        # Test injection is deliberately serial: closures/fakes are not safely
        # picklable, and ecCodes must never share one Windows process across
        # concurrent threads.
        for info in items:
            consume(info, decoder(info, sites))
        execution_backend = "serial_synthetic_test_override"
        observed_worker_limit = 1
    else:
        _require(
            decoder is decode_offline_grib_message,
            "production replay decoder identity mismatch",
        )
        spawn_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=DECODE_MAX_WORKERS,
            mp_context=spawn_context,
        ) as pool:
            for offset in range(0, len(items), DECODE_MAX_WORKERS):
                wave = items[offset : offset + DECODE_MAX_WORKERS]
                futures = [
                    pool.submit(decode_offline_grib_process_worker, info, sites)
                    for info in wave
                ]
                try:
                    decoded_wave = [future.result() for future in futures]
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
                for info, decoded in zip(wave, decoded_wave):
                    consume(info, decoded)
        execution_backend = "spawn_process_pool_eccodes_serial_per_process"
        observed_worker_limit = DECODE_MAX_WORKERS
        _require(
            expected_runtime_identity_sha256 is not None
            and decoder_runtime_digests == {expected_runtime_identity_sha256},
            "spawned ecCodes worker runtime differs from authorized runtime",
        )

    _require(message_count == contract.range_rows, "offline replay message closure mismatch")
    _require(site_comparisons == contract.range_rows * contract.site_count, "offline replay site comparison closure mismatch")
    _require(group_comparisons == contract.range_rows * len(contract.groups), "offline replay group comparison closure mismatch")
    return {
        "decoder": (
            "synthetic_test_decoder_override"
            if synthetic_decoder_override
            else "independent_offline_eccodes_all_messages"
        ),
        "messages": message_count,
        "metadata_and_single_message_gates": message_count,
        "site_value_comparisons": site_comparisons,
        "group_value_comparisons": group_comparisons,
        "execution_backend": execution_backend,
        "max_streaming_workers": observed_worker_limit,
        "decoder_processes_observed": len(decoder_process_ids),
        "decoder_runtime_identity_sha256": (
            expected_runtime_identity_sha256
            if not synthetic_decoder_override
            else None
        ),
        "predeclared_absolute_tolerance": DECODE_ABSOLUTE_TOLERANCE,
        **counters,
        "network_requests": 0,
        "external_data_array_bytes": 0,
    }


def _verify_inventory(
    records: Any, paths: Iterable[Path], root: Path, *, label: str
) -> None:
    _require(isinstance(records, list), f"{label} is not a list")
    expected = [identity(path, root) for path in sorted(paths)]
    _require(records == expected, f"{label} exact identity inventory mismatch")


def _recovery_fixed_root_relatives(
    failed_attempt: str, completed_rows: int
) -> dict[str, str]:
    return {
        "incident": RECOVERY_INCIDENT_RELATIVE,
        "sealer_entrypoint_incident": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
        "flawed_launch_incident": RECOVERY_FLAWED_LAUNCH_INCIDENT_RELATIVE,
        "launch_identity_correction": RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE,
        "original_authorization": "prereg/raw_launch_authorization_v1.json",
        "original_independent_go": "independent_redteam/TRACK_A_RAW_LAUNCH_GO.json",
        "original_failed_lock": (
            f"raw/launch_history/{failed_attempt}__failed.lock"
        ),
        "original_final_progress": (
            f"raw/progress/raw_ranges__{failed_attempt}__{completed_rows:06d}.json"
        ),
        "original_stderr": "logs/TRACK_A_RAW_RUN.stderr.log",
        "original_stdout": "logs/TRACK_A_RAW_RUN.stdout.log",
        "field_range_census": "census/field_range_census.parquet",
        "census_manifest": "manifest_census_v1.json",
        "coordinate_lock": "prereg/authoritative_turbine_coordinate_lock_v1.json",
        "raw_cache_lock": RECOVERY_RAW_CACHE_LOCK_RELATIVE,
        "real_spawn_preflight": RECOVERY_REAL_PREFLIGHT_RELATIVE,
    }


def _recovery_bootstrap_competing_candidates(
    source_root: Path, source_path: Path
) -> list[str]:
    alias = "noaa_gfs_decode_recovery_bootstrap"
    exact_source_name = f"{alias}.py"
    competitors: list[str] = []
    for child in source_root.iterdir():
        folded_name = child.name.casefold()
        if not (
            folded_name == alias.casefold()
            or folded_name.startswith(f"{alias}.".casefold())
        ):
            continue
        if (
            child.name == exact_source_name
            and child == source_path
            and not _linklike(child)
            and child.is_file()
        ):
            continue
        competitors.append(child.name)
    return sorted(competitors, key=lambda value: (value.casefold(), value))


def _expected_recovery_bootstrap_import_contract() -> dict[str, Any]:
    source_root = (REPO / "src").resolve()
    _require(
        source_root.is_dir() and not _linklike(source_root),
        "recovery bootstrap import root is absent or a link",
    )
    require_no_symlink_chain(
        RECOVERY_BOOTSTRAP, REPO, label="recovery bootstrap source path"
    )
    _require(
        RECOVERY_BOOTSTRAP.is_file() and not _linklike(RECOVERY_BOOTSTRAP),
        "recovery bootstrap source is not an exact regular file",
    )
    stdlib_roots = sorted(
        {name.split(".", 1)[0] for name in RECOVERY_BOOTSTRAP_STDLIB_IMPORTS}
    )
    shadows = sorted(
        child.name
        for child in source_root.iterdir()
        if any(
            child.name.casefold() == stdlib.casefold()
            or child.name.casefold().startswith(f"{stdlib}.".casefold())
            for stdlib in stdlib_roots
        )
    )
    _require(
        not shadows,
        f"workspace src shadows bootstrap stdlib roots: {shadows[:5]}",
    )
    competitors = _recovery_bootstrap_competing_candidates(
        source_root, RECOVERY_BOOTSTRAP
    )
    _require(
        not competitors,
        f"workspace src has top-level bootstrap competitors: {competitors[:5]}",
    )
    return {
        "module_name": "noaa_gfs_decode_recovery_bootstrap",
        "source_root": str(source_root),
        "sys_path_index": 0,
        "pythonpath_exact": str(source_root),
        "src_package_initializer_executed": False,
        "bootstrap_module_package": "",
        "stdlib_import_roots": stdlib_roots,
        "stdlib_shadow_candidates": [],
        "top_level_module_competing_candidates": [],
        "bootstrap_source_path": str(RECOVERY_BOOTSTRAP.resolve()),
    }


def _validate_recovery_bootstrap_import_contract(
    observed: Any,
) -> dict[str, Any]:
    expected = _expected_recovery_bootstrap_import_contract()
    _require(
        observed == expected,
        "recovery bootstrap import contract mismatch",
    )
    return expected


def _audit_current_recovery_bootstrap_pyc_absence(
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Prove the recovery bootstrap has no currently importable bytecode cache.

    The sealed preflight records the before/after test state, but that historical
    statement cannot establish the postrun state.  Inspect only the two Python
    cache namespaces that can match this fixed top-level bootstrap module.  A
    cache-directory link is rejected before enumeration so this check cannot be
    redirected outside the trusted source root.
    """

    lexical_root = source_root if source_root is not None else REPO / "src"
    _require(
        not _linklike(lexical_root),
        "recovery bootstrap bytecode source root is absent or a link",
    )
    _require(
        lexical_root.is_dir(),
        "recovery bootstrap bytecode source root is absent or a link",
    )
    checked_root = lexical_root.resolve()
    legacy_path = checked_root / "noaa_gfs_decode_recovery_bootstrap.pyc"
    cache_root = checked_root / "__pycache__"
    matches: list[Path] = []
    if os.path.lexists(legacy_path):
        matches.append(legacy_path)
    if os.path.lexists(cache_root):
        _require(
            not _linklike(cache_root),
            "recovery bootstrap bytecode cache directory is a forbidden link",
        )
        _require(
            cache_root.is_dir(),
            "recovery bootstrap bytecode cache namespace is not a directory",
        )
        for candidate in cache_root.iterdir():
            folded_name = candidate.name.casefold()
            if (
                folded_name.startswith(
                    "noaa_gfs_decode_recovery_bootstrap.".casefold()
                )
                and folded_name.endswith(".pyc")
            ):
                matches.append(candidate)
    _require(
        not matches,
        "matching recovery bootstrap pyc exists at postrun audit",
    )
    return {
        "status": "PASS_CURRENT_MATCHING_BOOTSTRAP_PYC_EXACT_ZERO",
        "source_root": str(checked_root),
        "legacy_relative_path": "noaa_gfs_decode_recovery_bootstrap.pyc",
        "cache_tag_glob": (
            "__pycache__/noaa_gfs_decode_recovery_bootstrap.*.pyc"
        ),
        "matching_pyc_files": 0,
    }


def _audit_recovery_control_temporary_files_absent(
    root: Path,
) -> dict[str, Any]:
    """Reject abandoned no-overwrite publisher siblings for every control."""

    matches: list[str] = []
    for relative in RECOVERY_CONTROL_RELATIVES:
        destination = root / relative
        parent = destination.parent
        if not os.path.lexists(parent):
            continue
        require_no_symlink_chain(
            parent, root, label="recovery control temporary-file parent"
        )
        _require(
            parent.is_dir(),
            "recovery control temporary-file parent is not a directory",
        )
        prefix = f"{destination.name}.tmp.".casefold()
        matches.extend(
            child.relative_to(root).as_posix()
            for child in parent.iterdir()
            if child.name.casefold().startswith(prefix)
        )
    _require(
        not matches,
        f"recovery control publisher temporary file remains: {sorted(matches)[:5]}",
    )
    return {
        "status": "PASS_RECOVERY_CONTROL_PUBLISHER_TEMPORARY_FILES_EXACT_ZERO",
        "checked_destination_count": len(RECOVERY_CONTROL_RELATIVES),
        "checked_destination_relative_paths": list(RECOVERY_CONTROL_RELATIVES),
        "matching_temporary_files": 0,
    }


def _v6_validate_v5_false_reject_incident(
    root: Path, record: Any
) -> dict[str, Any]:
    expected = {
        "path": V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": V6_HISTORICAL_FALSE_REJECT_INCIDENT_SIZE_BYTES,
        "sha256": V6_HISTORICAL_FALSE_REJECT_INCIDENT_SHA256,
    }
    _v4_require_exact_declared_identity(
        record, expected, label="V5 historical-identity false-reject incident"
    )
    actual = _v4_verify_exact_root_identity(
        root,
        record,
        expected,
        label="V5 historical-identity false-reject incident",
    )
    payload = _v5_load_strict_json(
        path_under(root, V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE),
        label="V5 historical-identity false-reject incident",
    )
    _require(
        set(payload)
        == {
            "artifact_type", "authority_scope", "created_utc",
            "exact_mismatch_set", "execution_boundary", "failed_execution",
            "historical_incident", "immutable_v5_chain", "postfailure_state",
            "prohibitions", "required_v6_supersession", "root_cause",
            "schema_version", "status",
        }
        and payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_HISTORICAL_IDENTITY_FALSE_REJECT_INCIDENT"
        and payload.get("status")
        == "SEALED_V5_POSTRUN_AUDITOR_HISTORICAL_IDENTITY_FALSE_REJECT_AFTER_FULL_REPLAY_NO_REPORT_V6_SUPERSESSION_REQUIRED"
        and payload.get("authority_scope")
        == {"documentary_only": True, "v5_retry_authorized": False, "v6_authorized": False}
        and payload.get("prohibitions")
        == {
            "mutate_v1_through_v5": False,
            "network_or_recovery_rerun": False,
            "publish_or_run_v6": False,
            "rerun_v5": False,
        },
        "V5 historical-identity false-reject incident schema/header mismatch",
    )
    created = _v4_created_utc(
        payload.get("created_utc"), label="V5 historical false-reject incident"
    )
    historical = payload.get("historical_incident")
    _require(
        historical
        == {
            "identity": {
                "path": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
                "size_bytes": 7_732,
                "sha256": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_SHA256,
            },
            "failed_heads": {
                "size_bytes": V6_HISTORICAL_FAILED_HEADS_CANONICAL_BYTES,
                "sha256": V6_HISTORICAL_FAILED_HEADS_CANONICAL_SHA256,
            },
            "zero_state": {
                "size_bytes": V6_HISTORICAL_ZERO_STATE_CANONICAL_BYTES,
                "sha256": V6_HISTORICAL_ZERO_STATE_CANONICAL_SHA256,
            },
            "absent_control_paths": {
                "size_bytes": V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_BYTES,
                "sha256": V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_SHA256,
            },
        },
        "V5 false-reject incident historical commitment mismatch",
    )
    chain = payload.get("immutable_v5_chain")
    expected_code = {
        "auditor": {
            "path": "scripts/audit_noaa_gfs_multiseason_raw_postrun_v5.py",
            "size_bytes": 481_173,
            "sha256": FROZEN_V5_SOURCE_IDENTITIES["superseded_v5_auditor"]["sha256"],
        },
        "auditor_test": {
            "path": "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py",
            "size_bytes": 228_289,
            "sha256": FROZEN_V5_SOURCE_IDENTITIES["superseded_v5_auditor_test"]["sha256"],
        },
        "sealer": {
            "path": "scripts/seal_noaa_gfs_multiseason_postrun_audit_v5.py",
            "size_bytes": 76_117,
            "sha256": FROZEN_V5_SOURCE_IDENTITIES["superseded_v5_sealer"]["sha256"],
        },
        "sealer_test": {
            "path": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py",
            "size_bytes": 39_039,
            "sha256": FROZEN_V5_SOURCE_IDENTITIES["superseded_v5_sealer_test"]["sha256"],
        },
    }
    _require(
        isinstance(chain, Mapping)
        and set(chain) == {"audit_attempt_id", "code", "controls"}
        and chain.get("audit_attempt_id") == FROZEN_V5_AUDIT_ATTEMPT_ID
        and chain.get("code") == expected_code
        and chain.get("controls") == FROZEN_V5_CONTROL_IDENTITIES,
        "V5 false-reject incident immutable V5 chain mismatch",
    )
    mismatch = payload.get("exact_mismatch_set")
    drift = mismatch.get("failed_head_role_path_drift") if isinstance(mismatch, Mapping) else None
    zero_drift = mismatch.get("zero_state_control_version_drift") if isinstance(mismatch, Mapping) else None
    _require(
        isinstance(drift, Mapping)
        and drift.get("matching_roles") == ["recovery_core", "recovery_bootstrap"]
        and isinstance(drift.get("mismatches"), list)
        and len(drift["mismatches"]) == 5
        and {row.get("role") for row in drift["mismatches"]}
        == {
            "recovery_runner", "recovery_sealer", "recovery_sealer_test",
            "planned_postrun_auditor", "planned_postrun_auditor_test",
        }
        and isinstance(zero_drift, Mapping)
        and zero_drift.get("historical_v1_absent_paths")
        == list(RECOVERY_V1_CONTROL_RELATIVES)
        and zero_drift.get("incorrect_current_v2_expected_paths")
        == list(RECOVERY_V2_CONTROL_RELATIVES),
        "V5 false-reject exact mismatch set mismatch",
    )
    execution = payload.get("failed_execution")
    stdout = execution.get("stdout_normalized_lf") if isinstance(execution, Mapping) else None
    _require(
        isinstance(execution, Mapping)
        and execution.get("exit_code") == 1
        and execution.get("wall_seconds") == 262.4
        and isinstance(stdout, Mapping)
        and stdout.get("size_bytes") == 278
        and stdout.get("sha256")
        == "13b8efe41d7176a5a63af0724a1d2457a7f52cde7bf7532b8865ec657e1cd6d9"
        and stdout.get("payload", {}).get("reason")
        == "sealer-entrypoint historical identity invalid: recovery_runner"
        and stdout.get("payload", {}).get("audit_network_requests") == 0
        and stdout.get("payload", {}).get("audit_files_written") == 0,
        "V5 false-reject execution evidence mismatch",
    )
    required = payload.get("required_v6_supersession")
    _require(
        isinstance(required, Mapping)
        and required.get("code")
        == [
            "scripts/audit_noaa_gfs_multiseason_raw_postrun_v6.py",
            "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py",
            "scripts/seal_noaa_gfs_multiseason_postrun_audit_v6.py",
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py",
        ]
        and required.get("controls")
        == [V6_AUTH_RELATIVE, V6_REVIEW_RELATIVE, V6_GO_RELATIVE]
        and required.get("incident_relative_path")
        == V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE
        and required.get("must_bind_this_incident_identity") is True
        and required.get("report_file") is None
        and required.get("status") == "REQUIRED_NOT_YET_AUTHORIZED",
        "V5 false-reject required V6 supersession mismatch",
    )
    root_cause = payload.get("root_cause")
    _require(
        root_cause
        == {
            "category": "HISTORICAL_RECORD_CURRENT_CONSTANT_CONFLATION",
            "first_reject": "failed_head_identities.recovery_runner",
            "latent_second_reject": "verified_zero_state_after_failure.absent_control_paths",
            "production_9e74_positive_test_absent": True,
            "test_fixture_self_consistent_with_current_v2_constants": True,
        },
        "V5 false-reject root-cause mismatch",
    )
    return {"payload": payload, "identity": actual, "created": created}


def _v6_validate_historical_9e74_payload(
    payload: Mapping[str, Any], incident_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate immutable V1 records as literals, never as current file roles."""

    _require(
        incident_identity
        == {
            "path": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
            "size_bytes": 7_732,
            "sha256": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_SHA256,
        },
        "historical direct-file incident identity mismatch",
    )
    _require(
        isinstance(payload, Mapping)
        and set(payload)
        == {
            "artifact_type", "schema_version", "status", "created_utc",
            "failed_execution", "failure_boundary", "failed_head_identities",
            "immutable_raw_prestate", "verified_zero_state_after_failure",
            "root_cause", "mandatory_remediation",
        }
        and payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
        and payload.get("status")
        == "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE",
        "historical direct-file incident schema/header mismatch",
    )
    failed_heads = payload.get("failed_head_identities")
    zero_state = payload.get("verified_zero_state_after_failure")
    _require(
        failed_heads == V6_HISTORICAL_FAILED_HEAD_IDENTITIES
        and len(failed_heads) == 7,
        "historical failed-head exact-seven literal mismatch",
    )
    failed_head_bytes = _v4_canonical_bytes(failed_heads)
    _require(
        len(failed_head_bytes) == V6_HISTORICAL_FAILED_HEADS_CANONICAL_BYTES
        and hashlib.sha256(failed_head_bytes).hexdigest()
        == V6_HISTORICAL_FAILED_HEADS_CANONICAL_SHA256,
        "historical failed-head canonical commitment mismatch",
    )
    _require(
        zero_state == V6_HISTORICAL_ZERO_STATE,
        "historical verified zero-state literal mismatch",
    )
    zero_bytes = _v4_canonical_bytes(zero_state)
    absent = zero_state["absent_control_paths"]
    absent_bytes = _v4_canonical_bytes(absent)
    _require(
        len(zero_bytes) == V6_HISTORICAL_ZERO_STATE_CANONICAL_BYTES
        and hashlib.sha256(zero_bytes).hexdigest()
        == V6_HISTORICAL_ZERO_STATE_CANONICAL_SHA256
        and absent == list(RECOVERY_V1_CONTROL_RELATIVES)
        and len(absent) == 5
        and len(absent_bytes) == V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_BYTES
        and hashlib.sha256(absent_bytes).hexdigest()
        == V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_SHA256,
        "historical V1 absent-control/zero-state commitment mismatch",
    )
    active_paths = {
        "recovery_core": str(RECOVERY_MODULE.resolve()),
        "recovery_bootstrap": str(RECOVERY_BOOTSTRAP.resolve()),
        "recovery_runner": str(RECOVERY_RUNNER.resolve()),
        "recovery_sealer": str(RECOVERY_SEALER.resolve()),
        "recovery_sealer_test": str(RECOVERY_SEALER_TEST.resolve()),
        "planned_postrun_auditor": str(SUPERSEDED_V2_AUDITOR.resolve()),
        "planned_postrun_auditor_test": str(SUPERSEDED_V2_AUDITOR_TEST.resolve()),
    }
    matching_roles = {
        role
        for role, record in failed_heads.items()
        if os.path.normcase(str(record["path"]))
        == os.path.normcase(active_paths[role])
    }
    _require(
        matching_roles == {"recovery_core", "recovery_bootstrap"}
        and all(
            os.path.normcase(str(failed_heads[role]["path"]))
            != os.path.normcase(active_paths[role])
            for role in set(failed_heads) - matching_roles
        )
        and failed_heads["recovery_runner"]["sha256"]
        not in {
            RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY["sha256"],
            RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]["sha256"],
        },
        "historical/current/transient/final role distinction mismatch",
    )
    return {
        "incident": dict(incident_identity),
        "historical_incident_contract": {
            "failed_head_identities": {
                "canonical_size_bytes": V6_HISTORICAL_FAILED_HEADS_CANONICAL_BYTES,
                "canonical_sha256": V6_HISTORICAL_FAILED_HEADS_CANONICAL_SHA256,
            },
            "verified_zero_state_after_failure": {
                "canonical_size_bytes": V6_HISTORICAL_ZERO_STATE_CANONICAL_BYTES,
                "canonical_sha256": V6_HISTORICAL_ZERO_STATE_CANONICAL_SHA256,
            },
            "absent_control_paths": {
                "canonical_size_bytes": V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_BYTES,
                "canonical_sha256": V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_SHA256,
            },
        },
        "validation_mode": (
            "IMMUTABLE_9E74_HISTORICAL_CONTRACT_PREDATA_NO_CURRENT_ROLE_DEREFERENCE"
        ),
        "validation_moved_before_parquet_and_data_reads": True,
        "current_role_path_dereference_forbidden": True,
    }


def _validate_recovery_sealer_entrypoint_incident(
    payload: Mapping[str, Any],
    incident_identity: Mapping[str, Any],
    root: Path,
    authorization: Mapping[str, Any],
    contract: AuditContract,
) -> None:
    historical_contract = (
        _v6_validate_historical_9e74_payload(payload, incident_identity)
        if contract == PRODUCTION_CONTRACT
        else None
    )
    _require(
        set(payload)
        == {
            "artifact_type",
            "schema_version",
            "status",
            "created_utc",
            "failed_execution",
            "failure_boundary",
            "failed_head_identities",
            "immutable_raw_prestate",
            "verified_zero_state_after_failure",
            "root_cause",
            "mandatory_remediation",
        },
        "sealer-entrypoint incident schema mismatch",
    )
    _require(
        int(payload.get("schema_version", -1)) == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
        and payload.get("status")
        == "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE",
        "sealer-entrypoint incident header mismatch",
    )
    _parse_exact_utc_z(
        payload.get("created_utc"), label="sealer-entrypoint incident created_utc"
    )
    if contract == PRODUCTION_CONTRACT:
        _require(
            int(incident_identity.get("size_bytes", -1)) == 7_732
            and incident_identity.get("sha256")
            == RECOVERY_SEALER_ENTRYPOINT_INCIDENT_SHA256,
            "production sealer-entrypoint incident constant mismatch",
        )

    execution = payload.get("failed_execution")
    _require(
        isinstance(execution, Mapping)
        and set(execution)
        == {
            "observed_start_utc",
            "command",
            "invocation_mode",
            "execution_wrapper_observed_exit_code",
            "sealer_failure_handler_declared_exit_code",
            "observed_stdout_json",
            "observed_stderr_exact_normalized",
            "primary_exception_exact",
            "secondary_child_exception_exact",
            "secondary_exception_is_parent_abort_consequence",
        },
        "sealer-entrypoint failed-execution schema mismatch",
    )
    _parse_exact_utc_z(
        execution.get("observed_start_utc"),
        label="sealer-entrypoint observed_start_utc",
    )
    command = execution.get("command")
    stdout = execution.get("observed_stdout_json")
    _require(
        isinstance(command, list)
        and len(command) == 5
        and all(isinstance(part, str) for part in command)
        and os.path.normcase(str(Path(os.path.abspath(command[0]))))
        == os.path.normcase(str(Path(sys.executable).resolve()))
        and command[1:4]
        == [
            "-B",
            "scripts/seal_noaa_gfs_multiseason_decode_recovery_v1.py",
            "--root",
        ]
        and bool(command[4])
        and execution.get("invocation_mode") == "DIRECT_FILE"
        and int(execution.get("execution_wrapper_observed_exit_code", -1)) == 1
        and int(execution.get("sealer_failure_handler_declared_exit_code", -1)) == 2
        and isinstance(stdout, Mapping)
        and set(stdout) == {"status", "error_type", "error", "network_requests"}
        and stdout.get("status") == "FAIL_DECODE_RECOVERY_SEAL"
        and stdout.get("error_type") == "PicklingError"
        and "src.noaa_gfs_decode_recovery_bootstrap" in str(stdout.get("error", ""))
        and int(stdout.get("network_requests", -1)) == 0
        and isinstance(execution.get("observed_stderr_exact_normalized"), str)
        and "multiprocessing" in execution["observed_stderr_exact_normalized"]
        and execution.get("primary_exception_exact")
        == (
            "PicklingError: Can't pickle initialize_spawned_worker because import "
            "of module src.noaa_gfs_decode_recovery_bootstrap failed"
        )
        and isinstance(execution.get("secondary_child_exception_exact"), str)
        and execution.get("secondary_exception_is_parent_abort_consequence") is True,
        "sealer-entrypoint direct-file failure contract mismatch",
    )

    _require(
        payload.get("failure_boundary")
        == {
            "raw_cache_validation_completed_read_only": True,
            "spawned_pilot_submission_started": True,
            "spawned_pilot_tasks_completed": 0,
            "immutable_test_subprocesses_started": 0,
            "seal_publications_created": 0,
            "independent_review_created": 0,
            "independent_go_created": 0,
            "recovery_launch_started": False,
            "network_requests": 0,
            "network_bytes": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        "sealer-entrypoint failure boundary mismatch",
    )
    failed_heads = payload.get("failed_head_identities")
    if contract == PRODUCTION_CONTRACT:
        _require(
            failed_heads == V6_HISTORICAL_FAILED_HEAD_IDENTITIES
            and historical_contract is not None
            and historical_contract["current_role_path_dereference_forbidden"] is True,
            "sealer-entrypoint historical failed-head contract mismatch",
        )
    else:
        expected_head_paths = {
            "recovery_core": RECOVERY_MODULE,
            "recovery_bootstrap": RECOVERY_BOOTSTRAP,
            "recovery_runner": RECOVERY_RUNNER,
            "recovery_sealer": RECOVERY_SEALER,
            "recovery_sealer_test": RECOVERY_SEALER_TEST,
            "planned_postrun_auditor": SUPERSEDED_V2_AUDITOR,
            "planned_postrun_auditor_test": SUPERSEDED_V2_AUDITOR_TEST,
        }
        _require(
            isinstance(failed_heads, Mapping)
            and set(failed_heads) == set(expected_head_paths)
            and all(
                isinstance(failed_heads[role], Mapping)
                and set(failed_heads[role]) == {"path", "size_bytes", "sha256"}
                and os.path.normcase(
                    str(Path(os.path.abspath(str(failed_heads[role]["path"]))))
                )
                == os.path.normcase(str(Path(os.path.abspath(expected_path))))
                for role, expected_path in expected_head_paths.items()
            ),
            "synthetic sealer-entrypoint failed-head contract mismatch",
        )

    raw_prestate = payload.get("immutable_raw_prestate")
    _require(
        isinstance(raw_prestate, Mapping)
        and set(raw_prestate)
        == {
            "prior_decode_failure_incident",
            "final_original_progress",
            "original_failed_lock",
            "documented_preplan_remnants",
            "raw_grib_files",
            "raw_sidecar_files",
            "request_event_files",
            "progress_files",
            "launch_history_files",
            "validated_raw_payload_bytes",
        }
        and raw_prestate.get("prior_decode_failure_incident")
        == authorization.get("incident")
        and raw_prestate.get("final_original_progress")
        == authorization.get("original_final_progress")
        and raw_prestate.get("original_failed_lock")
        == authorization.get("original_failed_lock")
        and raw_prestate.get("documented_preplan_remnants")
        == authorization.get("documented_preplan_remnants")
        and int(raw_prestate.get("raw_grib_files", -1)) == contract.range_rows
        and int(raw_prestate.get("raw_sidecar_files", -1)) == contract.range_rows
        and int(raw_prestate.get("request_event_files", -1))
        == contract.range_rows * 2
        and int(raw_prestate.get("progress_files", -1))
        == len(recovery_progress_counts(contract))
        and int(raw_prestate.get("launch_history_files", -1)) == 1
        and int(raw_prestate.get("validated_raw_payload_bytes", -1))
        == contract.range_bytes,
        "sealer-entrypoint immutable raw prestate mismatch",
    )
    expected_zero_state = (
        V6_HISTORICAL_ZERO_STATE
        if contract == PRODUCTION_CONTRACT
        else {
            "absent_control_paths": list(RECOVERY_V2_CONTROL_RELATIVES),
            "canonical_recovery_outputs_absent": True,
            "recovery_active_locks_absent": True,
            "seal_temporary_files_absent": True,
            "matching_bootstrap_pyc_files": 0,
            "only_original_failed_output_transaction_present": True,
        }
    )
    _require(
        payload.get("verified_zero_state_after_failure") == expected_zero_state,
        "sealer-entrypoint historical/synthetic zero-state boundary mismatch",
    )
    _require(
        payload.get("root_cause")
        == {
            "category": "WINDOWS_SPAWN_PICKLE_IMPORTABILITY",
            "direct_file_sys_path_root_missing": True,
            "exact_loaded_bootstrap_module_name": (
                "src.noaa_gfs_decode_recovery_bootstrap"
            ),
            "parent_package_importability_not_established": True,
            "raw_data_or_decoder_value_failure": False,
        },
        "sealer-entrypoint root-cause mismatch",
    )
    _require(
        payload.get("mandatory_remediation")
        == {
            "canonical_write_invocation": (
                "python -B -m scripts.seal_noaa_gfs_multiseason_decode_recovery_v1"
            ),
            "direct_file_write_invocation_forbidden": True,
            "bootstrap_top_level_module_name": (
                "noaa_gfs_decode_recovery_bootstrap"
            ),
            "bootstrap_import_root": str((REPO / "src").resolve()),
            "src_package_initializer_must_not_execute": True,
            "bootstrap_source_origin_hash_ast_and_pyc_gates_remain_required": True,
            "real_module_entrypoint_seven_spawn_regression_required": True,
            "revised_code_test_auditor_hash_chain_and_independent_pass_required": True,
            "retry_authorized": False,
        },
        "sealer-entrypoint mandatory remediation mismatch",
    )


def _validate_v2_launch_supersession_incident(payload: Mapping[str, Any]) -> None:
    required_v2 = payload.get("required_v2_supersession")
    root_cause = payload.get("root_cause")
    postfailure = payload.get("postfailure_state_closure")
    disposition = payload.get("v1_disposition")
    expected_required = {
        "new_attempt_id_required",
        "new_postrun_auditor_binding_required",
        "new_raw_lock_preflight_authorization_review_and_go_required",
        "new_runner_and_dedicated_run_prefix_regression_required",
        "new_sealer_and_process_evidence_required",
        "old_v1_chain_must_not_be_overwritten",
        "regression_must_execute_run_through_postpilot_check",
        "regression_must_stop_before_recovery_claim_or_target_write",
        "retry_performed_by_this_incident_seal",
        "this_incident_must_be_transitively_bound",
    }
    _require(
        set(payload)
        == {
            "artifact_type", "captured_result", "created_utc",
            "failed_recovery_attempt_id", "failed_v1_chain",
            "first_monitor_observation", "invocation", "network_evidence",
            "observed_time_bounds", "postfailure_state_closure",
            "required_v2_supersession", "root_cause", "schema_version",
            "status", "target_free_scope", "v1_disposition",
        }
        and payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_LAUNCH_BOOTSTRAP_IMPORT_CONTRACT_NAMEERROR_INCIDENT"
        and payload.get("status")
        == "CLOSED_FAIL_SAFE_REQUIRES_APPEND_ONLY_V2_SUPERSESSION"
        and payload.get("failed_recovery_attempt_id")
        == "decode_recovery_v1__20260810T194716214040Z"
        and payload.get("failed_v1_chain") == RECOVERY_FLAWED_V1_CHAIN_IDENTITIES
        and isinstance(required_v2, Mapping)
        and set(required_v2) == expected_required
        and all(
            required_v2[key] is True
            for key in expected_required
            if key != "retry_performed_by_this_incident_seal"
        )
        and required_v2.get("retry_performed_by_this_incident_seal") is False
        and isinstance(root_cause, Mapping)
        and root_cause.get("classification") == "RUN_PATH_LOCAL_BINDING_OMISSION"
        and root_cause.get("frozen_runner_sha256")
        == RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]["sha256"]
        and root_cause.get("recovery_claim_acquired") is False
        and isinstance(postfailure, Mapping)
        and postfailure.get("canonical_outputs_present") == 0
        and postfailure.get("new_recovery_transaction_entries_present") == 0
        and postfailure.get("decoded_directory_present") is False
        and postfailure.get("raw_active_lock_present") is False
        and postfailure.get("decoded_active_lock_present") is False
        and postfailure.get("recovery_target_writes") == 0
        and isinstance(disposition, Mapping)
        and disposition.get("v1_retry_allowed") is False
        and disposition.get("v1_runner_remains_byte_frozen") is True,
        "preserved flawed V1 launch incident semantic mismatch",
    )


def _validate_v2_launch_identity_correction(
    payload: Mapping[str, Any],
    *,
    expected_flawed_identity: Mapping[str, Any],
) -> None:
    policy = payload.get("supersession_policy")
    rejected = payload.get("rejected_transient_identity")
    authority = payload.get("authority_basis")
    unchanged = payload.get("unchanged_fail_safe_state")
    postrehash = payload.get("postcorrection_rehash")
    preserved = payload.get("preserved_failure_evidence")
    preserved_digests = payload.get("preserved_failure_evidence_canonical_sha256")
    policy_keys = {
        "correction_is_authoritative_for_failed_v1_chain",
        "flawed_incident_authority_superseded",
        "flawed_incident_preserved_append_only",
        "rejected_transient_identity_must_never_be_selected",
        "v1_runner_controls_review_and_go_remain_immutable",
        "v2_chain_must_bind_both_flawed_incident_and_this_correction",
        "v2_chain_must_use_only_authoritative_failed_v1_chain",
        "v2_execution_or_retry_authorized_by_this_document",
    }
    digest_pass = (
        isinstance(preserved, Mapping)
        and isinstance(preserved_digests, Mapping)
        and set(preserved) == set(preserved_digests)
        and all(
            canonical_payload_sha256(value) == preserved_digests.get(key)
            for key, value in preserved.items()
        )
    )
    _require(
        set(payload)
        == {
            "artifact_type", "authoritative_failed_v1_chain",
            "authoritative_v1_support", "authority_basis", "correction_reason",
            "created_utc", "postcorrection_rehash", "preserved_failure_evidence",
            "preserved_failure_evidence_canonical_sha256", "prior_incidents",
            "rejected_transient_identity", "schema_version", "status",
            "superseded_flawed_incident", "supersession_policy",
            "target_free_scope", "unchanged_fail_safe_state",
        }
        and payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_LAUNCH_NAMEERROR_INCIDENT_IDENTITY_CORRECTION"
        and payload.get("status")
        == "PASS_APPEND_ONLY_CORRECTION_SUPERSEDES_FLAWED_IDENTITY_AUTHORITY"
        and payload.get("authoritative_failed_v1_chain")
        == RECOVERY_V1_CHAIN_IDENTITIES
        and payload.get("authoritative_v1_support")
        == RECOVERY_V1_SUPPORT_IDENTITIES
        and payload.get("superseded_flawed_incident")
        == expected_flawed_identity
        and digest_pass
        and isinstance(policy, Mapping)
        and set(policy) == policy_keys
        and all(
            policy[key] is True
            for key in policy_keys
            if key != "v2_execution_or_retry_authorized_by_this_document"
        )
        and policy.get("v2_execution_or_retry_authorized_by_this_document") is False
        and isinstance(rejected, Mapping)
        and set(rejected)
        == {
            "identity",
            "disposition",
            "historical_authority",
            "executed_in_failed_launch",
            "authorized_by_v1_controls",
        }
        and rejected.get("identity") == RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY
        and rejected.get("disposition")
        == "REJECTED_UNLAUNCHED_UNAUTHORIZED_MID_EDIT_OBSERVATION"
        and rejected.get("historical_authority") is False
        and rejected.get("executed_in_failed_launch") is False
        and rejected.get("authorized_by_v1_controls") is False
        and isinstance(authority, Mapping)
        and authority.get("v1_authorization_review_and_go_bind_runner_sha256")
        == RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]["sha256"]
        and authority.get("restored_v1_runner_rehash_matches_authorized_identity")
        is True
        and isinstance(postrehash, Mapping)
        and postrehash.get("all_exact") is True
        and postrehash.get("v1_chain") == RECOVERY_V1_CHAIN_IDENTITIES
        and postrehash.get("v1_support") == RECOVERY_V1_SUPPORT_IDENTITIES
        and isinstance(unchanged, Mapping)
        and unchanged.get("canonical_outputs_present") == 0
        and unchanged.get("decoded_directory_present") is False
        and unchanged.get("new_recovery_transaction_entries_present") == 0
        and unchanged.get("recovery_claim_acquired") is False
        and unchanged.get("recovery_target_writes") == 0
        and unchanged.get("runner_guard_and_failure_report_network_requests") == 0,
        "authoritative V1 launch identity correction semantic mismatch",
    )


def _recovery_file_inventory(paths: Iterable[Path], root: Path) -> dict[str, Any]:
    rows = [identity(path, root) for path in sorted(paths)]
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "identity_rows_sha256": canonical_payload_sha256(rows),
    }


def _audit_v1_failed_launch_zero_state_after_v2(root: Path) -> dict[str, Any]:
    """Reject any V1 claim/transaction residue in the exact V2 afterstate."""

    forbidden: list[str] = []
    for directory in (
        root / "raw" / "output_transactions",
        root / "raw" / "progress",
        root / "raw" / "launch_history",
        root / "decoded" / "recovery_progress",
        root / "decoded" / "recovery_history",
    ):
        _require(
            directory.is_dir() and not _linklike(directory),
            f"V1 zero-state audit directory invalid: {directory}",
        )
        for path in directory.rglob("*"):
            _require(
                not _linklike(path),
                "V1 zero-state audit encountered link/junction",
            )
            if "decode_recovery_v1__" in path.name:
                forbidden.append(path.relative_to(root).as_posix())
    _require(
        not forbidden,
        f"failed V1 recovery left forbidden afterstate: {sorted(forbidden)[:5]}",
    )
    return {
        "failed_v1_attempt_id": "decode_recovery_v1__20260810T194716214040Z",
        "recovery_claim_acquired": False,
        "decoded_tree_absent": True,
        "canonical_outputs_absent": True,
        "v1_transaction_entries": [],
        "v1_progress_entries": [],
        "v1_history_entries": [],
    }


def _recovery_runtime_file_paths() -> dict[str, Path]:
    package = REPO / ".venv" / "Lib" / "site-packages"
    return {
        "library": package / "eccodes" / "eccodes.dll",
        "memfs_library": package / "eccodes" / "eccodes_memfs.dll",
        "python_binding": package / "eccodes" / "_eccodes.cp313-win_amd64.pyd",
        "gribapi_binding": package / "gribapi" / "bindings.py",
        "eccodes_python_api": package / "eccodes" / "eccodes.py",
        "gribapi_python_api": package / "gribapi" / "gribapi.py",
        "gribapi_errors": package / "gribapi" / "errors.py",
    }


def _expected_original_runtime_report(runtime: Mapping[str, Any]) -> dict[str, Any]:
    packages = runtime.get("packages")
    _require(isinstance(packages, Mapping), "original runtime package map absent")
    observed_packages: dict[str, Any] = {}
    for name, record in packages.items():
        _require(isinstance(record, Mapping), f"original runtime package invalid: {name}")
        _require(
            set(record)
            == {
                "module_file",
                "module_file_size_bytes",
                "module_file_sha256",
                "version",
            },
            f"original runtime package schema mismatch: {name}",
        )
        observed_packages[str(name)] = {
            "path": str(record["module_file"]),
            "version": str(record["version"]),
        }
    return {
        "python_executable": str(runtime.get("python_executable", "")),
        "python_version": str(runtime.get("python_version", "")),
        "platform": str(runtime.get("platform", "")),
        "packages": observed_packages,
    }


def _validate_recovery_runtime(
    runtime_lock: Any,
    runtime_report: Any,
    authorization: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(runtime_lock, Mapping), "recovery runtime lock is absent")
    _require(
        set(runtime_lock)
        == {
            "api_version",
            "definitions_path",
            "library",
            "memfs_library",
            "python_binding",
            "gribapi_binding",
            "eccodes_python_api",
            "gribapi_python_api",
            "gribapi_errors",
            "frozen_runner",
            "recovery_module",
            "original_runtime_identity",
        },
        "recovery runtime-lock schema mismatch",
    )
    _require(
        runtime_lock.get("api_version") == "2.47.0"
        and runtime_lock.get("definitions_path") == "/MEMFS/definitions",
        "recovery ecCodes API/definitions identity mismatch",
    )
    runtime_actuals: dict[str, dict[str, Any]] = {}
    for key, expected_path in _recovery_runtime_file_paths().items():
        runtime_actuals[key] = verify_identity(
            runtime_lock.get(key, {}),
            expected_path,
            root=None,
            label=f"recovery runtime {key}",
        )
    _require(
        runtime_lock.get("frozen_runner") == authorization.get("original_runner")
        and runtime_lock.get("recovery_module")
        == authorization.get("recovery_module")
        and runtime_lock.get("original_runtime_identity")
        == authorization.get("original_runtime_identity")
        == provenance.get("runtime_identity"),
        "recovery runtime lock provenance mismatch",
    )
    _require(isinstance(runtime_report, Mapping), "recovery runtime report is absent")
    _require(
        set(runtime_report)
        == {
            "pid",
            "api_version",
            "definitions_path",
            "library_path",
            "memfs_library_path",
            "python_binding_path",
            "gribapi_binding_path",
            "eccodes_python_api_path",
            "gribapi_python_api_path",
            "gribapi_errors_path",
            "frozen_runner_path",
            "frozen_runner_sha256",
            "recovery_module_path",
            "recovery_module_sha256",
            "original_runtime_identity",
        },
        "recovery runtime-report schema mismatch",
    )
    expected_paths = {
        "library_path": _recovery_runtime_file_paths()["library"],
        "memfs_library_path": _recovery_runtime_file_paths()["memfs_library"],
        "python_binding_path": _recovery_runtime_file_paths()["python_binding"],
        "gribapi_binding_path": _recovery_runtime_file_paths()["gribapi_binding"],
        "eccodes_python_api_path": _recovery_runtime_file_paths()["eccodes_python_api"],
        "gribapi_python_api_path": _recovery_runtime_file_paths()["gribapi_python_api"],
        "gribapi_errors_path": _recovery_runtime_file_paths()["gribapi_errors"],
        "frozen_runner_path": Path(str(authorization["original_runner"]["path"])),
        "recovery_module_path": RECOVERY_MODULE,
    }
    for field, expected_path in expected_paths.items():
        _require(
            Path(str(runtime_report.get(field, ""))).resolve()
            == expected_path.resolve(),
            f"recovery runtime report path mismatch: {field}",
        )
    _require(
        int(runtime_report.get("pid", -1)) > 0
        and runtime_report.get("api_version") == runtime_lock.get("api_version")
        and runtime_report.get("definitions_path")
        == runtime_lock.get("definitions_path")
        and runtime_report.get("frozen_runner_sha256")
        == authorization["original_runner"]["sha256"]
        and runtime_report.get("recovery_module_sha256")
        == authorization["recovery_module"]["sha256"]
        and runtime_report.get("original_runtime_identity")
        == _expected_original_runtime_report(provenance["runtime_identity"]),
        "recovery runtime report semantic mismatch",
    )
    return {
        "runtime_file_identities": runtime_actuals,
        "runtime_report": dict(runtime_report),
    }


def _validate_recovery_pilot_report(report: Any, *, label: str) -> None:
    _require(isinstance(report, Mapping), f"{label} is absent")
    _require(
        set(report)
        == {
            "status",
            "workers_requested",
            "worker_pids_observed",
            "tasks",
            "first_wave_synchronized",
            "reference",
            "network_requests",
        },
        f"{label} schema mismatch",
    )
    pids = report.get("worker_pids_observed")
    reference = report.get("reference")
    _require(
        report.get("status")
        == "PASS_REAL_GRIB_STDLIB_BOOTSTRAP_PROCESS_ISOLATION"
        and int(report.get("workers_requested", -1)) == DECODE_MAX_WORKERS
        and isinstance(pids, list)
        and len(pids) == DECODE_MAX_WORKERS
        and len({int(pid) for pid in pids}) == DECODE_MAX_WORKERS
        and all(int(pid) > 0 for pid in pids)
        and int(report.get("tasks", -1)) == 70
        and report.get("first_wave_synchronized") is True
        and int(report.get("network_requests", -1)) == 0
        and isinstance(reference, Mapping)
        and set(reference) == {"metadata", "value_count", "value_min", "value_max"}
        and int(reference.get("value_count", -1)) == 1440 * 721,
        f"{label} semantic mismatch",
    )


def _validate_recovery_production_worker_report(report: Any) -> None:
    _require(isinstance(report, Mapping), "production worker report is absent")
    _require(
        set(report)
        == {
            "tasks",
            "worker_pids_observed",
            "worker_count",
            "first_wave_synchronized",
            "feature",
            "site_count",
            "site_values_binary64_sha256",
            "all_rows_array_equal",
            "network_requests",
        },
        "production worker report schema mismatch",
    )
    pids = report.get("worker_pids_observed")
    _require(
        int(report.get("tasks", -1)) == 70
        and isinstance(pids, list)
        and len(pids) == DECODE_MAX_WORKERS
        and len({int(pid) for pid in pids}) == DECODE_MAX_WORKERS
        and int(report.get("worker_count", -1)) == DECODE_MAX_WORKERS
        and report.get("first_wave_synchronized") is True
        and report.get("feature") == "HPBL_surface"
        and int(report.get("site_count", -1)) == 17
        and report.get("site_values_binary64_sha256")
        == "edeca661e457c32431218ccb50c5dbf0c428905a36e03c28cfe4d76d32ddcc0b"
        and report.get("all_rows_array_equal") is True
        and int(report.get("network_requests", -1)) == 0,
        "production worker report semantic mismatch",
    )


def _validate_recovery_test_evidence(
    evidence: Any,
    preflight: Mapping[str, Any],
    external_actuals: Mapping[str, Mapping[str, Any]],
    sealer_actuals: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(isinstance(evidence, Mapping), "immutable recovery test evidence absent")
    evidence_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "bound_code_and_test_identities",
        "source_compile_result",
        "compiled_source_identities",
        "pytest_isolation",
        "pytest_runs",
        "pytest_summary",
        "noncontract_monolithic_diagnostic",
        "isolated_subprocess_count",
        "all_pytest_exit_codes_zero",
        "pytest_child_network_guard_sha256",
        "pytest_child_network_guard_installed",
        "pytest_cacheprovider_disabled",
        "required_test_names",
        "required_test_names_present",
        "python_executable",
        "python_dont_write_bytecode_env",
        "python_dont_write_bytecode_flag",
        "python_pycache_prefix_absent",
        "bootstrap_matching_pyc_absent_before_tests",
        "bootstrap_matching_pyc_absent_after_tests",
        "network_requests",
        "labels_read",
        "arrays_2024_read",
        "arrays_2025_read",
        "models_fit",
        "submission_csv_created",
    }
    _require(set(evidence) == evidence_keys, "immutable recovery test-evidence schema mismatch")
    _require(
        int(evidence.get("schema_version", -1)) == 1
        and evidence.get("artifact_type")
        == "DECODE_RECOVERY_IMMUTABLE_CODE_TEST_EVIDENCE"
        and evidence.get("status") == "PASS_BOUND_CODE_COMPILE_AND_TEST_SUITE"
        and evidence.get("created_utc") == preflight.get("created_utc")
        and evidence.get("source_compile_result") == "PASS"
        and evidence.get("pytest_isolation")
        == "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS"
        and evidence.get("noncontract_monolithic_diagnostic")
        == RECOVERY_NONCONTRACT_MONOLITHIC_DIAGNOSTIC
        and int(evidence.get("isolated_subprocess_count", -1))
        == len(RECOVERY_TEST_RELATIVES)
        and evidence.get("all_pytest_exit_codes_zero") is True
        and evidence.get("pytest_child_network_guard_installed") is True
        and evidence.get("pytest_cacheprovider_disabled") is True
        and evidence.get("required_test_names") == RECOVERY_REQUIRED_TEST_NAMES
        and evidence.get("required_test_names_present") is True
        and evidence.get("python_dont_write_bytecode_env") == "1"
        and evidence.get("python_dont_write_bytecode_flag") is True
        and evidence.get("python_pycache_prefix_absent") is True
        and evidence.get("bootstrap_matching_pyc_absent_before_tests") is True
        and evidence.get("bootstrap_matching_pyc_absent_after_tests") is True
        and int(evidence.get("network_requests", -1)) == 0,
        "immutable recovery test-evidence gates mismatch",
    )
    require_zero_facts(
        evidence,
        {
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="immutable recovery test evidence",
    )

    expected_bound = {
        key: external_actuals[key]
        for key in (
            "original_runner",
            "recovery_runner",
            "recovery_module",
            "recovery_bootstrap",
            "recovery_test",
            "recovery_bootstrap_test",
            "recovery_runner_test",
            "postrun_auditor",
            "postrun_auditor_test",
        )
    }
    expected_bound.update(sealer_actuals)
    _require(
        evidence.get("bound_code_and_test_identities") == expected_bound,
        "immutable recovery test evidence code/test bindings mismatch",
    )

    compiled = evidence.get("compiled_source_identities")
    _require(
        isinstance(compiled, list)
        and len(compiled) == len(RECOVERY_COMPILED_RELATIVES),
        "immutable recovery compiled-source inventory mismatch",
    )
    for record, relative in zip(
        compiled, RECOVERY_COMPILED_RELATIVES, strict=True
    ):
        verify_identity(
            record,
            REPO / relative,
            root=None,
            label=f"immutable compiled source {relative}",
        )

    verify_identity(
        evidence.get("python_executable", {}),
        Path(sys.executable).resolve(),
        root=None,
        label="immutable-test Python executable",
    )
    runs = evidence.get("pytest_runs")
    _require(
        isinstance(runs, list) and len(runs) == len(RECOVERY_TEST_RELATIVES),
        "immutable recovery pytest-run inventory mismatch",
    )
    guard_sha = evidence.get("pytest_child_network_guard_sha256")
    _require(
        isinstance(guard_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", guard_sha) is not None,
        "immutable recovery pytest guard digest invalid",
    )
    _require(
        guard_sha == RECOVERY_PYTEST_NETWORK_GUARD_SHA256,
        "immutable recovery pytest guard is not the approved network-deny source",
    )
    summaries: list[str] = []
    executable_lexical = os.path.normcase(
        str(Path(os.path.abspath(sys.executable)))
    )
    for run, relative in zip(runs, RECOVERY_TEST_RELATIVES, strict=True):
        _require(
            isinstance(run, Mapping)
            and set(run)
            == {
                "test_file",
                "command",
                "exit_code",
                "summary",
                "stdout_sha256",
                "stderr_sha256",
            },
            f"immutable recovery pytest-run schema mismatch: {relative}",
        )
        command = run.get("command")
        _require(
            isinstance(command, list)
            and len(command) == 8
            and all(isinstance(part, str) for part in command),
            f"immutable recovery pytest command invalid: {relative}",
        )
        command_executable = os.path.normcase(
            str(Path(os.path.abspath(command[0])))
        )
        _require(
            command_executable == executable_lexical
            and command[1:3] == ["-B", "-c"]
            and command[4:] == ["-q", "-p", "no:cacheprovider", relative]
            and hashlib.sha256(command[3].encode("utf-8")).hexdigest()
            == guard_sha,
            f"immutable recovery pytest command boundary mismatch: {relative}",
        )
        summary = run.get("summary")
        _require(
            run.get("test_file") == relative
            and type(run.get("exit_code")) is int
            and run.get("exit_code") == 0
            and isinstance(summary, str)
            and re.fullmatch(
                r"[0-9]+ passed(?:, [0-9]+ skipped)? in [0-9]+(?:\.[0-9]+)?s",
                summary,
            )
            is not None
            and all(
                isinstance(run.get(field), str)
                and re.fullmatch(r"[0-9a-f]{64}", run[field]) is not None
                for field in ("stdout_sha256", "stderr_sha256")
            ),
            f"immutable recovery pytest result invalid: {relative}",
        )
        summaries.append(f"{relative}: {summary}")
    _require(
        evidence.get("pytest_summary") == " | ".join(summaries),
        "immutable recovery pytest summary mismatch",
    )


def _audit_recovery_raw_prestate(
    root: Path,
    authorization: Mapping[str, Any],
    transaction: Mapping[str, Any],
    events: Mapping[str, Any],
    raw_info: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    _require(
        events.get("starts") == contract.range_rows
        and events.get("completions") == contract.range_rows
        and events.get("errors") == 0
        and events.get("indeterminate_starts") == 0
        and events.get("max_global_attempt_number") == contract.range_rows,
        "recovery input request-event ledger is not exact no-retry/no-error cache",
    )
    selected_event_paths: set[Path] = set()
    raw_paths: list[Path] = []
    for key, info in raw_info.items():
        meta = info.get("meta")
        _require(isinstance(meta, Mapping), f"recovery raw sidecar absent: {key}")
        evidence = meta.get("request_completion_evidence")
        _require(
            isinstance(evidence, Mapping)
            and set(evidence)
            == {
                "selected_complete_event",
                "selected_start_event",
                "semantically_identical_completion_event_count",
            }
            and int(evidence.get("semantically_identical_completion_event_count", -1))
            == 1
            and int(meta.get("resumed_from_bytes", -1)) == 0
            and "resume_prefix_evidence" not in meta,
            f"recovery raw cache is not exact no-retry/no-resume: {key}",
        )
        for event_role in ("selected_start_event", "selected_complete_event"):
            record = evidence[event_role]
            _require(isinstance(record, Mapping), f"recovery sidecar {event_role} invalid")
            selected_event_paths.add(path_under(root, str(record.get("path", ""))))
        raw_path = Path(info["path"])
        raw_paths.extend((raw_path, raw_path.with_suffix(".grib2.meta.json")))
    event_paths = {
        path_under(root, str(record["path"])) for record in events["event_identities"]
    }
    _require(
        selected_event_paths == event_paths
        and len(event_paths) == contract.range_rows * 2,
        "recovery selected request-event inventory is not globally exact",
    )
    for event_path in sorted(event_paths):
        if not event_path.name.endswith("_start.json"):
            continue
        start_payload = load_json(event_path)
        _require(
            all(
                type(start_payload.get(field)) is int
                for field in (
                    "attempt",
                    "global_raw_attempt_number",
                    "range_start",
                    "range_end",
                )
            )
            and start_payload.get("attempt") == 1,
            f"recovery original START integer/attempt mismatch: {event_path.name}",
        )
        _parse_exact_utc_z(
            start_payload.get("created_utc"), label="original START created_utc"
        )

    failed_attempt = str(transaction["failed_attempt_id"])
    progress_root = root / "raw" / "progress"
    _require(
        progress_root.is_dir() and not _linklike(progress_root),
        "original raw progress root absent or a link",
    )
    progress_entries = list(progress_root.rglob("*"))
    _require(
        not any(_linklike(path) for path in progress_entries)
        and all(path.is_file() for path in progress_entries),
        "original raw progress inventory contains a link/directory/special object",
    )
    progress_counts = recovery_progress_counts(contract)
    expected_progress_paths = {
        progress_root / f"raw_ranges__{failed_attempt}__{count:06d}.json"
        for count in progress_counts
    }
    actual_progress_paths = {path for path in progress_entries if path.is_file()}
    _require(
        actual_progress_paths == expected_progress_paths,
        "original raw progress checkpoint inventory mismatch",
    )
    previous_bytes = -1
    previous_attempts = -1
    prior_checkpoint_created: datetime | None = None
    ordered_raw_infos = list(raw_info.values())
    start_times: list[datetime] = []
    for attempt in events["attempt_inventory"]:
        started = _parse_exact_utc_z(
            attempt.get("start_created_utc"), label="request START created_utc"
        )
        start_times.append(started)
    failed_lock = (
        root / "raw" / "launch_history" / f"{failed_attempt}__failed.lock"
    )
    failed_payload = load_json(failed_lock)
    _require(
        set(failed_payload) == {"attempt_id", "pid", "created_utc"}
        and failed_payload.get("attempt_id") == failed_attempt
        and type(failed_payload.get("pid")) is int
        and failed_payload.get("pid") == 3536,
        "original failed-lock exact payload mismatch",
    )
    failed_created = _parse_exact_utc_z(
        failed_payload.get("created_utc"), label="original failed-lock created_utc"
    )
    _require(
        min(start_times) > failed_created,
        "original request START does not strictly follow failed-attempt launch lock",
    )
    payload_sequence: list[dict[str, Any]] = []
    filename_sequence: list[str] = []
    for count in progress_counts:
        progress_path = (
            progress_root / f"raw_ranges__{failed_attempt}__{count:06d}.json"
        )
        checkpoint = load_json(progress_path)
        _require(
            set(checkpoint)
            == {
                "phase",
                "attempt_id",
                "completed_ranges",
                "completed_key_set_sha256",
                "network_requests_this_invocation_so_far",
                "network_bytes_this_invocation_so_far",
                "actual_http_attempts_cumulative",
                "checkpoint_name",
                "created_utc",
            },
            f"original raw progress schema mismatch: {progress_path.name}",
        )
        current_bytes = int(checkpoint.get("network_bytes_this_invocation_so_far", -1))
        current_attempts = int(checkpoint.get("actual_http_attempts_cumulative", -1))
        parsed_created = _parse_exact_utc_z(
            checkpoint.get("created_utc"), label="original progress created_utc"
        )
        prefix_infos = ordered_raw_infos[:count]
        expected_prefix_bytes = sum(int(info["size_bytes"]) for info in prefix_infos)
        prefix_keys = [
            "|".join(
                (
                    str(info["row"]["object_key"]),
                    str(int(info["row"]["forecast_hour"])),
                    str(info["row"]["variable"]),
                    str(info["row"]["level"]),
                )
            )
            for info in prefix_infos
        ]
        expected_prefix_digest = hashlib.sha256(
            ("\n".join(sorted(prefix_keys)) + "\n").encode("utf-8")
        ).hexdigest()
        attempts_at_checkpoint = sum(started <= parsed_created for started in start_times)
        _require(
            all(
                type(checkpoint.get(field)) is int
                for field in (
                    "completed_ranges",
                    "network_requests_this_invocation_so_far",
                    "network_bytes_this_invocation_so_far",
                    "actual_http_attempts_cumulative",
                )
            )
            and
            checkpoint.get("phase") == "RAW_RANGE_DOWNLOAD"
            and checkpoint.get("attempt_id") == failed_attempt
            and int(checkpoint.get("completed_ranges", -1)) == count
            and int(checkpoint.get("network_requests_this_invocation_so_far", -1))
            == count
            and previous_attempts <= current_attempts
            and count <= current_attempts <= min(contract.range_rows, count + 7)
            and current_attempts == attempts_at_checkpoint
            and current_bytes == expected_prefix_bytes
            and previous_bytes <= current_bytes <= contract.range_bytes
            and checkpoint.get("checkpoint_name") == progress_path.stem
            and parsed_created > failed_created
            and (
                prior_checkpoint_created is None
                or parsed_created > prior_checkpoint_created
            )
            and checkpoint.get("completed_key_set_sha256") == expected_prefix_digest,
            f"original raw progress value mismatch: {progress_path.name}",
        )
        previous_bytes = current_bytes
        previous_attempts = current_attempts
        prior_checkpoint_created = parsed_created
        payload_sequence.append(checkpoint)
        filename_sequence.append(progress_path.name)
    final_progress = load_json(
        progress_root
        / f"raw_ranges__{failed_attempt}__{contract.range_rows:06d}.json"
    )
    completed_keys = [
        "|".join(
            (
                str(info["row"]["object_key"]),
                str(int(info["row"]["forecast_hour"])),
                str(info["row"]["variable"]),
                str(info["row"]["level"]),
            )
        )
        for info in raw_info.values()
    ]
    expected_key_digest = hashlib.sha256(
        ("\n".join(sorted(completed_keys)) + "\n").encode("utf-8")
    ).hexdigest()
    _require(
        int(final_progress.get("network_bytes_this_invocation_so_far", -1))
        == contract.range_bytes
        and int(final_progress.get("actual_http_attempts_cumulative", -1))
        == contract.range_rows
        and final_progress.get("completed_key_set_sha256") == expected_key_digest,
        "original final raw progress byte/key digest mismatch",
    )
    final_progress_path = (
        progress_root
        / f"raw_ranges__{failed_attempt}__{contract.range_rows:06d}.json"
    )
    verify_identity(
        authorization.get("original_final_progress", {}),
        final_progress_path,
        root=root,
        label="authorized original final raw progress",
    )
    progress_semantics = {
        "progress_checkpoint_count": len(payload_sequence),
        "checkpoint_completed_sequence_sha256": canonical_payload_sha256(
            progress_counts
        ),
        "checkpoint_filename_sequence_sha256": canonical_payload_sha256(
            filename_sequence
        ),
        "progress_payload_sequence_sha256": canonical_payload_sha256(
            payload_sequence
        ),
        "first_checkpoint_created_utc": payload_sequence[0]["created_utc"],
        "final_checkpoint_created_utc": final_progress["created_utc"],
        "final_completed_ranges": contract.range_rows,
        "final_network_requests": contract.range_rows,
        "final_network_bytes": contract.range_bytes,
        "final_actual_http_attempts": contract.range_rows,
        "final_completed_key_set_sha256": final_progress[
            "completed_key_set_sha256"
        ],
        "start_event_count": contract.range_rows,
        "start_global_attempts_exact_1_through_10368": True,
        "attempt_count_reconstructed_from_start_event_timestamps": True,
        "failed_lock_payload_sha256": canonical_payload_sha256(failed_payload),
        "failed_lock_attempt_id": failed_attempt,
        "failed_lock_pid": 3536,
    }
    inventories = {
        "raw_ranges_and_sidecars": _recovery_file_inventory(raw_paths, root),
        "request_events": _recovery_file_inventory(event_paths, root),
        "original_progress": _recovery_file_inventory(actual_progress_paths, root),
        "original_launch_history": _recovery_file_inventory([failed_lock], root),
        "original_progress_and_failed_lock_semantics": progress_semantics,
        "validated_raw_payload_bytes": contract.range_bytes,
        "raw_file_count": contract.range_rows,
        "sidecar_file_count": contract.range_rows,
        "request_event_file_count": contract.range_rows * 2,
    }
    return {
        "inventories": inventories,
        "final_raw_progress": final_progress,
        "final_raw_progress_identity": identity(final_progress_path, root),
        "original_progress_checkpoint_count": len(actual_progress_paths),
    }


def _reconstruct_recovery_fixed_input_snapshot(
    root: Path,
    authorization: Mapping[str, Any],
    go: Mapping[str, Any],
    failed_attempt: str,
    contract: AuditContract,
    external_paths: Mapping[str, Path],
) -> dict[str, Any]:
    authorization_path = root / RECOVERY_AUTH_RELATIVE
    go_path = root / RECOVERY_GO_RELATIVE
    root_records = [identity(authorization_path, root), identity(go_path, root)]
    fixed_relatives = _recovery_fixed_root_relatives(
        failed_attempt, contract.range_rows
    )
    for key, relative in sorted(fixed_relatives.items()):
        path = path_under(root, relative)
        verify_identity(
            authorization.get(key, {}), path, root=root, label=f"snapshot {key}"
        )
        root_records.append(identity(path, root))
    for role, record in sorted(RECOVERY_V1_CHAIN_IDENTITIES.items()):
        if role == "recovery_runner":
            continue
        if contract == PRODUCTION_CONTRACT:
            path = path_under(root, str(record["path"]))
            verify_identity(
                record,
                path,
                root=root,
                label=f"snapshot authoritative V1 {role}",
            )
            root_records.append(identity(path, root))
        else:
            # Synthetic fixtures have no live V1 control plane.  Preserve the
            # exact declared root identity row without dereferencing it.
            root_records.append(dict(record))
    remnants = authorization.get("documented_preplan_remnants")
    _require(isinstance(remnants, list), "snapshot remnant list absent")
    for record in remnants:
        _require(isinstance(record, Mapping), "snapshot remnant identity invalid")
        path = path_under(root, str(record.get("path", "")))
        verify_identity(record, path, root=root, label="snapshot preplan remnant")
        root_records.append(identity(path, root))
    review_path = root / RECOVERY_REVIEW_RELATIVE
    verify_identity(
        go.get("independent_review", {}),
        review_path,
        root=root,
        label="snapshot recovery review",
    )
    root_records.append(identity(review_path, root))

    external_records: list[dict[str, Any]] = []
    for key in (
        "original_runner",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "real_grib_pilot",
    ):
        expected = external_paths[key]
        verify_identity(
            authorization.get(key, {}),
            expected,
            root=None,
            label=f"snapshot external {key}",
        )
        external_records.append(identity(expected.resolve()))
    support_records = authorization.get("superseded_v1_support")
    _require(
        isinstance(support_records, Mapping)
        and support_records == _authoritative_v1_support_external(),
        "snapshot authoritative V1 support map mismatch",
    )
    for role, record in sorted(support_records.items()):
        expected = Path(str(record["path"]))
        verify_identity(
            record,
            expected,
            root=None,
            label=f"snapshot authoritative V1 support {role}",
        )
        external_records.append(identity(expected.resolve()))
    runtime_lock = authorization["runtime_lock"]
    for key, expected in _recovery_runtime_file_paths().items():
        verify_identity(
            runtime_lock.get(key, {}),
            expected,
            root=None,
            label=f"snapshot runtime {key}",
        )
        external_records.append(identity(expected.resolve()))
    original_runtime = authorization["original_runtime_identity"]
    executable = Path(str(original_runtime["python_executable"])).resolve()
    verify_identity(
        {
            "path": original_runtime["python_executable"],
            "size_bytes": original_runtime["python_executable_size_bytes"],
            "sha256": original_runtime["python_executable_sha256"],
        },
        executable,
        root=None,
        label="snapshot original Python executable",
    )
    external_records.append(identity(executable))
    packages = original_runtime["packages"]
    _require(isinstance(packages, Mapping), "snapshot original package map invalid")
    for name, record in sorted(packages.items()):
        _require(isinstance(record, Mapping), f"snapshot package invalid: {name}")
        module_path = Path(str(record["module_file"])).resolve()
        verify_identity(
            {
                "path": record["module_file"],
                "size_bytes": record["module_file_size_bytes"],
                "sha256": record["module_file_sha256"],
            },
            module_path,
            root=None,
            label=f"snapshot original runtime package {name}",
        )
        external_records.append(identity(module_path))
    root_records.sort(key=lambda row: str(row["path"]))
    external_records.sort(key=lambda row: str(row["path"]))
    return {
        "root_file_count": len(root_records),
        "root_files_sha256": canonical_payload_sha256(root_records),
        "external_file_count": len(external_records),
        "external_files_sha256": canonical_payload_sha256(external_records),
        "combined_sha256": canonical_payload_sha256(
            {"root": root_records, "external": external_records}
        ),
    }


def _audit_recovery_control_plane(
    root: Path,
    provenance: Mapping[str, Any],
    transaction: Mapping[str, Any],
    events: Mapping[str, Any],
    raw_info: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    authorization_path = root / RECOVERY_AUTH_RELATIVE
    go_path = root / RECOVERY_GO_RELATIVE
    review_path = root / RECOVERY_REVIEW_RELATIVE
    incident_path = root / RECOVERY_INCIDENT_RELATIVE
    entrypoint_incident_path = root / RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    flawed_launch_incident_path = root / RECOVERY_FLAWED_LAUNCH_INCIDENT_RELATIVE
    launch_identity_correction_path = (
        root / RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE
    )
    raw_cache_lock_path = root / RECOVERY_RAW_CACHE_LOCK_RELATIVE
    preflight_path = root / RECOVERY_REAL_PREFLIGHT_RELATIVE
    for path, label in (
        (authorization_path, "recovery authorization"),
        (go_path, "recovery independent GO"),
        (review_path, "recovery independent review"),
        (incident_path, "recovery incident"),
        (entrypoint_incident_path, "recovery sealer-entrypoint incident"),
        (flawed_launch_incident_path, "preserved flawed V1 launch incident"),
        (launch_identity_correction_path, "authoritative V1 launch correction"),
        (raw_cache_lock_path, "recovery raw-cache lock"),
        (preflight_path, "recovery process preflight"),
    ):
        require_no_symlink_chain(path, root, label=label)
        _require(not _linklike(path), f"{label} link is forbidden")
    authorization = load_json(authorization_path)
    go = load_json(go_path)
    review = load_json(review_path)
    incident = load_json(incident_path)
    entrypoint_incident = load_json(entrypoint_incident_path)
    flawed_launch_incident = load_json(flawed_launch_incident_path)
    launch_identity_correction = load_json(launch_identity_correction_path)
    raw_cache_lock = load_json(raw_cache_lock_path)
    preflight = load_json(preflight_path)
    failed_attempt = str(transaction["failed_attempt_id"])
    fixed_relatives = _recovery_fixed_root_relatives(
        failed_attempt, contract.range_rows
    )
    authorization_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "experiment_id",
        "recovery_attempt_id",
        "incident",
        "superseded_v1_chain",
        "superseded_v1_support",
        "original_runner",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        *fixed_relatives.keys(),
        "documented_preplan_remnants",
        "runtime_lock",
        "bootstrap_trust_policy",
        "bootstrap_import_contract",
        "real_grib_pilot",
        "original_runtime_identity",
        "expected_range_rows",
        "expected_range_bytes",
        "max_decode_processes",
        "network_requests_allowed",
        "raw_event_progress_mutation_allowed",
        "labels_read",
        "2024_arrays_read",
        "2025_arrays_read",
        "models_fit",
        "submission_csv_created",
    }
    _require(set(authorization) == authorization_keys, "recovery authorization schema mismatch")
    _require(
        int(authorization.get("schema_version", -1)) == 2
        and authorization.get("artifact_type")
        == "NOAA_GFS_NETWORK_ZERO_DECODE_RECOVERY_AUTHORIZATION_V2"
        and authorization.get("status") == "AUTHORIZED_PENDING_INDEPENDENT_GO_V2"
        and authorization.get("experiment_id")
        == "noaa_gfs_dminus2_12z_multiseason_target_free_decode_recovery_v2"
        and authorization.get("recovery_attempt_id") == transaction["attempt_id"]
        and int(authorization.get("expected_range_rows", -1)) == contract.range_rows
        and int(authorization.get("expected_range_bytes", -1)) == contract.range_bytes
        and int(authorization.get("max_decode_processes", -1)) == DECODE_MAX_WORKERS
        and int(authorization.get("network_requests_allowed", -1)) == 0
        and authorization.get("raw_event_progress_mutation_allowed") is False
        and authorization.get("bootstrap_trust_policy")
        == RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "recovery authorization policy mismatch",
    )
    expected_bootstrap_import_contract = (
        _validate_recovery_bootstrap_import_contract(
            authorization.get("bootstrap_import_contract")
        )
    )
    current_bootstrap_pyc_gate = (
        _audit_current_recovery_bootstrap_pyc_absence()
    )
    require_zero_facts(
        authorization,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="recovery authorization",
    )
    for key, relative in fixed_relatives.items():
        verify_identity(
            authorization.get(key, {}),
            path_under(root, relative),
            root=root,
            label=f"recovery-authorized {key}",
        )
    authoritative_v1_support_external = _authoritative_v1_support_external()
    _require(
        authorization.get("incident") == transaction["recovery_incident_identity"]
        and authorization.get("flawed_launch_incident")
        == transaction["flawed_launch_incident_identity"]
        and authorization.get("launch_identity_correction")
        == transaction["launch_identity_correction_identity"]
        and authorization.get("superseded_v1_chain")
        == RECOVERY_V1_CHAIN_IDENTITIES
        and authorization.get("superseded_v1_support")
        == authoritative_v1_support_external
        and authorization.get("original_authorization")
        == provenance["authorization_identity"]
        and authorization.get("original_independent_go") == provenance["go_identity"]
        and authorization.get("original_runner") == provenance["runner_identity"]
        and authorization.get("field_range_census")
        == identity(Path(provenance["field_census_path"]), root)
        and authorization.get("census_manifest")
        == provenance["census_manifest_identity"]
        and authorization.get("coordinate_lock") == provenance["coordinate_identity"]
        and authorization.get("original_runtime_identity")
        == provenance["runtime_identity"],
        "recovery authorization differs from frozen original provenance",
    )

    external_paths = {
        "original_runner": Path(str(provenance["runner_path"])),
        "recovery_runner": RECOVERY_RUNNER,
        "recovery_module": RECOVERY_MODULE,
        "recovery_bootstrap": RECOVERY_BOOTSTRAP,
        "recovery_test": RECOVERY_TEST,
        "recovery_bootstrap_test": RECOVERY_BOOTSTRAP_TEST,
        "recovery_runner_test": RECOVERY_RUNNER_TEST,
        "postrun_auditor": SUPERSEDED_V2_AUDITOR,
        "postrun_auditor_test": SUPERSEDED_V2_AUDITOR_TEST,
        "real_grib_pilot": RECOVERY_REAL_PILOT,
    }
    external_actuals: dict[str, dict[str, Any]] = {}
    for key, expected_path in external_paths.items():
        external_actuals[key] = verify_identity(
            authorization.get(key, {}),
            expected_path,
            root=None,
            label=f"recovery-authorized {key}",
        )
        if key in RECOVERY_FROZEN_EXTERNAL_SHA256:
            _require(
                external_actuals[key]["sha256"]
                == RECOVERY_FROZEN_EXTERNAL_SHA256[key],
                f"frozen recovery external constant mismatch: {key}",
            )

    _require(
        set(incident)
        == {
            "schema_version",
            "artifact_type",
            "created_kst",
            "status",
            "failed_attempt_id",
            "failure_boundary",
            "cause",
            "immutable_failure_evidence",
            "documented_preplan_staging_remnants",
            "preservation_policy",
            "recovery_gate",
        },
        "recovery incident schema mismatch",
    )
    _require(
        int(incident.get("schema_version", -1)) == 1
        and incident.get("artifact_type")
        == "TRACK_A_RAW_DECODE_MEMFS_THREAD_INIT_FAILURE_INCIDENT"
        and incident.get("status")
        == "RAW_DOWNLOAD_COMPLETE_DECODE_FAILED_RECOVERY_NOT_AUTHORIZED"
        and incident.get("failed_attempt_id") == failed_attempt
        and incident.get("failure_boundary")
        == {
            "raw_ranges_completed": contract.range_rows,
            "raw_bytes_completed": contract.range_bytes,
            "successful_network_requests": contract.range_rows,
            "actual_http_attempts": contract.range_rows,
            "decoded_messages_completed": 0,
            "canonical_outputs_committed": 0,
            "active_lock_present_after_failure": False,
            "complete_lock_present_after_failure": False,
        }
        and incident.get("recovery_gate", {}).get("currently_authorized") is False,
        "recovery incident semantic boundary mismatch",
    )
    evidence_by_path = {
        str(record.get("path")): record
        for record in incident.get("immutable_failure_evidence", [])
        if isinstance(record, Mapping)
    }
    expected_evidence = {
        fixed_relatives[key]: authorization[key]
        for key in (
            "original_failed_lock",
            "original_final_progress",
            "original_stderr",
            "original_stdout",
        )
    }
    _require(
        len(evidence_by_path) == 4 and evidence_by_path == expected_evidence,
        "incident immutable failure evidence mismatch",
    )
    entrypoint_incident_identity = identity(entrypoint_incident_path, root)
    _require(
        authorization.get("sealer_entrypoint_incident")
        == entrypoint_incident_identity,
        "recovery authorization sealer-entrypoint incident mismatch",
    )
    _validate_recovery_sealer_entrypoint_incident(
        entrypoint_incident,
        entrypoint_incident_identity,
        root,
        authorization,
        contract,
    )
    flawed_launch_incident_identity = identity(flawed_launch_incident_path, root)
    launch_identity_correction_identity = identity(
        launch_identity_correction_path, root
    )
    _require(
        authorization.get("flawed_launch_incident")
        == flawed_launch_incident_identity
        and authorization.get("launch_identity_correction")
        == launch_identity_correction_identity,
        "V2 authorization launch-incident/correction identity mismatch",
    )
    if contract == PRODUCTION_CONTRACT:
        _require(
            flawed_launch_incident_identity["size_bytes"] == 7_172
            and flawed_launch_incident_identity["sha256"]
            == RECOVERY_FLAWED_LAUNCH_INCIDENT_SHA256
            and launch_identity_correction_identity["size_bytes"] == 16_374
            and launch_identity_correction_identity["sha256"]
            == RECOVERY_LAUNCH_IDENTITY_CORRECTION_SHA256,
            "production V1 launch supersession/correction constant mismatch",
        )
    _validate_v2_launch_supersession_incident(flawed_launch_incident)
    _validate_v2_launch_identity_correction(
        launch_identity_correction,
        expected_flawed_identity=flawed_launch_incident_identity,
    )
    _require(
        launch_identity_correction.get("superseded_flawed_incident")
        == flawed_launch_incident_identity,
        "authoritative correction does not bind preserved flawed incident",
    )
    preserved = launch_identity_correction["preserved_failure_evidence"]
    preserved_digests = launch_identity_correction[
        "preserved_failure_evidence_canonical_sha256"
    ]
    for key, value in preserved.items():
        _require(
            flawed_launch_incident.get(key) == value
            and canonical_payload_sha256(value) == preserved_digests.get(key),
            f"authoritative correction changed preserved failure evidence: {key}",
        )

    # The correction is the sole authority for the failed V1 chain.  The
    # flawed incident is retained as evidence only and its transient 82944
    # runner identity must never be selected.
    if contract == PRODUCTION_CONTRACT:
        for role, record in RECOVERY_V1_CHAIN_IDENTITIES.items():
            if role == "recovery_runner":
                absolute_runner = authoritative_v1_support_external[
                    "recovery_runner"
                ]
                _require(
                    absolute_runner["size_bytes"] == record["size_bytes"]
                    and absolute_runner["sha256"] == record["sha256"],
                    "authoritative failed V1 runner/support identity mismatch",
                )
                verify_identity(
                    absolute_runner,
                    REPO / str(record["path"]),
                    root=None,
                    label="authoritative failed V1 runner",
                )
            else:
                verify_identity(
                    record,
                    path_under(root, str(record["path"])),
                    root=root,
                    label=f"authoritative failed V1 {role}",
                )
        for role, record in authoritative_v1_support_external.items():
            verify_identity(
                record,
                Path(str(record["path"])),
                root=None,
                label=f"authoritative V1 support {role}",
            )
    _require(
        RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]
        != RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY
        and authorization.get("recovery_runner", {}).get("sha256")
        != RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY["sha256"],
        "rejected transient 82944 V1 runner selected as authority",
    )

    runtime_validation = _validate_recovery_runtime(
        authorization.get("runtime_lock"),
        transaction["manifest"].get("runtime_identity"),
        authorization,
        provenance,
    )

    preflight_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "incident",
        "flawed_launch_incident",
        "launch_identity_correction",
        "superseded_v1_chain",
        "superseded_v1_support",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "recovery_sealer",
        "recovery_sealer_test",
        "bootstrap_trust_policy",
        "runtime_lock",
        "bootstrap_import_contract",
        "test_evidence",
        "process_scheduling_flake_incident",
        "sealer_entrypoint_incident",
        "real_grib_pilot",
        "max_spawn_processes",
        "thread_decoders",
        "network_requests",
        "production_decode_task_worker_real_test",
        "stdlib_bootstrap_worker_real_test",
        "poisoned_pyc_source_execution_test",
        "bootstrap_ast_stdlib_policy_pass",
        "python_dont_write_bytecode_env",
        "python_dont_write_bytecode_flag",
        "python_pycache_prefix_absent",
        "bootstrap_matching_pyc_absent_before_tests",
        "bootstrap_matching_pyc_absent_after_tests",
        "all_seven_worker_pids_observed",
        "pilot_report",
        "production_worker_report",
        "run_prefix_regression",
        "focused_test_result",
        "source_compile_result",
        "labels_read",
        "arrays_2024_read",
        "arrays_2025_read",
        "models_fit",
        "submission_csv_created",
    }
    _require(set(preflight) == preflight_keys, "recovery process-preflight schema mismatch")
    for key in (
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
    ):
        _require(preflight.get(key) == authorization.get(key), f"preflight {key} mismatch")
    sealer_paths = {
        "recovery_sealer": RECOVERY_SEALER,
        "recovery_sealer_test": RECOVERY_SEALER_TEST,
    }
    sealer_actuals: dict[str, dict[str, Any]] = {}
    for key, expected_path in sealer_paths.items():
        sealer_actuals[key] = verify_identity(
            preflight.get(key, {}),
            expected_path,
            root=None,
            label=f"recovery preflight {key}",
        )
        if key in RECOVERY_FROZEN_EXTERNAL_SHA256:
            _require(
                sealer_actuals[key]["sha256"]
                == RECOVERY_FROZEN_EXTERNAL_SHA256[key],
                f"frozen recovery sealer constant mismatch: {key}",
            )
    _validate_recovery_test_evidence(
        preflight.get("test_evidence"),
        preflight,
        external_actuals,
        sealer_actuals,
    )
    _require(
        int(preflight.get("schema_version", -1)) == 2
        and preflight.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2"
        and preflight.get("status")
        == "PASS_V2_REAL_PILOT_PRODUCTION_AND_RUN_PREFIX_EXACT_7_SPAWN"
        and preflight.get("incident") == authorization.get("incident")
        and preflight.get("flawed_launch_incident")
        == authorization.get("flawed_launch_incident")
        and preflight.get("launch_identity_correction")
        == authorization.get("launch_identity_correction")
        and preflight.get("superseded_v1_chain")
        == authorization.get("superseded_v1_chain")
        and preflight.get("superseded_v1_support")
        == authorization.get("superseded_v1_support")
        and preflight.get("bootstrap_trust_policy")
        == RECOVERY_BOOTSTRAP_TRUST_POLICY
        and preflight.get("runtime_lock") == authorization.get("runtime_lock")
        and preflight.get("bootstrap_import_contract")
        == expected_bootstrap_import_contract
        and preflight.get("process_scheduling_flake_incident")
        == RECOVERY_PROCESS_SCHEDULING_FLAKE_INCIDENT
        and preflight.get("run_prefix_regression")
        == {
            "status": "PASS_V2_RUN_REACHED_POSTPILOT_ALIAS_CHECK_BEFORE_CLAIM",
            "run_function_called": True,
            "shared_preclaim_function_called": True,
            "postpilot_bootstrap_alias_check_reached": True,
            "recovery_claim_acquired": False,
            "target_writes": 0,
            "network_requests": 0,
            "dedicated_test_name": (
                "test_v2_run_prefix_reaches_postpilot_alias_check_before_claim_and_writes_nothing"
            ),
        }
        and preflight.get("sealer_entrypoint_incident")
        == entrypoint_incident_identity
        and preflight.get("real_grib_pilot") == authorization.get("real_grib_pilot")
        and int(preflight.get("max_spawn_processes", -1)) == DECODE_MAX_WORKERS
        and int(preflight.get("thread_decoders", -1)) == 0
        and int(preflight.get("network_requests", -1)) == 0
        and all(
            preflight.get(key) is True
            for key in (
                "production_decode_task_worker_real_test",
                "stdlib_bootstrap_worker_real_test",
                "poisoned_pyc_source_execution_test",
                "bootstrap_ast_stdlib_policy_pass",
                "python_dont_write_bytecode_flag",
                "python_pycache_prefix_absent",
                "bootstrap_matching_pyc_absent_before_tests",
                "bootstrap_matching_pyc_absent_after_tests",
                "all_seven_worker_pids_observed",
            )
        )
        and preflight.get("python_dont_write_bytecode_env") == "1"
        and preflight.get("focused_test_result")
        == preflight["test_evidence"]["pytest_summary"]
        and preflight.get("source_compile_result")
        == preflight["test_evidence"]["source_compile_result"],
        "recovery process-preflight evidence mismatch",
    )
    require_zero_facts(
        preflight,
        {
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="recovery process preflight",
    )
    _validate_recovery_pilot_report(preflight.get("pilot_report"), label="sealed pilot report")
    _validate_recovery_production_worker_report(preflight.get("production_worker_report"))

    go_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "authorization",
        "incident",
        "flawed_launch_incident",
        "launch_identity_correction",
        "superseded_v1_chain",
        "superseded_v1_support",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "bootstrap_trust_policy",
        "raw_cache_lock",
        "real_spawn_preflight",
        "independent_review",
        "network_requests_allowed",
        "max_decode_processes",
    }
    _require(set(go) == go_keys, "independent recovery GO schema mismatch")
    _require(
        int(go.get("schema_version", -1)) == 2
        and go.get("artifact_type") == "TRACK_A_DECODE_RECOVERY_INDEPENDENT_GO_V2"
        and go.get("status") == "GO_V2_NETWORK_ZERO_PROCESS_ISOLATED_DECODE_RECOVERY"
        and int(go.get("network_requests_allowed", -1)) == 0
        and int(go.get("max_decode_processes", -1)) == DECODE_MAX_WORKERS,
        "independent recovery GO policy mismatch",
    )
    verify_identity(
        go.get("authorization", {}),
        authorization_path,
        root=root,
        label="GO recovery authorization",
    )
    verify_identity(
        go.get("independent_review", {}),
        review_path,
        root=root,
        label="GO independent recovery review",
    )
    for key in (
        "incident",
        "flawed_launch_incident",
        "launch_identity_correction",
        "superseded_v1_chain",
        "superseded_v1_support",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "bootstrap_trust_policy",
        "raw_cache_lock",
        "real_spawn_preflight",
    ):
        _require(go.get(key) == authorization.get(key), f"recovery GO {key} mismatch")

    review_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "authorization",
        "incident",
        "flawed_launch_incident",
        "launch_identity_correction",
        "superseded_v1_chain",
        "superseded_v1_support",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "recovery_sealer",
        "recovery_sealer_test",
        "bootstrap_trust_policy",
        "raw_cache_lock",
        "real_spawn_preflight",
        "max_decode_processes",
        "network_requests_allowed",
        "verdict",
        "independent_checks",
    }
    _require(set(review) == review_keys, "independent recovery review schema mismatch")
    review_checks = {
        "incident_authorization_and_go_chain_exact",
        "recovery_code_test_sealer_identities_exact",
        "runtime_memfs_and_bootstrap_trust_policy_exact",
        "raw_range_event_progress_and_failed_lock_closure_exact",
        "prerecovery_topology_remnants_and_canonical_absence_exact",
        "network_zero_target_free_scope_exact",
        "spawn_worker_core_source_binding_exact",
        "generic_resume_path_absent",
        "transaction_create_if_absent_no_overwrite_exact",
        "real_process_and_test_evidence_exact",
        "postrun_recovery_branch_afterstate_coverage_exact",
        "bootstrap_trust_anchor_limitation_recorded",
        "launch_nameerror_incident_exact",
        "v1_chain_immutable_and_not_selected",
        "v2_run_prefix_regression_reached_postpilot_before_claim",
    }
    _require(
        int(review.get("schema_version", -1)) == 2
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V2"
        and review.get("status")
        == "PASS_V2_NETWORK_ZERO_PROCESS_ISOLATED_RECOVERY_AUTHORIZED"
        and review.get("verdict") == "GO"
        and int(review.get("max_decode_processes", -1)) == DECODE_MAX_WORKERS
        and int(review.get("network_requests_allowed", -1)) == 0
        and isinstance(review.get("independent_checks"), Mapping)
        and set(review["independent_checks"]) == review_checks
        and all(review["independent_checks"].get(key) is True for key in review_checks),
        "independent recovery review verdict/check mismatch",
    )
    for key in (
        "authorization",
        "incident",
        "flawed_launch_incident",
        "launch_identity_correction",
        "superseded_v1_chain",
        "superseded_v1_support",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "bootstrap_trust_policy",
        "raw_cache_lock",
        "real_spawn_preflight",
    ):
        _require(review.get(key) == go.get(key), f"recovery review {key} mismatch")
    for key in ("recovery_sealer", "recovery_sealer_test"):
        _require(review.get(key) == preflight.get(key), f"recovery review {key} mismatch")

    raw_prestate = _audit_recovery_raw_prestate(
        root, authorization, transaction, events, raw_info, contract
    )
    _require(
        set(raw_cache_lock)
        == {
            "schema_version",
            "artifact_type",
            "status",
            "created_utc",
            "incident",
            "flawed_launch_incident",
            "launch_identity_correction",
            "superseded_v1_chain",
            "superseded_v1_support",
            "inventories",
            "prerecovery_topology",
            "failed_v1_recovery_zero_state",
            "documented_preplan_remnants",
            "canonical_outputs_absent",
            "network_requests_during_preflight",
            "raw_event_progress_mutations",
            "labels_read",
            "2024_arrays_read",
            "2025_arrays_read",
        },
        "recovery raw-cache lock schema mismatch",
    )
    remnant_paths = [
        path_under(root, str(record["path"]))
        for record in authorization["documented_preplan_remnants"]
    ]
    expected_topology = {
        "raw_top_level_entries": sorted(
            {"launch_history", "output_transactions", "progress", "ranges", "request_events"}
        ),
        "decoded_tree_absent": True,
        "documented_preplan_transactions": _recovery_file_inventory(remnant_paths, root),
    }
    failed_v1_zero_state = _audit_v1_failed_launch_zero_state_after_v2(root)
    _require(
        int(raw_cache_lock.get("schema_version", -1)) == 2
        and raw_cache_lock.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_RAW_CACHE_LOCK_V2"
        and raw_cache_lock.get("status")
        == "PASS_V2_IMMUTABLE_RAW_CACHE_AND_V1_ZERO_STATE"
        and raw_cache_lock.get("incident") == authorization.get("incident")
        and raw_cache_lock.get("flawed_launch_incident")
        == authorization.get("flawed_launch_incident")
        and raw_cache_lock.get("launch_identity_correction")
        == authorization.get("launch_identity_correction")
        and raw_cache_lock.get("superseded_v1_chain")
        == authorization.get("superseded_v1_chain")
        and raw_cache_lock.get("superseded_v1_support")
        == authorization.get("superseded_v1_support")
        and raw_cache_lock.get("inventories") == raw_prestate["inventories"]
        and raw_cache_lock.get("prerecovery_topology") == expected_topology
        and raw_cache_lock.get("failed_v1_recovery_zero_state")
        == failed_v1_zero_state
        and raw_cache_lock.get("documented_preplan_remnants")
        == authorization.get("documented_preplan_remnants")
        and raw_cache_lock.get("canonical_outputs_absent") is True
        and int(raw_cache_lock.get("network_requests_during_preflight", -1)) == 0
        and int(raw_cache_lock.get("raw_event_progress_mutations", -1)) == 0,
        "recovery raw-cache lock value mismatch",
    )
    require_zero_facts(
        raw_cache_lock,
        {"labels_read": False, "2024_arrays_read": False, "2025_arrays_read": False},
        label="recovery raw-cache lock",
    )
    _require(
        authorization.get("raw_cache_lock") == identity(raw_cache_lock_path, root)
        and authorization.get("real_spawn_preflight") == identity(preflight_path, root),
        "recovery authorization seal identities mismatch",
    )

    snapshot = _reconstruct_recovery_fixed_input_snapshot(
        root,
        authorization,
        go,
        failed_attempt,
        contract,
        external_paths,
    )
    recovery_external_paths = {
        Path(path).resolve()
        for path in provenance["external_identity_paths"]
    }
    recovery_external_paths.update(
        path.resolve()
        for path in (
            *external_paths.values(),
            *sealer_paths.values(),
            *_recovery_runtime_file_paths().values(),
        )
    )
    recovery_external_paths.update(
        (REPO / str(record["path"])).resolve()
        for record in RECOVERY_V1_SUPPORT_IDENTITIES.values()
    )
    original_runtime = provenance["runtime_identity"]
    recovery_external_paths.add(
        Path(str(original_runtime["python_executable"])).resolve()
    )
    for record in original_runtime["packages"].values():
        recovery_external_paths.add(Path(str(record["module_file"])).resolve())
    recovery_external_identity_bytes = sum(
        path.stat().st_size for path in recovery_external_paths
    )
    return {
        "authorization": authorization,
        "authorization_identity": identity(authorization_path, root),
        "go": go,
        "go_identity": identity(go_path, root),
        "review": review,
        "review_identity": identity(review_path, root),
        "incident": incident,
        "incident_identity": identity(incident_path, root),
        "flawed_launch_incident": flawed_launch_incident,
        "flawed_launch_incident_identity": flawed_launch_incident_identity,
        "launch_identity_correction": launch_identity_correction,
        "launch_identity_correction_identity": launch_identity_correction_identity,
        "sealer_entrypoint_incident": entrypoint_incident,
        "sealer_entrypoint_incident_identity": entrypoint_incident_identity,
        "raw_cache_lock": raw_cache_lock,
        "raw_cache_lock_identity": identity(raw_cache_lock_path, root),
        "preflight": preflight,
        "preflight_identity": identity(preflight_path, root),
        "runtime": runtime_validation,
        "current_bootstrap_pyc_gate": current_bootstrap_pyc_gate,
        "current_bootstrap_import_contract": expected_bootstrap_import_contract,
        "external_identities": external_actuals,
        "sealer_identities": sealer_actuals,
        "fixed_input_snapshot": snapshot,
        "raw_prestate": raw_prestate,
        "failed_v1_zero_state": failed_v1_zero_state,
        "expected_prerecovery_topology": expected_topology,
        "external_identity_unique_file_count": len(recovery_external_paths),
        "external_identity_unique_file_bytes": recovery_external_identity_bytes,
    }


def _audit_recovery_canonical_metadata(
    root: Path,
    provenance: Mapping[str, Any],
    transaction: Mapping[str, Any],
    events: Mapping[str, Any],
    decoded: Mapping[str, Any],
    raw_info: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    outputs = transaction["outputs"]
    manifest = transaction["manifest"]
    lock = load_json(outputs["lock"])
    access = load_json(outputs["access"])
    control = _audit_recovery_control_plane(
        root, provenance, transaction, events, raw_info, contract
    )
    authorization = control["authorization"]
    provenance_keys = {
        "decode_failure_incident",
        "flawed_launch_incident",
        "launch_identity_correction",
        "superseded_v1_chain",
        "superseded_v1_support",
        "recovery_authorization_v2",
        "independent_recovery_go_v2",
        "independent_recovery_review_v2",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "bootstrap_trust_policy",
        "bootstrap_import_contract",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "original_frozen_runner",
        "raw_cache_lock_v2",
        "real_spawn_preflight_v2",
        "pilot_runtime_report",
        "failed_v1_recovery_zero_state",
        "fixed_inputs_initial",
        "locked_prestate_topology",
        "locked_fixed_inputs",
    }
    recovery_provenance = manifest.get("recovery_provenance")
    _require(
        isinstance(recovery_provenance, Mapping)
        and set(recovery_provenance) == provenance_keys,
        "recovered output provenance schema mismatch",
    )
    expected_provenance_values = {
        "decode_failure_incident": control["incident_identity"],
        "flawed_launch_incident": control["flawed_launch_incident_identity"],
        "launch_identity_correction": control[
            "launch_identity_correction_identity"
        ],
        "superseded_v1_chain": RECOVERY_V1_CHAIN_IDENTITIES,
        "superseded_v1_support": _authoritative_v1_support_external(),
        "recovery_authorization_v2": control["authorization_identity"],
        "independent_recovery_go_v2": control["go_identity"],
        "independent_recovery_review_v2": control["review_identity"],
        "recovery_runner": authorization["recovery_runner"],
        "recovery_module": authorization["recovery_module"],
        "recovery_bootstrap": authorization["recovery_bootstrap"],
        "bootstrap_trust_policy": RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "bootstrap_import_contract": control[
            "current_bootstrap_import_contract"
        ],
        "recovery_test": authorization["recovery_test"],
        "recovery_bootstrap_test": authorization["recovery_bootstrap_test"],
        "recovery_runner_test": authorization["recovery_runner_test"],
        "postrun_auditor": authorization["postrun_auditor"],
        "postrun_auditor_test": authorization["postrun_auditor_test"],
        "original_frozen_runner": authorization["original_runner"],
        "raw_cache_lock_v2": control["raw_cache_lock_identity"],
        "real_spawn_preflight_v2": control["preflight_identity"],
        "failed_v1_recovery_zero_state": control["failed_v1_zero_state"],
        "fixed_inputs_initial": control["fixed_input_snapshot"],
        "locked_fixed_inputs": control["fixed_input_snapshot"],
    }
    for key, expected in expected_provenance_values.items():
        _require(
            recovery_provenance.get(key) == expected,
            f"recovered output provenance mismatch: {key}",
        )
    _validate_recovery_pilot_report(
        recovery_provenance.get("pilot_runtime_report"),
        label="immediate prerecovery pilot report",
    )
    locked_topology = recovery_provenance.get("locked_prestate_topology")
    _require(
        isinstance(locked_topology, Mapping)
        and set(locked_topology)
        == {
            "raw_top_level_directories",
            "documented_preplan_transactions",
            "raw_mutex",
            "decoded_mutex",
            "mutex_payload",
            "canonical_outputs_absent",
        },
        "locked recovery prestate topology schema mismatch",
    )
    history_root = root / "decoded" / "recovery_history"
    raw_history_lock = (
        history_root
        / f"{transaction['attempt_id']}__raw_mutex__complete.lock"
    )
    decoded_history_lock = (
        history_root
        / f"{transaction['attempt_id']}__decode_mutex__complete.lock"
    )
    mutex_payload = load_json(raw_history_lock)
    expected_locked_mutexes = {
        "raw_mutex": {
            **identity(raw_history_lock, root),
            "path": "raw/RAW_LAUNCH_ACTIVE.lock",
        },
        "decoded_mutex": {
            **identity(decoded_history_lock, root),
            "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
        },
    }
    _require(
        locked_topology.get("raw_top_level_directories")
        == sorted(
            {"launch_history", "output_transactions", "progress", "ranges", "request_events"}
        )
        and locked_topology.get("documented_preplan_transactions")
        == control["expected_prerecovery_topology"][
            "documented_preplan_transactions"
        ]
        and locked_topology.get("raw_mutex") == expected_locked_mutexes["raw_mutex"]
        and locked_topology.get("decoded_mutex")
        == expected_locked_mutexes["decoded_mutex"]
        and locked_topology.get("mutex_payload") == mutex_payload
        and locked_topology.get("canonical_outputs_absent") is True,
        "locked recovery prestate topology value mismatch",
    )
    for payload, label in (
        (lock, "recovered decoded lock"),
        (access, "recovered access ledger"),
        (decoded["physical_payload"], "recovered physical audit"),
    ):
        _require(
            payload.get("recovery_provenance") == recovery_provenance,
            f"{label} recovery provenance mismatch",
        )

    _require(
        set(manifest)
        == {
            "artifact_type",
            "schema_version",
            "created_utc",
            "launch_attempt_id",
            "recovery_provenance",
            "runtime_identity",
            "decoded_matrix_lock",
            "exact_ranges",
            "exact_bytes",
            "decode_processes",
            "decode_threads",
            "network_requests",
            "labels_read",
            "2024_arrays_read",
            "2025_arrays_read",
            "models_fit",
            "submission_csv_created",
        },
        "recovered producer manifest schema mismatch",
    )
    _require(
        manifest.get("artifact_type")
        == "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST_RECOVERED_V2"
        and int(manifest.get("schema_version", -1)) == 2
        and manifest.get("launch_attempt_id") == transaction["attempt_id"]
        and int(manifest.get("exact_ranges", -1)) == contract.range_rows
        and int(manifest.get("exact_bytes", -1)) == contract.range_bytes
        and int(manifest.get("decode_processes", -1)) == DECODE_MAX_WORKERS
        and int(manifest.get("decode_threads", -1)) == 0
        and int(manifest.get("network_requests", -1)) == 0
        and manifest.get("runtime_identity")
        == control["runtime"]["runtime_report"],
        "recovered producer manifest values mismatch",
    )
    require_zero_facts(
        manifest,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="recovered producer manifest",
    )
    verify_identity(
        manifest.get("decoded_matrix_lock", {}),
        outputs["lock"],
        root=root,
        label="recovered manifest decoded lock",
    )

    _require(
        set(lock)
        == {
            "artifact_type",
            "schema_version",
            "recovery_provenance",
            "runtime_identity",
            "raw_manifest_parquet",
            "raw_manifest_csv",
            "site_matrix",
            "group_matrix",
            "physical_and_coverage_audit",
            "access_ledger",
            "physical_family_ready_for_independent_postrun",
            "independent_postrun_audit_required_before_target_free_estimator",
            "target_free_duplicate_estimator_may_start",
            "network_requests",
            "labels_read",
        },
        "recovered decoded matrix lock schema mismatch",
    )
    _require(
        lock.get("artifact_type")
        == "TARGET_FREE_DECODED_MATRIX_LOCK_RECOVERED_NETWORK_ZERO_V2"
        and int(lock.get("schema_version", -1)) == 2
        and lock.get("runtime_identity") == control["runtime"]["runtime_report"]
        and lock.get("physical_family_ready_for_independent_postrun")
        is decoded["downstream_target_free_estimator_may_start"]
        and lock.get(
            "independent_postrun_audit_required_before_target_free_estimator"
        ) is True
        and lock.get("target_free_duplicate_estimator_may_start") is False
        and int(lock.get("network_requests", -1)) == 0,
        "recovered decoded matrix lock value mismatch",
    )
    require_zero_facts(lock, {"labels_read": False}, label="recovered decoded lock")
    for field, output_name in {
        "raw_manifest_parquet": "raw_parquet",
        "raw_manifest_csv": "raw_csv",
        "site_matrix": "site",
        "group_matrix": "group",
        "physical_and_coverage_audit": "audit",
        "access_ledger": "access",
    }.items():
        verify_identity(
            lock.get(field, {}),
            outputs[output_name],
            root=root,
            label=f"recovered decoded lock {field}",
        )

    _require(
        set(access)
        == {
            "artifact_type",
            "expected_ranges",
            "expected_bytes",
            "network_requests_this_invocation",
            "network_bytes_this_invocation",
            "raw_cache_reuses",
            "decode_processes",
            "decode_threads",
            "before_inventories",
            "after_inventories",
            "inventories_equal",
            "raw_event_progress_mutations",
            "documented_preplan_remnants_preserved",
            "labels_read",
            "2024_arrays_read",
            "2025_arrays_read",
            "models_fit",
            "submission_csv_created",
            "recovery_provenance",
        },
        "recovered raw access ledger schema mismatch",
    )
    expected_inventories = control["raw_prestate"]["inventories"]
    _require(
        access.get("artifact_type")
        == "TRACK_A_RAW_ACCESS_LEDGER_RECOVERED_NETWORK_ZERO_V2"
        and int(access.get("expected_ranges", -1)) == contract.range_rows
        and int(access.get("expected_bytes", -1)) == contract.range_bytes
        and int(access.get("network_requests_this_invocation", -1)) == 0
        and int(access.get("network_bytes_this_invocation", -1)) == 0
        and int(access.get("raw_cache_reuses", -1)) == contract.range_rows
        and int(access.get("decode_processes", -1)) == DECODE_MAX_WORKERS
        and int(access.get("decode_threads", -1)) == 0
        and access.get("before_inventories") == expected_inventories
        and access.get("after_inventories") == expected_inventories
        and access.get("inventories_equal") is True
        and int(access.get("raw_event_progress_mutations", -1)) == 0
        and access.get("documented_preplan_remnants_preserved")
        == authorization["documented_preplan_remnants"],
        "recovered raw access ledger accounting mismatch",
    )
    require_zero_facts(
        access,
        {
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        label="recovered raw access ledger",
    )

    postcommit = transaction["recovery_postcommit_audit"]
    _require(
        postcommit.get("initial") == control["fixed_input_snapshot"]
        and postcommit.get("preplan") == control["fixed_input_snapshot"]
        and postcommit.get("postcommit") == control["fixed_input_snapshot"],
        "recovery postcommit snapshots differ from independent rehash",
    )
    _require(
        transaction["plan"].get("recovery_authorization")
        == control["authorization_identity"]
        and transaction["plan"].get("independent_recovery_go")
        == control["go_identity"]
        and transaction["plan"].get("decode_failure_incident")
        == control["incident_identity"]
        and transaction["plan"].get("flawed_launch_incident")
        == control["flawed_launch_incident_identity"]
        and transaction["plan"].get("launch_identity_correction")
        == control["launch_identity_correction_identity"]
        and transaction["plan"].get("superseded_v1_chain")
        == RECOVERY_V1_CHAIN_IDENTITIES
        and transaction["plan"].get("superseded_v1_support")
        == _authoritative_v1_support_external(),
        "recovery transaction control-plane binding mismatch",
    )
    final_progress = transaction["recovery_progress_paths"][-1]
    final_checkpoint = load_json(final_progress)
    _require(
        int(final_checkpoint.get("completed_messages", -1)) == contract.range_rows,
        "recovery final decode progress count mismatch",
    )
    return {
        "recovery_mode": True,
        "network_ranges_this_final_invocation": 0,
        "network_bytes_this_final_invocation": 0,
        "cached_ranges_this_final_invocation": contract.range_rows,
        "decode_processes": DECODE_MAX_WORKERS,
        "decode_threads": 0,
        "recovery_progress_checkpoint_count": len(
            transaction["recovery_progress_paths"]
        ),
        "recovery_history_file_count": len(transaction["recovery_history_paths"]),
        "incident_bound_preplan_remnant_count": transaction[
            "incident_bound_preplan_remnant_count"
        ],
        "original_raw_progress_checkpoint_count": control["raw_prestate"][
            "original_progress_checkpoint_count"
        ],
        "recovery_authorization": control["authorization_identity"],
        "independent_recovery_go": control["go_identity"],
        "independent_recovery_review": control["review_identity"],
        "recovery_incident": control["incident_identity"],
        "preserved_flawed_v1_launch_incident": control[
            "flawed_launch_incident_identity"
        ],
        "authoritative_v1_launch_identity_correction": control[
            "launch_identity_correction_identity"
        ],
        "failed_v1_recovery_zero_state": control["failed_v1_zero_state"],
        "recovery_sealer_entrypoint_incident": control[
            "sealer_entrypoint_incident_identity"
        ],
        "recovery_raw_cache_lock": control["raw_cache_lock_identity"],
        "recovery_real_process_preflight": control["preflight_identity"],
        "recovery_current_bootstrap_pyc_gate": control[
            "current_bootstrap_pyc_gate"
        ],
        "recovery_current_bootstrap_import_contract": control[
            "current_bootstrap_import_contract"
        ],
        "recovery_fixed_input_snapshot": control["fixed_input_snapshot"],
        "external_identity_unique_file_count": control[
            "external_identity_unique_file_count"
        ],
        "external_identity_unique_file_bytes": control[
            "external_identity_unique_file_bytes"
        ],
        "raw_http_attempts_cumulative": events["starts"],
        "raw_http_attempts_without_terminal_event": events["indeterminate_starts"],
        "labels_or_2024_2025_arrays_or_models_or_submission_csv": 0,
    }


def _audit_canonical_metadata(
    root: Path,
    provenance: Mapping[str, Any],
    transaction: Mapping[str, Any],
    events: Mapping[str, Any],
    decoded: Mapping[str, Any],
    raw_info: Mapping[tuple[str, str, str], Mapping[str, Any]],
    contract: AuditContract,
) -> dict[str, Any]:
    if transaction.get("recovered") is True:
        return _audit_recovery_canonical_metadata(
            root,
            provenance,
            transaction,
            events,
            decoded,
            raw_info,
            contract,
        )
    outputs = transaction["outputs"]
    manifest = transaction["manifest"]
    lock = load_json(outputs["lock"])
    access = load_json(outputs["access"])
    require_no_unknown_keys(
        manifest,
        (
            "artifact_type", "schema_version", "created_utc", "runner",
            "launch_attempt_id", "authorization", "independent_go",
            "census_manifest", "coordinate_lock", "runtime_identity",
            "runtime_identity_sha256", "decoded_matrix_lock",
            "progress_checkpoints", "request_event_inventory", "exact_ranges",
            "exact_bytes", "labels_read", "2024_arrays_read", "2025_arrays_read",
            "models_fit", "submission_csv_created",
        ),
        label="producer manifest",
    )
    _require(len(manifest) == 21, "producer manifest required-field schema mismatch")
    _require(
        manifest.get("artifact_type") == "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST"
        and int(manifest.get("schema_version", -1)) == 1,
        "producer manifest header mismatch",
    )
    require_zero_facts(
        manifest,
        {"labels_read": False, "2024_arrays_read": False, "2025_arrays_read": False, "models_fit": 0, "submission_csv_created": False},
        label="producer manifest",
    )
    _require(int(manifest.get("exact_ranges", -1)) == contract.range_rows, "producer manifest range count mismatch")
    _require(int(manifest.get("exact_bytes", -1)) == contract.range_bytes, "producer manifest byte count mismatch")
    verify_identity(manifest.get("runner", {}), RUNNER_DEFAULT if provenance.get("runner_path") is None else Path(provenance["runner_path"]), root=None, label="producer-manifest runner")
    verify_identity(manifest.get("authorization", {}), root / "prereg" / "raw_launch_authorization_v1.json", root=root, label="producer-manifest authorization")
    verify_identity(manifest.get("independent_go", {}), root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json", root=root, label="producer-manifest GO")
    verify_identity(manifest.get("census_manifest", {}), root / "manifest_census_v1.json", root=root, label="producer-manifest census")
    verify_identity(manifest.get("coordinate_lock", {}), root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json", root=root, label="producer-manifest coordinates")
    verify_identity(manifest.get("decoded_matrix_lock", {}), outputs["lock"], root=root, label="producer-manifest decoded lock")
    _require(manifest.get("runtime_identity") == provenance["runtime_identity"], "producer manifest runtime identity mismatch")
    _require(manifest.get("runtime_identity_sha256") == provenance["runtime_identity_sha256"], "producer manifest runtime digest mismatch")

    event_paths = [path_under(root, record["path"]) for record in events["event_identities"]]
    _verify_inventory(manifest.get("request_event_inventory"), event_paths, root, label="producer request-event inventory")
    progress_root = root / "raw" / "progress"
    _require(progress_root.is_dir(), "raw progress directory is absent")
    progress_entries = list(progress_root.rglob("*"))
    progress_symlinks = [
        path.relative_to(root).as_posix()
        for path in progress_entries
        if _linklike(path)
    ]
    _require(
        not progress_symlinks,
        f"symlink forbidden in progress inventory: {progress_symlinks[:5]}",
    )
    progress_directories = [
        path.relative_to(root).as_posix()
        for path in progress_entries
        if path.is_dir()
    ]
    _require(
        not progress_directories,
        f"unexpected directory in progress inventory: {progress_directories[:5]}",
    )
    _require(
        all(path.is_file() for path in progress_entries),
        "progress inventory contains a special filesystem object",
    )
    progress_paths = sorted(path for path in progress_entries if path.is_file())
    for progress_path in progress_paths:
        match = PROGRESS_NAME.fullmatch(progress_path.name)
        _require(match is not None, f"unexpected progress filename: {progress_path.name}")
        assert match is not None
        kind = match.group("kind")
        checkpoint_attempt = match.group("attempt")
        checkpoint_count = int(match.group("count"))
        checkpoint = load_json(progress_path)
        expected_phase = (
            "RAW_RANGE_DOWNLOAD"
            if kind == "raw_ranges"
            else "ECCODES_BILINEAR_DECODE"
        )
        expected_count_key = (
            "completed_ranges" if kind == "raw_ranges" else "completed_messages"
        )
        expected_checkpoint_fields = (
            {
                "phase", "attempt_id", "completed_ranges",
                "completed_key_set_sha256",
                "network_requests_this_invocation_so_far",
                "network_bytes_this_invocation_so_far",
                "actual_http_attempts_cumulative",
            }
            if kind == "raw_ranges"
            else {"phase", "attempt_id", "completed_messages", "network_requests"}
        )
        _require(
            set(checkpoint) == expected_checkpoint_fields,
            f"progress payload schema mismatch: {progress_path.name}",
        )
        _require(
            checkpoint.get("attempt_id") == checkpoint_attempt
            and checkpoint.get("phase") == expected_phase
            and int(checkpoint.get(expected_count_key, -1)) == checkpoint_count
            and 0 < checkpoint_count <= contract.range_rows,
            f"progress filename/payload mismatch: {progress_path.name}",
        )
        if kind == "decoded_messages":
            _require(
                int(checkpoint.get("network_requests", -1)) == 0,
                f"decode progress records network access: {progress_path.name}",
            )
    _verify_inventory(manifest.get("progress_checkpoints"), progress_paths, root, label="producer progress inventory")
    attempt_id = transaction["attempt_id"]
    final_raw_checkpoint = root / "raw" / "progress" / f"raw_ranges__{attempt_id}__{contract.range_rows:06d}.json"
    final_decode_checkpoint = root / "raw" / "progress" / f"decoded_messages__{attempt_id}__{contract.range_rows:06d}.json"
    for path, phase, count_key in (
        (final_raw_checkpoint, "RAW_RANGE_DOWNLOAD", "completed_ranges"),
        (final_decode_checkpoint, "ECCODES_BILINEAR_DECODE", "completed_messages"),
    ):
        checkpoint = load_json(path)
        _require(checkpoint.get("attempt_id") == attempt_id and checkpoint.get("phase") == phase, f"final progress checkpoint identity mismatch: {phase}")
        _require(int(checkpoint.get(count_key, -1)) == contract.range_rows, f"final progress checkpoint count mismatch: {phase}")
        if phase == "RAW_RANGE_DOWNLOAD":
            completed_keys = [
                "|".join(
                    (
                        str(info["row"]["object_key"]),
                        str(int(info["row"]["forecast_hour"])),
                        str(info["row"]["variable"]),
                        str(info["row"]["level"]),
                    )
                )
                for info in raw_info.values()
            ]
            expected_key_digest = hashlib.sha256(
                ("\n".join(sorted(completed_keys)) + "\n").encode("utf-8")
            ).hexdigest()
            _require(
                checkpoint.get("completed_key_set_sha256") == expected_key_digest,
                "final raw progress key-set digest mismatch",
            )

    _require(
        lock.get("artifact_type") == "TARGET_FREE_DECODED_MATRIX_LOCK",
        "decoded matrix lock artifact type mismatch",
    )
    _require(
        set(lock)
        == {
            "artifact_type", "authorization", "independent_go", "coordinate_lock",
            "runtime_identity", "runtime_identity_sha256", "raw_manifest_parquet",
            "raw_manifest_csv", "site_matrix", "group_matrix",
            "physical_and_coverage_audit", "access_ledger",
            "target_free_duplicate_estimator_may_start", "labels_read",
        },
        "decoded matrix lock payload schema mismatch",
    )
    require_zero_facts(lock, {"labels_read": False}, label="decoded matrix lock")
    verify_identity(lock.get("authorization", {}), root / "prereg" / "raw_launch_authorization_v1.json", root=root, label="decoded-lock authorization")
    verify_identity(lock.get("independent_go", {}), root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json", root=root, label="decoded-lock GO")
    verify_identity(lock.get("coordinate_lock", {}), root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json", root=root, label="decoded-lock coordinates")
    for field, output_name in {
        "raw_manifest_parquet": "raw_parquet", "raw_manifest_csv": "raw_csv",
        "site_matrix": "site", "group_matrix": "group",
        "physical_and_coverage_audit": "audit", "access_ledger": "access",
    }.items():
        verify_identity(lock.get(field, {}), outputs[output_name], root=root, label=f"decoded-lock {field}")
    _require(lock.get("runtime_identity") == provenance["runtime_identity"], "decoded lock runtime identity mismatch")
    _require(lock.get("runtime_identity_sha256") == provenance["runtime_identity_sha256"], "decoded lock runtime digest mismatch")
    _require(lock.get("target_free_duplicate_estimator_may_start") is decoded["downstream_target_free_estimator_may_start"], "decoded lock downstream gate mismatch")

    require_zero_facts(
        access,
        {"labels_read": False, "2024_arrays_read": False, "2025_arrays_read": False, "models_fit": 0, "submission_csv_created": False},
        label="raw access ledger",
    )
    _require(
        set(access)
        == {
            "expected_ranges", "expected_bytes", "network_requests_this_invocation",
            "network_bytes_this_invocation", "actual_http_attempts_before_invocation",
            "actual_http_attempts_after_invocation", "actual_http_attempts_this_invocation",
            "raw_actual_http_attempt_budget", "census_worst_case_http_attempts",
            "census_plus_raw_max_http_attempts",
            "census_worst_case_plus_raw_budget_le_20000",
            "census_successful_http_requests",
            "census_plus_raw_actual_attempts_or_successes", "cached_reuses",
            "durable_transport_attempt_starts", "durable_transport_attempt_completions",
            "durable_transport_attempt_errors", "free_disk_before_bytes",
            "free_disk_after_bytes", "final_free_disk_before_transaction_bytes",
            "final_200gb_reserve_pass", "concurrency", "labels_read",
            "2024_arrays_read", "2025_arrays_read", "models_fit",
            "submission_csv_created",
        },
        "raw access ledger payload schema mismatch",
    )
    for key, expected in {
        "expected_ranges": contract.range_rows,
        "expected_bytes": contract.range_bytes,
        "raw_actual_http_attempt_budget": contract.raw_attempt_cap,
        "census_worst_case_http_attempts": contract.census_worst_case_attempts,
        "census_plus_raw_max_http_attempts": contract.combined_attempt_cap,
        "census_successful_http_requests": contract.census_successful_requests,
        "durable_transport_attempt_starts": events["starts"],
        "durable_transport_attempt_completions": events["completions"],
        "durable_transport_attempt_errors": events["errors"],
        "actual_http_attempts_after_invocation": events["starts"],
    }.items():
        _require(int(access.get(key, -1)) == expected, f"raw access ledger {key} mismatch")
    before = int(access.get("actual_http_attempts_before_invocation", -1))
    this_run = int(access.get("actual_http_attempts_this_invocation", -1))
    _require(before >= 0 and this_run >= 0 and before + this_run == events["starts"], "raw access invocation attempt arithmetic mismatch")
    invocation_attempts = [
        record
        for record in events["attempt_inventory"]
        if int(record["global_raw_attempt_number"]) > before
    ]
    invocation_completions = sum(
        record["outcome"] == "COMPLETE" for record in invocation_attempts
    )
    invocation_errors = sum(
        record["outcome"] == "ERROR" for record in invocation_attempts
    )
    invocation_start_only = sum(
        record["outcome"] == "START_ONLY" for record in invocation_attempts
    )
    invocation_complete_bytes = sum(
        int(record["complete_payload_bytes"])
        for record in invocation_attempts
        if record["outcome"] == "COMPLETE"
    )
    _require(
        len(invocation_attempts) == this_run
        and invocation_completions + invocation_errors + invocation_start_only
        == this_run,
        "raw access invocation reconstruction mismatch",
    )
    _require(
        invocation_start_only == 0,
        "successful transaction-producing invocation has START-only attempt",
    )
    invocation_completed_keys = [
        tuple(str(value) for value in record["raw_key"])
        for record in invocation_attempts
        if record["outcome"] == "COMPLETE"
    ]
    _require(
        len(set(invocation_completed_keys)) == invocation_completions,
        "final invocation COMPLETE events do not map to unique raw ranges",
    )
    network_ranges = int(access.get("network_requests_this_invocation", -1))
    cached = int(access.get("cached_reuses", -1))
    network_bytes = int(access.get("network_bytes_this_invocation", -1))
    _require(
        network_ranges == invocation_completions
        and 0 <= network_ranges <= contract.range_rows
        and cached == contract.range_rows - invocation_completions,
        "raw access cache/network range reconstruction mismatch",
    )
    _require(
        network_bytes == invocation_complete_bytes
        and 0 <= network_bytes <= contract.range_bytes,
        "raw access invocation payload-byte reconstruction mismatch",
    )
    raw_checkpoint = load_json(final_raw_checkpoint)
    _require(
        int(raw_checkpoint.get("network_requests_this_invocation_so_far", -1))
        == network_ranges
        and int(raw_checkpoint.get("network_bytes_this_invocation_so_far", -1))
        == network_bytes
        and int(raw_checkpoint.get("actual_http_attempts_cumulative", -1))
        == events["starts"],
        "final raw progress/access/event accounting mismatch",
    )
    _require(access.get("census_worst_case_plus_raw_budget_le_20000") is True, "raw access combined budget gate mismatch")
    _require(
        int(access.get("census_plus_raw_actual_attempts_or_successes", -1))
        == contract.census_successful_requests + events["starts"],
        "raw access census-plus-raw actual arithmetic mismatch",
    )
    _require(access.get("final_200gb_reserve_pass") is True, "raw access final disk gate mismatch")
    _require(int(access.get("final_free_disk_before_transaction_bytes", -1)) >= 200_000_000_000, "raw access recorded final disk reserve below 200 GB")
    _require(
        int(access.get("concurrency", -1)) == 8,
        "raw producer concurrency record mismatch",
    )
    _require(
        int(access.get("free_disk_before_bytes", -1)) - contract.range_bytes
        >= 200_000_000_000,
        "raw access start-projection disk reserve mismatch",
    )
    _require(
        int(access.get("free_disk_after_bytes", -1))
        == int(access.get("final_free_disk_before_transaction_bytes", -2)),
        "raw access final free-space records disagree",
    )
    return {
        "network_ranges_this_final_invocation": network_ranges,
        "network_bytes_this_final_invocation": network_bytes,
        "cached_ranges_this_final_invocation": cached,
        "reconstructed_attempt_starts_this_final_invocation": len(
            invocation_attempts
        ),
        "reconstructed_completions_this_final_invocation": invocation_completions,
        "reconstructed_errors_this_final_invocation": invocation_errors,
        "reconstructed_start_only_this_final_invocation": invocation_start_only,
        "reconstructed_complete_payload_bytes_this_final_invocation": invocation_complete_bytes,
        "raw_http_attempts_cumulative": events["starts"],
        "raw_http_attempts_without_terminal_event": events["indeterminate_starts"],
        "raw_http_attempt_cap": contract.raw_attempt_cap,
        "labels_or_2024_2025_arrays_or_models_or_submission_csv": 0,
    }


def _v4_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _v4_created_utc(value: Any, *, label: str) -> datetime:
    _require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z", value)
        is not None,
        f"{label} created_utc is not exact UTC RFC3339 microseconds",
    )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise AuditFailure(f"{label} created_utc is invalid") from exc
    return parsed


def _v4_require_exact_declared_identity(
    record: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    _require(
        isinstance(record, Mapping) and dict(record) == dict(expected),
        f"{label} declared identity mismatch",
    )


def _v4_verify_exact_root_identity(
    root: Path, record: Any, expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    _v4_require_exact_declared_identity(record, expected, label=label)
    return verify_identity(
        record,
        path_under(root, str(expected["path"])),
        root=root,
        label=label,
    )


def _v4_verify_external_identity(
    record: Any,
    expected_path: Path,
    *,
    label: str,
    expected_size_sha256: tuple[int, str] | None = None,
) -> dict[str, Any]:
    actual = verify_identity(
        record, expected_path, root=None, label=label
    )
    if expected_size_sha256 is not None:
        expected_size, expected_sha256 = expected_size_sha256
        _require(
            actual["size_bytes"] == expected_size
            and actual["sha256"] == expected_sha256,
            f"{label} frozen identity mismatch",
        )
    return actual


def _v4_required_command(root: Path, authorization_path: Path, go_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
        "--root",
        str(root.resolve()),
        "--authorization",
        str(authorization_path.resolve()),
        "--independent-go",
        str(go_path.resolve()),
    ]


def _v4_require_canonical_runtime_entrypoint(
    root: Path, authorization_path: Path, go_path: Path
) -> None:
    _require(
        __spec__ is not None
        and __spec__.name == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4"
        and __package__ == "scripts",
        "V4 production audit requires the canonical -m module entrypoint",
    )
    _require(
        sys.dont_write_bytecode is True
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None,
        "V4 production audit bytecode/cache environment mismatch",
    )
    expected_argv = [
        str(Path(__file__).resolve()),
        "--root",
        str(root.resolve()),
        "--authorization",
        str(authorization_path.resolve()),
        "--independent-go",
        str(go_path.resolve()),
    ]
    actual_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    _require(actual_argv == expected_argv, "V4 production argv is not exact canonical order")
    actual_orig_argv = [
        str(Path(sys.orig_argv[0]).resolve()), *sys.orig_argv[1:]
    ]
    _require(
        actual_orig_argv
        == _v4_required_command(root, authorization_path, go_path),
        "V4 production interpreter/module argv is not exact authorized command",
    )


def _v4_test_command(relative: str) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-c",
        V4_PYTEST_NETWORK_GUARD_SOURCE,
        "-q",
        "-p",
        "no:cacheprovider",
        relative,
    ]


def _v4_control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in (
        FROZEN_V3_AUTH_RELATIVE,
        FROZEN_V3_REVIEW_RELATIVE,
        FROZEN_V3_CANONICAL_GO_RELATIVE,
        FROZEN_V3_MISPLACED_GO_RELATIVE,
        V4_AUTH_RELATIVE,
        V4_REVIEW_RELATIVE,
        V4_GO_RELATIVE,
        V2_FALSE_REJECT_INCIDENT_RELATIVE,
        V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
    ):
        destination = path_under(root, relative)
        _require(
            destination.parent.is_dir() and not _linklike(destination.parent),
            f"V4 control parent is absent or link-like: {relative}",
        )
        folded = destination.name.casefold()
        for candidate in destination.parent.iterdir():
            if candidate.name == destination.name:
                continue
            name = candidate.name.casefold()
            if (
                name.startswith(f".{folded}.")
                or name.startswith(f"{folded}.")
                or name == f"{folded}.tmp"
            ):
                matches.append(candidate.relative_to(root).as_posix())
    return sorted(set(matches))


def _v4_postrun_report_candidates(root: Path) -> list[str]:
    """Inventory every forbidden persisted V2/V3/V4 stdout report candidate."""

    present: list[str] = []
    for relative in V4_POSTRUN_REPORT_CANDIDATES:
        path = path_under(root, relative)
        if not os.path.lexists(str(path)):
            continue
        _require(
            path.is_file() and not _linklike(path),
            f"V4 postrun report candidate is non-regular/link-like: {relative}",
        )
        present.append(relative)
    return present


def _v4_validate_false_reject_incident(root: Path, record: Any) -> dict[str, Any]:
    expected = {
        "path": V2_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": V2_FALSE_REJECT_INCIDENT_SIZE_BYTES,
        "sha256": V2_FALSE_REJECT_INCIDENT_SHA256,
    }
    incident_path = path_under(root, V2_FALSE_REJECT_INCIDENT_RELATIVE)
    _v4_verify_exact_root_identity(root, record, expected, label="V2 false-reject incident")
    incident = load_json(incident_path)
    _require(
        set(incident)
        == {
            "schema_version", "artifact_type", "status", "created_utc",
            "authority_scope", "recovery_attempt", "observation_time_bounds",
            "failed_auditor", "failed_auditor_test", "frozen_support",
            "immutable_controls", "failed_invocations", "failure_boundary",
            "root_cause", "postfailure_state", "required_v3_supersession",
            "prohibitions",
        },
        "V4 false-reject incident schema mismatch",
    )
    _require(
        incident.get("schema_version") == 1
        and incident.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V2_IDENTITY_SCHEMA_FALSE_REJECT_INCIDENT"
        and incident.get("status")
        == "SEALED_FALSE_REJECT_NO_POSTRUN_PASS_V3_SUPERSESSION_REQUIRED",
        "V4 false-reject incident header mismatch",
    )
    _require(
        incident.get("failed_auditor")
        == {
            "path": "scripts/audit_noaa_gfs_multiseason_raw_postrun_v2.py",
            "size_bytes": 300_379,
            "sha256": V4_SUPERSEDED_V2_AUDITOR_IDENTITY["sha256"],
        }
        and incident.get("failed_auditor_test")
        == {
            "path": "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py",
            "size_bytes": 133_607,
            "sha256": V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY["sha256"],
        },
        "V4 incident frozen V2 auditor/test identities mismatch",
    )
    frozen_support = incident.get("frozen_support")
    _require(
        isinstance(frozen_support, Mapping)
        and set(frozen_support)
        == {"recovery_runner", "recovery_runner_test", "recovery_sealer", "recovery_sealer_test"},
        "V4 incident frozen support schema mismatch",
    )
    for role in frozen_support:
        expected_path = V4_V2_SUPPORT_PATHS[role].relative_to(REPO).as_posix()
        expected_size, expected_sha = V4_V2_SUPPORT_SIZE_SHA256[role]
        _require(
            frozen_support[role]
            == {"path": expected_path, "size_bytes": expected_size, "sha256": expected_sha},
            f"V4 incident frozen support mismatch: {role}",
        )
    recovery_attempt = incident.get("recovery_attempt")
    root_cause = incident.get("root_cause")
    required = incident.get("required_v3_supersession")
    authority_scope = incident.get("authority_scope")
    prohibitions = incident.get("prohibitions")
    _require(
        isinstance(recovery_attempt, Mapping)
        and recovery_attempt.get("attempt_id") == V4_RECOVERY_ATTEMPT_ID
        and recovery_attempt.get("runner_reported_exit_code") == 0
        and recovery_attempt.get("runner_reported_network_requests") == 0,
        "V4 incident recovery attempt mismatch",
    )
    _require(
        isinstance(root_cause, Mapping)
        and root_cause.get("category") == "EXACT_IDENTITY_SCHEMA_FALSE_REJECT"
        and root_cause.get("immutable_authorization_field")
        == "independent_prelaunch_audit"
        and root_cause.get("immutable_authorization_record")
        == V4_ORIGINAL_V1_PROVENANCE["original_independent_prelaunch_audit_v1"]
        and root_cause.get("immutable_record_key_set")
        == ["path", "sha256", "size_bytes", "status"]
        and root_cause.get("frozen_auditor_call_omitted_allowed_extra_fields_status")
        is True,
        "V4 incident root-cause mismatch",
    )
    _require(
        isinstance(required, Mapping)
        and required.get("status") == "REQUIRED_NOT_YET_AUTHORIZED_FOR_EXECUTION"
        and required.get("new_paths_only") is True
        and required.get("v2_auditor_and_test_preserved_as_failed_history") is True
        and required.get("recovery_rerun_forbidden") is True
        and required.get("minimal_code_delta")
        == {
            "exact_call_site": "_audit_provenance independent_prelaunch_audit identity verification",
            "require_allowed_extra_fields": ["status"],
            "require_exact_status_value": "PASS_TO_CREATE_FINAL_AUTH_ONLY",
            "all_other_production_audit_logic_unchanged": True,
        }
        and required.get("required_append_only_controls")
        == {
            "authorization": FROZEN_V3_AUTH_RELATIVE,
            "independent_review": FROZEN_V3_REVIEW_RELATIVE,
            "independent_go": FROZEN_V3_CANONICAL_GO_RELATIVE,
        },
        "V4 incident supersession contract mismatch",
    )
    _require(
        isinstance(authority_scope, Mapping)
        and authority_scope.get("v3_postrun_audit_execution_authorized_by_this_incident")
        is False
        and authority_scope.get("recovery_retry_authorized") is False
        and authority_scope.get("model_or_label_or_submission_work_authorized") is False
        and isinstance(prohibitions, Mapping)
        and prohibitions.get("launch_v3_auditor_before_exact_code_test_freeze_and_independent_go")
        is False
        and prohibitions.get("use_network") is False,
        "V4 incident authority boundary mismatch",
    )
    _require(
        canonical_payload_sha256(incident.get("postfailure_state"))
        == V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
        "V4 incident postfailure-state digest mismatch",
    )
    return incident


def _v4_namespace_expected() -> dict[str, set[str]]:
    return {
        "prereg": {
            Path(FROZEN_V3_AUTH_RELATIVE).name,
            Path(FROZEN_V3_MISPLACED_GO_RELATIVE).name,
            Path(V4_AUTH_RELATIVE).name,
        },
        "independent_redteam": {
            Path(FROZEN_V3_REVIEW_RELATIVE).name,
            Path(V4_REVIEW_RELATIVE).name,
            Path(V4_GO_RELATIVE).name,
        },
        "incidents": {
            Path(V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
        },
    }


def _v4_validate_control_namespace(root: Path) -> dict[str, list[str]]:
    """Close every postrun-auditor control namespace, including aliases/temps."""

    prefixes = {
        "prereg": "decode_recovery_postrun_audit_",
        "independent_redteam": "track_a_decode_recovery_postrun_auditor_",
        "incidents": "track_a_decode_recovery_postrun_auditor_",
    }
    expected = _v4_namespace_expected()
    result: dict[str, list[str]] = {}
    for relative, prefix in prefixes.items():
        parent = path_under(root, relative)
        _require(
            os.path.lexists(str(parent))
            and parent.is_dir()
            and not _linklike(parent),
            f"V4 control namespace parent invalid: {relative}",
        )
        matched: list[str] = []
        for entry in parent.iterdir():
            if not entry.name.casefold().lstrip(".").startswith(prefix):
                continue
            _require(
                os.path.lexists(str(entry))
                and entry.is_file()
                and not _linklike(entry),
                f"V4 control namespace contains non-regular/link entry: {entry}",
            )
            matched.append(entry.name)
        _require(
            set(matched) == expected[relative]
            and len({name.casefold() for name in matched}) == len(matched),
            f"V4 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched, key=str.casefold)
    canonical_v3_go = path_under(root, FROZEN_V3_CANONICAL_GO_RELATIVE)
    _require(
        not os.path.lexists(str(canonical_v3_go)),
        "canonical V3 GO must remain absent permanently",
    )
    return result


def _v4_validate_path_mispublish_incident(
    root: Path, record: Any
) -> dict[str, Any]:
    expected = {
        "path": V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        "size_bytes": V4_PATH_MISPUBLISH_INCIDENT_SIZE_BYTES,
        "sha256": V4_PATH_MISPUBLISH_INCIDENT_SHA256,
    }
    incident_path = path_under(root, V4_PATH_MISPUBLISH_INCIDENT_RELATIVE)
    _v4_verify_exact_root_identity(
        root, record, expected, label="V3 GO path-mispublication incident"
    )
    incident = load_json(incident_path)
    _require(
        set(incident)
        == {
            "artifact_type", "authority_scope", "created_utc", "failure_boundary",
            "immutable_v3_chain", "mispublication", "observation_time_bounds",
            "prohibitions", "required_v4_supersession", "root_cause",
            "schema_version", "status",
        },
        "V3 path-mispublication incident schema mismatch",
    )
    _require(
        incident.get("schema_version") == 1
        and incident.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V3_GO_PATH_MISPUBLISH_INCIDENT_V1"
        and incident.get("status")
        == "SEALED_V3_GO_PATH_MISPUBLISH_NO_LIVE_AUDIT_V4_SUPERSESSION_REQUIRED",
        "V3 path-mispublication incident header mismatch",
    )
    chain = incident.get("immutable_v3_chain")
    _require(
        isinstance(chain, Mapping)
        and chain.get("authorization") == FROZEN_V3_AUTH_IDENTITY
        and chain.get("independent_review") == FROZEN_V3_REVIEW_IDENTITY
        and chain.get("false_reject_incident")
        == {
            "path": V2_FALSE_REJECT_INCIDENT_RELATIVE,
            "size_bytes": V2_FALSE_REJECT_INCIDENT_SIZE_BYTES,
            "sha256": V2_FALSE_REJECT_INCIDENT_SHA256,
        }
        and chain.get("v3_auditor")
        == FROZEN_V3_SOURCE_IDENTITIES["superseded_v3_auditor"]
        and chain.get("v3_auditor_test")
        == FROZEN_V3_SOURCE_IDENTITIES["superseded_v3_auditor_test"]
        and chain.get("v3_sealer")
        == FROZEN_V3_SOURCE_IDENTITIES["superseded_v3_sealer"]
        and chain.get("v3_sealer_test")
        == FROZEN_V3_SOURCE_IDENTITIES["superseded_v3_sealer_test"]
        and chain.get("authorization_status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V3"
        and chain.get("review_status") == "PASS_PENDING_INDEPENDENT_GO_V3"
        and chain.get("review_verdict") == "GO_RECOMMENDED"
        and chain.get("all_listed_files_preserved_unchanged") is True,
        "V3 path-mispublication incident immutable-chain mismatch",
    )
    mispublication = incident.get("mispublication")
    _require(
        isinstance(mispublication, Mapping)
        and mispublication.get("actual_noncanonical_file")
        == FROZEN_V3_MISPLACED_GO_IDENTITY
        and mispublication.get("actual_noncanonical_relative_path")
        == FROZEN_V3_MISPLACED_GO_RELATIVE
        and mispublication.get("intended_canonical_relative_path")
        == FROZEN_V3_CANONICAL_GO_RELATIVE
        and mispublication.get("intended_canonical_path_present") is False
        and mispublication.get("payload_canonical_sha256")
        == FROZEN_V3_MISPLACED_GO_PAYLOAD_CANONICAL_SHA256
        and mispublication.get("exact_v3_go_key_count") == len(FROZEN_V3_GO_KEYS)
        and mispublication.get("exact_v3_go_schema") is True
        and mispublication.get("authorization_crossbinding_exact") is True
        and mispublication.get("review_crossbinding_exact") is True
        and mispublication.get("recovered_afterstate_crossbinding_exact") is True,
        "V3 path-mispublication incident payload mismatch",
    )
    required = incident.get("required_v4_supersession")
    _require(
        isinstance(required, Mapping)
        and required.get("status") == "REQUIRED_NOT_YET_AUTHORIZED"
        and required.get("incident") == V4_PATH_MISPUBLISH_INCIDENT_RELATIVE
        and required.get("artifact_paths_root_relative")
        == {
            "authorization": V4_AUTH_RELATIVE,
            "independent_review": V4_REVIEW_RELATIVE,
            "independent_go": V4_GO_RELATIVE,
        }
        and required.get("code_paths_repo_relative")
        == {
            "auditor": "scripts/audit_noaa_gfs_multiseason_raw_postrun_v4.py",
            "auditor_test": "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py",
            "authorization_sealer": "scripts/seal_noaa_gfs_multiseason_postrun_audit_v4.py",
            "authorization_sealer_test": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py",
        }
        and required.get("audit_report_file") is None
        and required.get("audit_files_written_allowed") == 0
        and required.get("stdout_only_required") is True
        and required.get("v4_publication_or_execution_authorized_by_this_incident")
        is False,
        "V3 path-mispublication incident V4 contract mismatch",
    )
    must_bind = required.get("must_bind")
    _require(
        isinstance(must_bind, Mapping)
        and must_bind.get("rejected_noncanonical_v3_go")
        == FROZEN_V3_MISPLACED_GO_IDENTITY
        and must_bind.get("v3_authorization") == FROZEN_V3_AUTH_IDENTITY
        and must_bind.get("v3_independent_review") == FROZEN_V3_REVIEW_IDENTITY
        and must_bind.get("unchanged_recovered_afterstate_canonical_sha256")
        == V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
        "V3 path-mispublication incident must-bind mismatch",
    )
    authority_scope = incident.get("authority_scope")
    prohibitions = incident.get("prohibitions")
    root_cause = incident.get("root_cause")
    failure = incident.get("failure_boundary")
    _require(
        isinstance(authority_scope, Mapping)
        and authority_scope.get("documentary_only") is True
        and authority_scope.get("canonical_v3_go_publication_authorized") is False
        and authority_scope.get("live_postrun_auditor_authorized") is False
        and authority_scope.get("network_authorized") is False
        and isinstance(prohibitions, Mapping)
        and all(value is True for value in prohibitions.values())
        and isinstance(root_cause, Mapping)
        and root_cause.get("classification") == "DESTINATION_PATH_SELECTION_ONLY"
        and root_cause.get("salvage_with_canonical_v3_go_blocked") is True
        and root_cause.get("v3_auditor_code_or_test_defect_observed") is False
        and isinstance(failure, Mapping)
        and failure.get("failed_before_any_parquet_or_data_read") is True
        and failure.get("live_postrun_auditor_started") is False
        and failure.get("full_offline_raw_to_decoded_replay_started") is False
        and failure.get("postrun_audit_output_files_written") == 0
        and failure.get("network_requests") == 0,
        "V3 path-mispublication incident authority/failure boundary mismatch",
    )
    return incident


def _frozen_v3_required_command() -> list[str]:
    return [
        str(Path(sys.executable).resolve()), "-B", "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v3", "--root",
        str(ROOT_DEFAULT.resolve()), "--authorization",
        str(path_under(ROOT_DEFAULT, FROZEN_V3_AUTH_RELATIVE).resolve()),
        "--independent-go",
        str(path_under(ROOT_DEFAULT, FROZEN_V3_CANONICAL_GO_RELATIVE).resolve()),
    ]


def _v4_validate_frozen_v3_chain(root: Path) -> dict[str, Any]:
    canonical_go = path_under(root, FROZEN_V3_CANONICAL_GO_RELATIVE)
    _require(
        not os.path.lexists(str(canonical_go)),
        "canonical V3 GO must remain absent",
    )
    authorization_actual = _v4_verify_exact_root_identity(
        root, FROZEN_V3_AUTH_IDENTITY, FROZEN_V3_AUTH_IDENTITY,
        label="frozen V3 authorization",
    )
    review_actual = _v4_verify_exact_root_identity(
        root, FROZEN_V3_REVIEW_IDENTITY, FROZEN_V3_REVIEW_IDENTITY,
        label="frozen V3 independent review",
    )
    misplaced_actual = _v4_verify_exact_root_identity(
        root, FROZEN_V3_MISPLACED_GO_IDENTITY, FROZEN_V3_MISPLACED_GO_IDENTITY,
        label="rejected misplaced V3 GO",
    )
    source_actual: dict[str, dict[str, Any]] = {}
    for role, expected in FROZEN_V3_SOURCE_IDENTITIES.items():
        path = {
            "superseded_v3_auditor": FROZEN_V3_AUDITOR,
            "superseded_v3_auditor_test": FROZEN_V3_AUDITOR_TEST,
            "superseded_v3_sealer": FROZEN_V3_SEALER,
            "superseded_v3_sealer_test": FROZEN_V3_SEALER_TEST,
        }[role]
        source_actual[role] = _v4_verify_external_identity(
            expected, path, label=role,
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    authorization = load_json(path_under(root, FROZEN_V3_AUTH_RELATIVE))
    review = load_json(path_under(root, FROZEN_V3_REVIEW_RELATIVE))
    misplaced = load_json(path_under(root, FROZEN_V3_MISPLACED_GO_RELATIVE))
    _require(set(authorization) == FROZEN_V3_AUTH_KEYS, "frozen V3 authorization schema mismatch")
    _require(set(review) == FROZEN_V3_REVIEW_KEYS, "frozen V3 review schema mismatch")
    _require(set(misplaced) == FROZEN_V3_GO_KEYS, "misplaced V3 GO schema mismatch")
    auth_created = _v4_created_utc(authorization.get("created_utc"), label="frozen V3 authorization")
    review_created = _v4_created_utc(review.get("created_utc"), label="frozen V3 review")
    go_created = _v4_created_utc(misplaced.get("created_utc"), label="misplaced V3 GO")
    attempt_id = authorization.get("audit_attempt_id")
    _require(
        authorization.get("schema_version") == 3
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V3"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V3"
        and attempt_id == "postrun_audit_v3__20260810T225719300660Z"
        and authorization.get("required_command") == _frozen_v3_required_command()
        and authorization.get("incident")
        == {
            "path": V2_FALSE_REJECT_INCIDENT_RELATIVE,
            "size_bytes": V2_FALSE_REJECT_INCIDENT_SIZE_BYTES,
            "sha256": V2_FALSE_REJECT_INCIDENT_SHA256,
        },
        "frozen V3 authorization header/incident/command mismatch",
    )
    _v4_validate_policy(authorization, label="frozen V3 authorization")
    v3_role_map = {
        "v3_auditor": "superseded_v3_auditor",
        "v3_auditor_test": "superseded_v3_auditor_test",
        "v3_sealer": "superseded_v3_sealer",
        "v3_sealer_test": "superseded_v3_sealer_test",
    }
    _require(
        all(authorization.get(field) == source_actual[role] for field, role in v3_role_map.items())
        and authorization.get("superseded_v2_auditor") == V4_SUPERSEDED_V2_AUDITOR_IDENTITY
        and authorization.get("superseded_v2_auditor_test")
        == V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY,
        "frozen V3 source-role binding mismatch",
    )
    afterstate = _v4_recovered_afterstate_from_authorization(authorization)
    _require(
        review.get("schema_version") == 3
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V3"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V3"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == attempt_id
        and review.get("authorization") == authorization_actual
        and review.get("incident") == authorization.get("incident")
        and review.get("recovered_afterstate") == afterstate
        and review_created >= auth_created,
        "frozen V3 review cross-binding/chronology mismatch",
    )
    _v4_validate_policy(review, label="frozen V3 review")
    _require(
        misplaced.get("schema_version") == 3
        and misplaced.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V3"
        and misplaced.get("status")
        == "GO_V3_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY"
        and misplaced.get("audit_attempt_id") == attempt_id
        and misplaced.get("authorization") == authorization_actual
        and misplaced.get("independent_review") == review_actual
        and misplaced.get("incident") == authorization.get("incident")
        and misplaced.get("recovered_afterstate") == afterstate
        and misplaced.get("required_command") == _frozen_v3_required_command()
        and misplaced.get("recovery_rerun_authorized") is False
        and go_created >= review_created >= auth_created
        and canonical_payload_sha256(misplaced)
        == FROZEN_V3_MISPLACED_GO_PAYLOAD_CANONICAL_SHA256,
        "misplaced V3 GO schema/cross-binding/chronology mismatch",
    )
    _v4_validate_policy(misplaced, label="misplaced V3 GO")
    for payload in (review, misplaced):
        _require(
            all(payload.get(field) == source_actual[role] for field, role in v3_role_map.items()),
            "frozen V3 review/GO source-role mismatch",
        )
    state = {
        "incident": {
            "path": V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
            "size_bytes": V4_PATH_MISPUBLISH_INCIDENT_SIZE_BYTES,
            "sha256": V4_PATH_MISPUBLISH_INCIDENT_SHA256,
        },
        "authorization": authorization_actual,
        "independent_review": review_actual,
        "misplaced_go": misplaced_actual,
        "canonical_go": {
            "path": FROZEN_V3_CANONICAL_GO_RELATIVE,
            "present": False,
            "must_remain_absent": True,
        },
        "review_predates_misplaced_go": review_created < go_created,
        "misplaced_go_payload_canonical_sha256": FROZEN_V3_MISPLACED_GO_PAYLOAD_CANONICAL_SHA256,
        "misplaced_go_payload_exact_v3_schema_and_crossbindings": True,
        "misplaced_go_authoritative": False,
        "v3_auditor_execution_authorized": False,
    }
    _require(
        set(state) == V4_PATH_MISPUBLISH_STATE_KEYS
        and state["review_predates_misplaced_go"] is True,
        "V3 path-mispublication reusable-state mismatch",
    )
    return {
        "authorization": authorization,
        "review": review,
        "misplaced_go": misplaced,
        "afterstate": afterstate,
        "source_identities": source_actual,
        "state": state,
    }


def _v4_validate_progress_declaration(progress: Any) -> None:
    _require(
        isinstance(progress, Mapping)
        and set(progress)
        == {"file_count", "inventory", "inventory_canonical_sha256", "final_checkpoint"},
        "V4 recovery-progress declaration schema mismatch",
    )
    inventory = progress.get("inventory")
    _require(
        progress.get("file_count") == 104
        and isinstance(inventory, list)
        and len(inventory) == 104,
        "V4 recovery-progress declaration count mismatch",
    )
    paths: list[str] = []
    for record in inventory:
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "size_bytes", "sha256"}
            and isinstance(record.get("path"), str)
            and str(record["path"]).startswith(
                f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__"
            )
            and type(record.get("size_bytes")) is int
            and int(record["size_bytes"]) > 0
            and isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])) is not None,
            "V4 recovery-progress inventory identity invalid",
        )
        paths.append(str(record["path"]))
    _require(paths == sorted(paths) and len(set(paths)) == 104, "V4 progress order/uniqueness mismatch")
    encoded = _v4_canonical_bytes(inventory)
    _require(
        len(encoded) == V4_PROGRESS_INVENTORY_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256
        and progress.get("inventory_canonical_sha256")
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256
        and progress.get("final_checkpoint") == V4_FINAL_PROGRESS_IDENTITY,
        "V4 recovery-progress canonical digest/final mismatch",
    )


def _v4_recovered_afterstate_from_authorization(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    transaction = authorization.get("recovery_transaction")
    history = authorization.get("recovery_history")
    progress = authorization.get("recovery_progress")
    outputs = authorization.get("canonical_outputs")
    _require(
        isinstance(transaction, Mapping)
        and set(transaction) == {"plan", "commit"}
        and transaction == V4_TRANSACTION_IDENTITIES,
        "V4 recovery-transaction declaration mismatch",
    )
    _require(
        isinstance(history, Mapping)
        and set(history) == {"file_count", "completion_locks", "postcommit_input_audit"}
        and history.get("file_count") == 3
        and history.get("completion_locks") == V4_COMPLETION_LOCK_IDENTITIES
        and history.get("postcommit_input_audit") == V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY,
        "V4 recovery-history declaration mismatch",
    )
    _v4_validate_progress_declaration(progress)
    _require(
        isinstance(outputs, Mapping) and dict(outputs) == V4_CANONICAL_OUTPUT_IDENTITIES,
        "V4 canonical-output declaration mismatch",
    )
    afterstate = {
        "attempt_id": V4_RECOVERY_ATTEMPT_ID,
        "canonical_output_count": 8,
        "canonical_outputs": dict(outputs),
        "transaction": {
            "plan": transaction["plan"],
            "commit": transaction["commit"],
            "total_recursive_files": 4,
            "original_preplan_remnants": 2,
            "v2_plan_files": 1,
            "v2_commit_files": 1,
            "v2_staged_files_remaining": 0,
        },
        "recovery_history_file_count": 3,
        "completion_locks": history["completion_locks"],
        "postcommit_input_audit": history["postcommit_input_audit"],
        "recovery_progress": progress,
        "raw_active_lock_present": False,
        "decoded_active_lock_present": False,
    }
    encoded = _v4_canonical_bytes(afterstate)
    _require(
        len(encoded) == V4_RECOVERED_AFTERSTATE_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
        "V4 recovered-afterstate canonical closure mismatch",
    )
    return afterstate


def _v4_validate_zero_snapshot(
    snapshot: Any, *, incident_postfailure_sha256: str
) -> None:
    _require(
        isinstance(snapshot, Mapping)
        and set(snapshot)
        == {
            "incident_postfailure_state_canonical_sha256",
            "recovered_afterstate_canonical_sha256", "active_locks",
            "v4_control_temporary_files", "postrun_audit_output_files_present",
            "network_requests", "audit_files_written", "labels_read",
            "arrays_2024_read", "arrays_2025_read", "models_fit",
            "submission_csv_created",
        },
        "V4 preaudit zero-mutation snapshot schema mismatch",
    )
    _require(
        snapshot.get("incident_postfailure_state_canonical_sha256")
        == incident_postfailure_sha256
        and snapshot.get("recovered_afterstate_canonical_sha256")
        == V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        and snapshot.get("active_locks")
        == {
            "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
            "decoded": {"path": "decoded/DECODE_RECOVERY_ACTIVE.lock", "present": False},
        }
        and snapshot.get("v4_control_temporary_files")
        == {
            "checked_destination_relative_paths": [
                FROZEN_V3_AUTH_RELATIVE,
                FROZEN_V3_REVIEW_RELATIVE,
                FROZEN_V3_CANONICAL_GO_RELATIVE,
                FROZEN_V3_MISPLACED_GO_RELATIVE,
                V4_AUTH_RELATIVE,
                V4_REVIEW_RELATIVE,
                V4_GO_RELATIVE,
                V2_FALSE_REJECT_INCIDENT_RELATIVE,
                V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
            ],
            "matching_temporary_files": 0,
        }
        and snapshot.get("postrun_audit_output_files_present") == 0
        and snapshot.get("network_requests") == 0
        and snapshot.get("audit_files_written") == 0
        and snapshot.get("labels_read") is False
        and snapshot.get("arrays_2024_read") is False
        and snapshot.get("arrays_2025_read") is False
        and snapshot.get("models_fit") == 0
        and snapshot.get("submission_csv_created") is False,
        "V4 preaudit zero-mutation snapshot values mismatch",
    )


def _v4_validate_test_evidence(
    evidence: Any,
    *,
    authorization_created: datetime,
    bound_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(
        isinstance(evidence, Mapping) and set(evidence) == V4_TEST_EVIDENCE_KEYS,
        "V4 immutable test-evidence schema mismatch",
    )
    evidence_created = _v4_created_utc(evidence.get("created_utc"), label="V4 test evidence")
    _require(
        evidence_created <= authorization_created
        and evidence.get("schema_version") == 1
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_TEST_EVIDENCE"
        and evidence.get("status") == "PASS_FROZEN_V4_AUDITOR_AND_SEALER_TESTS"
        and evidence.get("pytest_isolation")
        == "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS"
        and evidence.get("required_test_names") == list(V4_REQUIRED_TEST_NAMES)
        and evidence.get("required_test_names_present") is True
        and evidence.get("all_exit_codes_zero") is True,
        "V4 immutable test-evidence header/result mismatch",
    )
    _require(
        evidence.get("bound_identities") == bound_identities,
        "V4 immutable test-evidence bound identities mismatch",
    )
    source_compile = evidence.get("source_compile")
    _require(
        isinstance(source_compile, Mapping)
        and set(source_compile) == {"method", "result", "files"}
        and source_compile.get("method") == "compile_exact_source_no_pyc"
        and source_compile.get("result") == "PASS"
        and source_compile.get("files")
        == [
            bound_identities["v4_auditor"],
            bound_identities["v4_auditor_test"],
            bound_identities["v4_sealer"],
            bound_identities["v4_sealer_test"],
        ],
        "V4 exact-source compile evidence mismatch",
    )
    runs = evidence.get("test_runs")
    expected_commands = {
        "v4_auditor_tests": _v4_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py"
        ),
        "v4_sealer_tests": _v4_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py"
        ),
        "frozen_v3_auditor_tests": _v4_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v3.py"
        ),
        "frozen_v3_sealer_tests": _v4_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v3.py"
        ),
        "frozen_v2_auditor_tests": _v4_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py"
        ),
    }
    _require(
        isinstance(runs, Mapping) and set(runs) == V4_TEST_RUN_KEYS,
        "V4 immutable pytest run map mismatch",
    )
    for role, expected_command in expected_commands.items():
        run = runs[role]
        _require(
            isinstance(run, Mapping)
            and set(run) == V4_TEST_RUN_RECORD_KEYS
            and run.get("command") == expected_command
            and run.get("exit_code") == 0
            and isinstance(run.get("summary"), str)
            and "passed" in str(run["summary"])
            and "failed" not in str(run["summary"]).casefold()
            and "error" not in str(run["summary"]).casefold()
            and isinstance(run.get("stdout_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stdout_sha256"])) is not None
            and isinstance(run.get("stderr_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stderr_sha256"])) is not None
            and run.get("network_guard_installed") is True
            and run.get("cacheprovider_disabled") is True,
            f"V4 immutable pytest run mismatch: {role}",
        )
    _require(
        evidence.get("production_shape_regression")
        == {
            "four_key_identity_passed": True,
            "transaction_reached": True,
            "raw_reached": True,
            "decoded_reached": True,
            "offline_replay_reached": True,
            "synthetic_fixture": True,
        }
        and evidence.get("real_seven_spawn_regression")
        == {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        }
        and evidence.get("stdout_only_regression") is True
        and evidence.get("network_guard")
        == {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        }
        and evidence.get("network_requests") == 0
        and evidence.get("audit_files_written") == 0
        and evidence.get("labels_read") is False
        and evidence.get("arrays_2024_read") is False
        and evidence.get("arrays_2025_read") is False
        and evidence.get("models_fit") == 0
        and evidence.get("submission_csv_created") is False,
        "V4 immutable test-evidence safety/regression mismatch",
    )


def _v4_validate_policy(payload: Mapping[str, Any], *, label: str) -> None:
    _require(
        payload.get("max_spawn_processes") == 7
        and payload.get("network_requests_allowed") == 0
        and payload.get("audit_files_written_allowed") == 0
        and payload.get("labels_read_allowed") is False
        and payload.get("arrays_2024_read_allowed") is False
        and payload.get("arrays_2025_read_allowed") is False
        and payload.get("models_fit_allowed") == 0
        and payload.get("submission_csv_allowed") is False
        and payload.get("full_offline_redecode_required") is True
        and payload.get("stdout_only") is True,
        f"{label} execution policy mismatch",
    )


def _v4_validate_predata_authority(
    root: Path, authorization_path: Path, go_path: Path
) -> dict[str, Any]:
    """Validate the complete V4 control chain before any output/data dereference."""

    expected_authorization_path = path_under(root, V4_AUTH_RELATIVE)
    expected_go_path = path_under(root, V4_GO_RELATIVE)
    expected_review_path = path_under(root, V4_REVIEW_RELATIVE)
    _require(
        Path(os.path.abspath(authorization_path)) == expected_authorization_path
        and Path(os.path.abspath(go_path)) == expected_go_path,
        "V4 authorization/GO paths are not canonical",
    )
    for path, label in (
        (expected_authorization_path, "V4 authorization"),
        (expected_review_path, "V4 independent review"),
        (expected_go_path, "V4 independent GO"),
    ):
        require_no_symlink_chain(path, root, label=label)
        _require(
            os.path.lexists(str(path)) and path.is_file() and not _linklike(path),
            f"{label} is absent, non-regular or link-like",
        )
    namespace_start = _v4_validate_control_namespace(root)
    _require(_v4_control_temporary_files(root) == [], "V4 control temporary files present")
    _require(
        _v4_postrun_report_candidates(root) == [],
        "persisted V2/V3/V4 postrun report candidate is forbidden",
    )

    authorization = load_json(expected_authorization_path)
    review = load_json(expected_review_path)
    go = load_json(expected_go_path)
    _require(set(authorization) == V4_AUTH_KEYS, "V4 authorization schema mismatch")
    _require(set(review) == V4_REVIEW_KEYS, "V4 independent review schema mismatch")
    _require(set(go) == V4_GO_KEYS, "V4 independent GO schema mismatch")
    authorization_created = _v4_created_utc(
        authorization.get("created_utc"), label="V4 authorization"
    )
    review_created = _v4_created_utc(review.get("created_utc"), label="V4 independent review")
    go_created = _v4_created_utc(go.get("created_utc"), label="V4 independent GO")
    attempt_id = authorization.get("audit_attempt_id")
    _require(
        authorization.get("schema_version") == 4
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V4"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4"
        and isinstance(attempt_id, str)
        and re.fullmatch(r"postrun_audit_v4__[0-9]{8}T[0-9]{12}Z", attempt_id)
        is not None
        and authorization.get("recovery_attempt_id") == V4_RECOVERY_ATTEMPT_ID
        and authorization.get("runtime_identity_sha256") == V4_RUNTIME_IDENTITY_SHA256
        and authorization.get("independent_review_required") is True
        and authorization.get("independent_go_required") is True,
        "V4 authorization header/binding mismatch",
    )
    _v4_validate_policy(authorization, label="V4 authorization")

    path_incident = _v4_validate_path_mispublish_incident(
        root, authorization.get("incident")
    )
    path_incident_identity = identity(
        path_under(root, V4_PATH_MISPUBLISH_INCIDENT_RELATIVE), root
    )
    v2_incident = _v4_validate_false_reject_incident(
        root, authorization.get("v2_false_reject_incident")
    )
    v2_incident_identity = identity(
        path_under(root, V2_FALSE_REJECT_INCIDENT_RELATIVE), root
    )
    frozen_v3 = _v4_validate_frozen_v3_chain(root)
    v3_state = frozen_v3["state"]
    _require(
        authorization.get("v3_path_mispublish_state") == v3_state,
        "V4 authorization V3 path-mispublication state mismatch",
    )

    for field, expected in (
        ("superseded_v2_auditor", V4_SUPERSEDED_V2_AUDITOR_IDENTITY),
        ("superseded_v2_auditor_test", V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY),
    ):
        _v4_require_exact_declared_identity(
            authorization.get(field), expected, label=field
        )
    _v4_verify_external_identity(
        authorization["superseded_v2_auditor"], SUPERSEDED_V2_AUDITOR,
        label="superseded V2 auditor",
        expected_size_sha256=(300_379, V4_SUPERSEDED_V2_AUDITOR_IDENTITY["sha256"]),
    )
    _v4_verify_external_identity(
        authorization["superseded_v2_auditor_test"], SUPERSEDED_V2_AUDITOR_TEST,
        label="superseded V2 auditor test",
        expected_size_sha256=(133_607, V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY["sha256"]),
    )
    for role, expected in FROZEN_V3_SOURCE_IDENTITIES.items():
        _v4_require_exact_declared_identity(
            authorization.get(role), expected, label=role
        )
        _require(
            frozen_v3["source_identities"][role] == expected,
            f"frozen V3 source rehash mismatch: {role}",
        )
    v4_actual = {
        "v4_auditor": _v4_verify_external_identity(
            authorization.get("v4_auditor"), Path(__file__).resolve(), label="V4 auditor"
        ),
        "v4_auditor_test": _v4_verify_external_identity(
            authorization.get("v4_auditor_test"), AUDITOR_TEST, label="V4 auditor test"
        ),
        "v4_sealer": _v4_verify_external_identity(
            authorization.get("v4_sealer"), V4_SEALER, label="V4 sealer"
        ),
        "v4_sealer_test": _v4_verify_external_identity(
            authorization.get("v4_sealer_test"), V4_SEALER_TEST, label="V4 sealer test"
        ),
    }

    controls = authorization.get("v2_recovery_controls")
    _require(
        isinstance(controls, Mapping) and set(controls) == set(V4_V2_CONTROL_IDENTITIES),
        "V4 V2-control binding schema mismatch",
    )
    for role, expected in V4_V2_CONTROL_IDENTITIES.items():
        _v4_verify_exact_root_identity(
            root, controls[role], expected, label=f"V4 bound V2 control {role}"
        )
    support = authorization.get("v2_recovery_support")
    _require(
        isinstance(support, Mapping) and set(support) == set(V4_V2_SUPPORT_PATHS),
        "V4 V2-support binding schema mismatch",
    )
    for role, expected_path in V4_V2_SUPPORT_PATHS.items():
        _v4_verify_external_identity(
            support[role], expected_path, label=f"V4 bound V2 support {role}",
            expected_size_sha256=V4_V2_SUPPORT_SIZE_SHA256[role],
        )

    original = authorization.get("original_v1_provenance")
    _require(
        isinstance(original, Mapping) and set(original) == set(V4_ORIGINAL_V1_PROVENANCE),
        "V4 original V1 provenance schema mismatch",
    )
    raw_record = original["original_raw_authorization_v1"]
    prelaunch_record = original["original_independent_prelaunch_audit_v1"]
    _v4_verify_exact_root_identity(
        root, raw_record, V4_ORIGINAL_V1_PROVENANCE["original_raw_authorization_v1"],
        label="V4 original raw authorization",
    )
    _v4_require_exact_declared_identity(
        prelaunch_record,
        V4_ORIGINAL_V1_PROVENANCE["original_independent_prelaunch_audit_v1"],
        label="V4 original independent prelaunch audit",
    )
    verify_identity(
        prelaunch_record, path_under(root, str(prelaunch_record["path"])), root=root,
        label="V4 original independent prelaunch audit", allowed_extra_fields=("status",),
    )
    _require(
        prelaunch_record.get("status") == "PASS_TO_CREATE_FINAL_AUTH_ONLY",
        "V4 original independent prelaunch status mismatch",
    )
    raw_authorization = load_json(path_under(root, str(raw_record["path"])))
    _require(
        raw_authorization.get("independent_prelaunch_audit") == prelaunch_record,
        "V4 original raw authorization four-key prelaunch binding mismatch",
    )
    runtime_identity = raw_authorization.get("runtime_identity")
    _require(
        isinstance(runtime_identity, Mapping)
        and set(runtime_identity)
        == {"packages", "platform", "python_executable", "python_executable_sha256",
            "python_executable_size_bytes", "python_version"}
        and isinstance(runtime_identity.get("packages"), Mapping)
        and set(runtime_identity["packages"]) == {"eccodes", "numpy", "pandas", "pyarrow"},
        "V4 original runtime identity schema mismatch",
    )
    for package, package_record in runtime_identity["packages"].items():
        _require(
            isinstance(package_record, Mapping)
            and set(package_record)
            == {"module_file", "module_file_sha256", "module_file_size_bytes", "version"},
            f"V4 original runtime package schema mismatch: {package}",
        )
    runtime_bytes = _v4_canonical_bytes(runtime_identity)
    _require(
        len(runtime_bytes) == 1_405
        and hashlib.sha256(runtime_bytes).hexdigest() == V4_RUNTIME_IDENTITY_SHA256
        and raw_authorization.get("runtime_identity_sha256") == V4_RUNTIME_IDENTITY_SHA256,
        "V4 original runtime identity canonical digest mismatch",
    )

    afterstate = _v4_recovered_afterstate_from_authorization(authorization)
    _require(afterstate == frozen_v3["afterstate"], "V4/V3 recovered afterstate mismatch")
    _v4_validate_zero_snapshot(
        authorization.get("preaudit_zero_mutation_snapshot"),
        incident_postfailure_sha256=V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
    )
    _require(
        authorization.get("required_command")
        == _v4_required_command(root, expected_authorization_path, expected_go_path),
        "V4 authorization required command mismatch",
    )
    bound_identities = {
        "v2_auditor": authorization["superseded_v2_auditor"],
        "v2_auditor_test": authorization["superseded_v2_auditor_test"],
        **{role: authorization[role] for role in FROZEN_V3_SOURCE_IDENTITIES},
        **v4_actual,
    }
    _v4_validate_test_evidence(
        authorization.get("test_evidence"),
        authorization_created=authorization_created,
        bound_identities=bound_identities,
    )

    _require(
        review.get("schema_version") == 4
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V4"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V4"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == attempt_id
        and review.get("auditor_execution_started") is False
        and review.get("independent_go_required") is True
        and review_created > authorization_created,
        "V4 independent review header/verdict mismatch",
    )
    _v4_validate_policy(review, label="V4 independent review")
    _require(
        go.get("schema_version") == 4
        and go.get("artifact_type") == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V4"
        and go.get("status") == "GO_V4_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY"
        and go.get("audit_attempt_id") == attempt_id
        and go.get("recovery_attempt_id") == V4_RECOVERY_ATTEMPT_ID
        and go.get("recovery_rerun_authorized") is False
        and go.get("required_command") == authorization.get("required_command")
        and go_created > review_created,
        "V4 independent GO header/binding mismatch",
    )
    _v4_validate_policy(go, label="V4 independent GO")
    authorization_actual = identity(expected_authorization_path, root)
    review_actual = identity(expected_review_path, root)
    _require(
        review.get("authorization") == authorization_actual
        and go.get("authorization") == authorization_actual
        and go.get("independent_review") == review_actual
        and review.get("incident") == path_incident_identity
        and go.get("incident") == path_incident_identity
        and review.get("v2_false_reject_incident") == v2_incident_identity
        and go.get("v2_false_reject_incident") == v2_incident_identity
        and review.get("v3_path_mispublish_state") == v3_state
        and go.get("v3_path_mispublish_state") == v3_state
        and all(
            review.get(role) == authorization[role]
            and go.get(role) == authorization[role]
            for role in (
                "superseded_v2_auditor", "superseded_v2_auditor_test",
                *FROZEN_V3_SOURCE_IDENTITIES.keys(),
                "v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test",
            )
        )
        and review.get("v2_recovery_controls") == controls
        and review.get("recovered_afterstate") == afterstate
        and go.get("recovered_afterstate") == afterstate,
        "V4 review/GO cross-binding mismatch",
    )
    checks = review.get("independent_checks")
    recheck = review.get("test_evidence_recheck")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == V4_REVIEW_CHECKS
        and all(checks.get(key) is True for key in V4_REVIEW_CHECKS),
        "V4 independent review check set/verdict mismatch",
    )
    _require(
        isinstance(recheck, Mapping)
        and set(recheck) == V4_REVIEW_TEST_RECHECK_KEYS
        and recheck.get("authorization_test_evidence_canonical_sha256")
        == canonical_payload_sha256(authorization["test_evidence"])
        and all(
            recheck.get(key) is True
            for key in V4_REVIEW_TEST_RECHECK_KEYS
            if key != "authorization_test_evidence_canonical_sha256"
        ),
        "V4 independent test-evidence recheck mismatch",
    )
    _require(
        _v4_validate_control_namespace(root) == namespace_start
        and _v4_control_temporary_files(root) == []
        and _v4_postrun_report_candidates(root) == [],
        "V4 predata control namespace changed",
    )
    return {
        "authorization": authorization,
        "authorization_identity": authorization_actual,
        "review": review,
        "review_identity": review_actual,
        "go": go,
        "go_identity": identity(expected_go_path, root),
        "incident": path_incident,
        "incident_identity": path_incident_identity,
        "v2_false_reject_incident": v2_incident,
        "v2_false_reject_incident_identity": v2_incident_identity,
        "v3_path_mispublish_state": v3_state,
        "frozen_v3_source_identities": frozen_v3["source_identities"],
        "recovered_afterstate": afterstate,
        "v4_auditor_identity": v4_actual["v4_auditor"],
        "v4_auditor_test_identity": v4_actual["v4_auditor_test"],
        "v4_sealer_identity": v4_actual["v4_sealer"],
        "v4_sealer_test_identity": v4_actual["v4_sealer_test"],
        "control_namespace": namespace_start,
    }


def _v7_validate_postauthority_control_state(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = authority["authorization"]
    afterstate = authority["recovered_afterstate"]
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock"))),
        "V7 recovered afterstate has an active lock",
    )
    commitment = _v5_recovered_afterstate_commitment(afterstate)
    _require(
        commitment == authority["recovered_afterstate_commitment"]
        and _v7_control_temporary_files(root) == []
        and _v7_postrun_report_candidates(root) == [],
        "V7 afterstate/commitment/temp/report recheck mismatch",
    )
    timestamp_incident = _v7_validate_timestamp_false_reject_incident(
        root, authorization["incident"]
    )
    _require(
        timestamp_incident["identity"] == authority["timestamp_incident"]["identity"],
        "V7 timestamp incident changed during audit",
    )
    false_reject = _v6_validate_v5_false_reject_incident(
        root,
        {
            "path": V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
            "size_bytes": V6_HISTORICAL_FALSE_REJECT_INCIDENT_SIZE_BYTES,
            "sha256": V6_HISTORICAL_FALSE_REJECT_INCIDENT_SHA256,
        },
    )
    _require(
        false_reject["identity"] == authority["false_reject_incident"]["identity"],
        "V7 transitive V5 false-reject incident changed during audit",
    )
    direct_path = path_under(root, RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE)
    direct_identity = identity(direct_path, root)
    direct_payload = _v5_load_strict_json(
        direct_path, label="postaudit historical 9e74"
    )
    historical = _v6_validate_historical_9e74_payload(
        direct_payload, direct_identity
    )
    _v6_validate_historical_contract(
        authorization["historical_failed_head_identity_contract"], historical
    )
    v5_base = _v6_load_and_validate_v5_authority_base(
        root, authorization["v5_authority_base"]
    )
    _require(
        v5_base["recovered_afterstate"] == afterstate
        and v5_base["commitment"] == commitment,
        "V7 reconstructed V5 afterstate changed during audit",
    )
    timestamp_chain = _v7_validate_timestamp_chain(
        timestamp_incident=timestamp_incident,
        direct_payload=direct_payload,
        false_reject=false_reject,
        v5_base=v5_base,
        authorization=authority["authorization"],
        review=authority["review"],
        go=authority["go"],
    )
    _require(
        timestamp_chain == authority["timestamp_chain"],
        "V7 raw timestamp chain changed during audit",
    )
    for role, record, relative in (
        ("authorization", authority["authorization_identity"], V7_AUTH_RELATIVE),
        ("independent review", authority["review_identity"], V7_REVIEW_RELATIVE),
        ("independent GO", authority["go_identity"], V7_GO_RELATIVE),
    ):
        verify_identity(
            record, path_under(root, relative), root=root, label=f"V7 postaudit {role}"
        )
    for role, path in (
        ("v7_auditor", Path(__file__).resolve()),
        ("v7_auditor_test", AUDITOR_TEST),
        ("v7_sealer", V7_SEALER),
        ("v7_sealer_test", V7_SEALER_TEST),
    ):
        _v4_verify_external_identity(
            authorization[role], path, label=f"V7 postaudit {role}"
        )
    frozen_v6_paths = {
        "superseded_v6_auditor": FROZEN_V6_AUDITOR,
        "superseded_v6_auditor_test": FROZEN_V6_AUDITOR_TEST,
        "superseded_v6_sealer": FROZEN_V6_SEALER,
        "superseded_v6_sealer_test": FROZEN_V6_SEALER_TEST,
    }
    for role, expected in FROZEN_V6_SOURCE_IDENTITIES.items():
        _v4_verify_external_identity(
            authorization[role], frozen_v6_paths[role], label=f"V7 postaudit {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    _v7_validate_zero_snapshot(root, authorization["preaudit_zero_mutation_snapshot"])
    _require(
        _v7_validate_control_namespace(root) == authority["control_namespace"],
        "V7 postauthority control namespace changed",
    )
    return commitment


def _v4_validate_postauthority_afterstate(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = authority["authorization"]
    afterstate = authority["recovered_afterstate"]
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock"))),
        "V4 recovered afterstate has an active lock",
    )
    for role, record in authorization["canonical_outputs"].items():
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V4 canonical output {role}",
        )
    for role, record in authorization["recovery_transaction"].items():
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V4 recovery transaction {role}",
        )
    history = authorization["recovery_history"]
    for role, record in history["completion_locks"].items():
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V4 recovery completion lock {role}",
        )
    verify_identity(
        history["postcommit_input_audit"],
        path_under(root, str(history["postcommit_input_audit"]["path"])),
        root=root,
        label="V4 recovery postcommit input audit",
    )
    history_dir = path_under(root, "decoded/recovery_history")
    _require(
        history_dir.is_dir() and not _linklike(history_dir),
        "V4 recovery-history directory invalid",
    )
    expected_history_paths = {
        str(record["path"])
        for record in history["completion_locks"].values()
    } | {str(history["postcommit_input_audit"]["path"])}
    actual_history_entries = list(history_dir.iterdir())
    _require(
        len(actual_history_entries) == 3
        and all(path.is_file() and not _linklike(path) for path in actual_history_entries)
        and {path.relative_to(root).as_posix() for path in actual_history_entries}
        == expected_history_paths,
        "V4 recovery-history exact filesystem closure mismatch",
    )

    transaction_root = path_under(root, "raw/output_transactions")
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "V4 output-transaction root invalid",
    )
    recovery_authorization_v2 = load_json(
        path_under(root, V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"])
    )
    remnants = recovery_authorization_v2.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list) and len(remnants) == 2,
        "V4 incident-bound preplan-remnant declaration mismatch",
    )
    expected_transaction_files = {
        str(authorization["recovery_transaction"]["plan"]["path"]),
        str(authorization["recovery_transaction"]["commit"]["path"]),
    }
    for index, record in enumerate(remnants):
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "size_bytes", "sha256"},
            f"V4 preplan remnant identity invalid: {index}",
        )
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V4 incident-bound preplan remnant {index}",
        )
        expected_transaction_files.add(str(record["path"]))
    expected_transaction_dirs = {
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged/raw",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    actual_transaction_files: set[str] = set()
    actual_transaction_dirs: set[str] = set()
    for path in transaction_root.rglob("*"):
        _require(not _linklike(path), "V4 transaction tree contains a link/junction")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            actual_transaction_files.add(relative)
        elif path.is_dir():
            actual_transaction_dirs.add(relative)
        else:
            raise AuditFailure("V4 transaction tree contains a special object")
    _require(
        actual_transaction_files == expected_transaction_files
        and actual_transaction_dirs == expected_transaction_dirs
        and len(actual_transaction_files) == 4,
        "V4 transaction recursive exact filesystem closure mismatch",
    )
    progress_dir = path_under(root, "decoded/recovery_progress")
    _require(progress_dir.is_dir() and not _linklike(progress_dir), "V4 progress directory invalid")
    actual_progress_paths = sorted(
        (path for path in progress_dir.iterdir()), key=lambda path: path.name
    )
    _require(
        len(actual_progress_paths) == 104
        and all(path.is_file() and not _linklike(path) for path in actual_progress_paths),
        "V4 recovery progress filesystem closure mismatch",
    )
    actual_inventory = [identity(path, root) for path in actual_progress_paths]
    _require(
        actual_inventory == authorization["recovery_progress"]["inventory"],
        "V4 recovery progress inventory rehash mismatch",
    )
    encoded = _v4_canonical_bytes(actual_inventory)
    _require(
        len(encoded) == V4_PROGRESS_INVENTORY_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "V4 recovery progress actual canonical digest mismatch",
    )
    _require(
        canonical_payload_sha256(afterstate)
        == V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        and _v4_control_temporary_files(root) == []
        and _v4_postrun_report_candidates(root) == [],
        "V4 afterstate/temp recheck mismatch",
    )
    _v4_validate_path_mispublish_incident(
        root, authority["authorization"]["incident"]
    )
    _v4_validate_false_reject_incident(
        root, authority["authorization"]["v2_false_reject_incident"]
    )
    for role, record, relative in (
        ("authorization", authority["authorization_identity"], V4_AUTH_RELATIVE),
        ("independent review", authority["review_identity"], V4_REVIEW_RELATIVE),
        ("independent GO", authority["go_identity"], V4_GO_RELATIVE),
    ):
        verify_identity(
            record, path_under(root, relative), root=root,
            label=f"V4 postaudit {role}",
        )
    for role, path in (
        ("v4_auditor", Path(__file__).resolve()),
        ("v4_auditor_test", AUDITOR_TEST),
        ("v4_sealer", V4_SEALER),
        ("v4_sealer_test", V4_SEALER_TEST),
    ):
        _v4_verify_external_identity(
            authority["authorization"][role], path, label=f"V4 postaudit {role}"
        )
    frozen_v3 = _v4_validate_frozen_v3_chain(root)
    _require(
        frozen_v3["state"] == authority["v3_path_mispublish_state"]
        and _v4_validate_control_namespace(root) == authority["control_namespace"],
        "V4 postauthority V3 chain/control namespace recheck mismatch",
    )
    return {
        "recovered_afterstate": afterstate,
        "recovered_afterstate_canonical_sha256": V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
        "canonical_output_count": 8,
        "transaction_recursive_file_count": 4,
        "transaction_recursive_directory_count": 7,
        "recovery_history_file_count": 3,
        "recovery_progress_file_count": 104,
        "active_locks_present": 0,
        "v4_control_temporary_files_present": 0,
        "canonical_v3_go_present": 0,
        "misplaced_v3_go_rejected_and_rehashed": True,
        "control_namespace_exact": True,
    }


def _v5_required_command(
    root: Path, authorization_path: Path, go_path: Path
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
        "--root",
        str(root.resolve()),
        "--authorization",
        str(authorization_path.resolve()),
        "--independent-go",
        str(go_path.resolve()),
    ]


def _v5_require_canonical_runtime_entrypoint(
    root: Path, authorization_path: Path, go_path: Path
) -> None:
    _require(
        __spec__ is not None
        and __spec__.name == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5"
        and __package__ == "scripts",
        "V5 production audit requires the canonical -m module entrypoint",
    )
    _require(
        sys.dont_write_bytecode is True
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None,
        "V5 production audit bytecode/cache environment mismatch",
    )
    expected_argv = [
        str(Path(__file__).resolve()),
        "--root",
        str(root.resolve()),
        "--authorization",
        str(authorization_path.resolve()),
        "--independent-go",
        str(go_path.resolve()),
    ]
    actual_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    _require(actual_argv == expected_argv, "V5 production argv is not exact canonical order")
    actual_orig_argv = [
        str(Path(sys.orig_argv[0]).resolve()), *sys.orig_argv[1:]
    ]
    _require(
        actual_orig_argv == _v5_required_command(root, authorization_path, go_path),
        "V5 production interpreter/module argv is not exact authorized command",
    )


def _v5_test_command(relative: str) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-c",
        V5_PYTEST_NETWORK_GUARD_SOURCE,
        "-q",
        "-p",
        "no:cacheprovider",
        relative,
    ]


def _v5_control_destination_relatives() -> tuple[str, ...]:
    return (
        FROZEN_V3_AUTH_RELATIVE,
        FROZEN_V3_REVIEW_RELATIVE,
        FROZEN_V3_CANONICAL_GO_RELATIVE,
        FROZEN_V3_MISPLACED_GO_RELATIVE,
        V4_AUTH_RELATIVE,
        V4_REVIEW_RELATIVE,
        V4_GO_RELATIVE,
        V5_AUTH_RELATIVE,
        V5_REVIEW_RELATIVE,
        V5_GO_RELATIVE,
        V2_FALSE_REJECT_INCIDENT_RELATIVE,
        V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        V5_TRANSPORT_INCIDENT_RELATIVE,
        V5_TRANSPORT_CORRECTION_RELATIVE,
    )


def _v5_control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in _v5_control_destination_relatives():
        destination = path_under(root, relative)
        _require(
            destination.parent.is_dir() and not _linklike(destination.parent),
            f"V5 control parent is absent or link-like: {relative}",
        )
        folded = destination.name.casefold()
        for candidate in destination.parent.iterdir():
            if candidate.name == destination.name:
                continue
            name = candidate.name.casefold()
            if (
                name.startswith(f".{folded}.")
                or name.startswith(f"{folded}.")
                or name == f"{folded}.tmp"
            ):
                matches.append(candidate.relative_to(root).as_posix())
    return sorted(set(matches))


def _v5_postrun_report_candidates(root: Path) -> list[str]:
    present: list[str] = []
    for relative in V5_POSTRUN_REPORT_CANDIDATES:
        path = path_under(root, relative)
        if not os.path.lexists(str(path)):
            continue
        _require(
            path.is_file() and not _linklike(path),
            f"V5 postrun report candidate is non-regular/link-like: {relative}",
        )
        present.append(relative)
    return present


def _v5_namespace_expected() -> dict[str, set[str]]:
    return {
        "prereg": {
            Path(FROZEN_V3_AUTH_RELATIVE).name,
            Path(FROZEN_V3_MISPLACED_GO_RELATIVE).name,
            Path(V4_AUTH_RELATIVE).name,
            Path(V5_AUTH_RELATIVE).name,
        },
        "independent_redteam": {
            Path(FROZEN_V3_REVIEW_RELATIVE).name,
            Path(V4_REVIEW_RELATIVE).name,
            Path(V5_REVIEW_RELATIVE).name,
            Path(V5_GO_RELATIVE).name,
        },
        "incidents": {
            Path(V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
            Path(V5_TRANSPORT_INCIDENT_RELATIVE).name,
            Path(V5_TRANSPORT_CORRECTION_RELATIVE).name,
        },
    }


def _v5_validate_control_namespace(root: Path) -> dict[str, list[str]]:
    prefixes = {
        "prereg": "decode_recovery_postrun_audit_",
        "independent_redteam": "track_a_decode_recovery_postrun_auditor_",
        "incidents": "track_a_decode_recovery_postrun_auditor_",
    }
    expected = _v5_namespace_expected()
    result: dict[str, list[str]] = {}
    for relative, prefix in prefixes.items():
        parent = path_under(root, relative)
        _require(
            os.path.lexists(str(parent))
            and parent.is_dir()
            and not _linklike(parent),
            f"V5 control namespace parent invalid: {relative}",
        )
        matched: list[str] = []
        for entry in parent.iterdir():
            if not entry.name.casefold().lstrip(".").startswith(prefix):
                continue
            _require(
                os.path.lexists(str(entry))
                and entry.is_file()
                and not _linklike(entry),
                f"V5 control namespace contains non-regular/link entry: {entry}",
            )
            matched.append(entry.name)
        _require(
            set(matched) == expected[relative]
            and len({name.casefold() for name in matched}) == len(matched),
            f"V5 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched, key=str.casefold)
    for relative, label in (
        (FROZEN_V3_CANONICAL_GO_RELATIVE, "canonical V3 GO"),
        (FROZEN_V4_CANONICAL_GO_RELATIVE, "canonical V4 GO"),
    ):
        _require(
            not os.path.lexists(str(path_under(root, relative))),
            f"{label} must remain absent permanently",
        )
    return result


def _v5_validate_policy(payload: Mapping[str, Any], *, label: str) -> None:
    _require(
        payload.get("max_spawn_processes") == 7
        and payload.get("network_requests_allowed") == 0
        and payload.get("audit_files_written_allowed") == 0
        and payload.get("labels_read_allowed") is False
        and payload.get("arrays_2024_read_allowed") is False
        and payload.get("arrays_2025_read_allowed") is False
        and payload.get("models_fit_allowed") == 0
        and payload.get("submission_csv_allowed") is False
        and payload.get("full_offline_redecode_required") is True
        and payload.get("stdout_only") is True,
        f"{label} execution policy mismatch",
    )


def _v5_load_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is absent")

    def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditFailure(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except AuditFailure:
        raise
    except Exception as exc:
        raise AuditFailure(f"malformed {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} root is not an object")
    return payload


def _v5_validate_incident_chain(
    root: Path, incident_record: Any, correction_record: Any
) -> dict[str, Any]:
    incident_expected = {
        "path": V5_TRANSPORT_INCIDENT_RELATIVE,
        "size_bytes": V5_TRANSPORT_INCIDENT_SIZE_BYTES,
        "sha256": V5_TRANSPORT_INCIDENT_SHA256,
    }
    correction_expected = {
        "path": V5_TRANSPORT_CORRECTION_RELATIVE,
        "size_bytes": V5_TRANSPORT_CORRECTION_SIZE_BYTES,
        "sha256": V5_TRANSPORT_CORRECTION_SHA256,
    }
    _v4_verify_exact_root_identity(
        root, incident_record, incident_expected, label="V4 review truncation incident"
    )
    _v4_verify_exact_root_identity(
        root,
        correction_record,
        correction_expected,
        label="V4 review truncation inventory-digest correction",
    )
    raw_incident = load_json(path_under(root, V5_TRANSPORT_INCIDENT_RELATIVE))
    correction = load_json(path_under(root, V5_TRANSPORT_CORRECTION_RELATIVE))
    raw_incident_bytes = _v4_canonical_bytes(raw_incident)
    correction_bytes = _v4_canonical_bytes(correction)
    _require(
        len(raw_incident_bytes) == V5_PRIMARY_INCIDENT_CANONICAL_BYTES
        and hashlib.sha256(raw_incident_bytes).hexdigest()
        == V5_PRIMARY_INCIDENT_CANONICAL_SHA256
        and len(correction_bytes) == V5_CORRECTION_SEMANTIC_CANONICAL_BYTES
        and hashlib.sha256(correction_bytes).hexdigest()
        == V5_CORRECTION_SEMANTIC_CANONICAL_SHA256,
        "V4 review truncation incident/correction raw semantic digest mismatch",
    )
    _require(
        set(correction)
        == {
            "artifact_type", "authority_scope", "correction", "created_utc",
            "erroneous_incident", "independent_validation", "prohibitions",
            "required_v5_binding", "schema_version", "status",
        }
        and correction.get("schema_version") == 1
        and correction.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_REVIEW_TRANSPORT_TRUNCATION_INVENTORY_DIGEST_CORRECTION_V1"
        and correction.get("status")
        == "SEALED_V4_REVIEW_TRANSPORT_TRUNCATION_INVENTORY_DIGEST_CORRECTION_NO_GO_NO_LIVE_AUDIT_V5_SUPERSESSION_REQUIRED",
        "V4 review truncation correction must validate before primary semantic use",
    )
    correction_payload = correction.get("correction")
    erroneous = correction.get("erroneous_incident")
    _require(
        isinstance(correction_payload, Mapping)
        and set(correction_payload)
        == {
            "actual_review", "authoritative_canonical_bytes", "authoritative_sha256",
            "canonicalization", "corrected_json_pointer", "erroneous_sha256",
            "parsed_inventory_count", "source_json_pointer",
            "this_is_the_only_authoritative_override",
        }
        and correction_payload.get("actual_review") == FROZEN_V4_REVIEW_IDENTITY
        and correction_payload.get("authoritative_canonical_bytes")
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_BYTES
        and correction_payload.get("authoritative_sha256")
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
        and correction_payload.get("corrected_json_pointer")
        == "/mispublication/actual_review_progress_inventory_canonical_sha256"
        and correction_payload.get("erroneous_sha256")
        == V5_ERRONEOUS_PROGRESS_INVENTORY_SHA256
        and correction_payload.get("parsed_inventory_count") == 101
        and correction_payload.get("source_json_pointer")
        == "/recovered_afterstate/recovery_progress/inventory"
        and correction_payload.get("this_is_the_only_authoritative_override") is True
        and isinstance(erroneous, Mapping)
        and erroneous.get("path") == V5_TRANSPORT_INCIDENT_RELATIVE
        and erroneous.get("size_bytes") == V5_TRANSPORT_INCIDENT_SIZE_BYTES
        and erroneous.get("sha256") == V5_TRANSPORT_INCIDENT_SHA256
        and erroneous.get("status")
        == "SEALED_V4_REVIEW_TRANSPORT_TRUNCATION_NO_GO_NO_LIVE_AUDIT_V5_SUPERSESSION_REQUIRED",
        "V4 review truncation exact correction override mismatch",
    )
    required_binding = correction.get("required_v5_binding")
    _require(
        isinstance(required_binding, Mapping)
        and required_binding.get(
            "correction_must_be_applied_before_any_v5_semantic_use_of_erroneous_incident"
        )
        is True
        and required_binding.get("erroneous_incident_alone_must_fail") is True
        and required_binding.get("full_progress_inventory_must_not_be_duplicated_in_v5_controls")
        is True
        and required_binding.get("invalid_v4_review_must_remain_rejected_history") is True
        and required_binding.get("must_bind_correction_record") is True
        and required_binding.get("must_bind_erroneous_incident") is True
        and required_binding.get("status")
        == "REQUIRED_NOT_YET_AUTHORIZED_FOR_PUBLICATION_OR_EXECUTION",
        "V4 review truncation correction V5 binding mismatch before primary use",
    )
    raw_mispublication = raw_incident.get("mispublication")
    _require(
        isinstance(raw_mispublication, Mapping)
        and raw_mispublication.get(
            "actual_review_progress_inventory_canonical_sha256"
        )
        == V5_ERRONEOUS_PROGRESS_INVENTORY_SHA256,
        "V4 review truncation primary does not contain the one declared bad pointer",
    )
    corrected = json.loads(json.dumps(raw_incident, ensure_ascii=False))
    corrected["mispublication"][
        "actual_review_progress_inventory_canonical_sha256"
    ] = FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
    _require(
        canonical_payload_sha256(corrected)
        == V5_CORRECTED_PRIMARY_INCIDENT_CANONICAL_SHA256,
        "corrected V4 review truncation incident semantic digest mismatch",
    )
    incident = corrected
    _require(
        set(incident)
        == {
            "artifact_type", "authority_scope", "created_utc", "failure_boundary",
            "immutable_v4_chain", "mispublication", "postfailure_state",
            "prohibitions", "required_v5_supersession", "root_cause",
            "schema_version", "status",
        },
        "V4 review truncation incident schema mismatch",
    )
    incident_created = _v4_created_utc(
        incident.get("created_utc"), label="V4 review truncation incident"
    )
    correction_created = _v4_created_utc(
        correction.get("created_utc"), label="V4 review truncation correction"
    )
    _require(
        correction_created > incident_created
        and incident.get("schema_version") == 1
        and incident.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_REVIEW_TRANSPORT_TRUNCATION_INCIDENT_V1"
        and incident.get("status")
        == "SEALED_V4_REVIEW_TRANSPORT_TRUNCATION_NO_GO_NO_LIVE_AUDIT_V5_SUPERSESSION_REQUIRED"
        and correction.get("schema_version") == 1
        and correction.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_REVIEW_TRANSPORT_TRUNCATION_INVENTORY_DIGEST_CORRECTION_V1"
        and correction.get("status")
        == "SEALED_V4_REVIEW_TRANSPORT_TRUNCATION_INVENTORY_DIGEST_CORRECTION_NO_GO_NO_LIVE_AUDIT_V5_SUPERSESSION_REQUIRED",
        "V4 review truncation incident/correction header mismatch",
    )
    _require(
        incident.get("authority_scope")
        == {
            "documentary_only": True,
            "incident_write_authorized": True,
            "live_postrun_auditor_authorized": False,
            "network_authorized": False,
            "recovery_rerun_authorized": False,
            "v4_go_publication_authorized": False,
            "v5_code_test_or_control_publication_authorized_by_this_incident": False,
        }
        and correction.get("authority_scope")
        == {
            "correction_record_write_authorized": True,
            "documentary_only": True,
            "live_postrun_auditor_authorized": False,
            "network_authorized": False,
            "recovery_rerun_authorized": False,
            "v4_go_publication_authorized": False,
            "v5_code_test_or_control_publication_authorized_by_this_record": False,
        },
        "V4 review truncation incident/correction authority scope mismatch",
    )
    _require(
        incident.get("prohibitions")
        == {
            "do_not_delete_move_rename_copy_hardlink_or_mutate_invalid_v4_review": True,
            "do_not_execute_v4_live_auditor": True,
            "do_not_fit_models_or_create_submission_csv": True,
            "do_not_publish_v4_go": True,
            "do_not_read_labels_or_2024_2025_arrays": True,
            "do_not_rerun_recovery": True,
            "do_not_use_network": True,
            "preserve_v1_v2_v3_v4_artifacts_append_only": True,
        }
        and correction.get("prohibitions")
        == {
            "do_not_delete_move_rename_copy_hardlink_or_mutate_erroneous_incident": True,
            "do_not_delete_move_rename_copy_hardlink_or_mutate_invalid_v4_review": True,
            "do_not_execute_v4_live_auditor": True,
            "do_not_fit_models_or_create_submission_csv": True,
            "do_not_publish_v4_go": True,
            "do_not_read_labels_or_2024_2025_arrays": True,
            "do_not_rerun_recovery": True,
            "do_not_use_network": True,
        },
        "V4 review truncation incident/correction prohibitions mismatch",
    )
    independent_validation = correction.get("independent_validation")
    _require(
        independent_validation
        == {
            "all_other_checked_incident_facts_unchanged": True,
            "recovery_redteam_recomputed_exact": True,
            "root_recomputed_exact": True,
            "secondary_reviewer_recomputed_exact": True,
            "three_way_authoritative_bytes": 20_605,
            "three_way_authoritative_sha256": (
                FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
            ),
        },
        "V4 review truncation correction independent validation mismatch",
    )
    incident_bytes = _v4_canonical_bytes(incident)
    _require(
        len(incident_bytes) == V5_PRIMARY_INCIDENT_CANONICAL_BYTES
        and hashlib.sha256(incident_bytes).hexdigest()
        == V5_CORRECTED_PRIMARY_INCIDENT_CANONICAL_SHA256,
        "V4 review truncation incident/correction semantic digest mismatch",
    )
    chain = incident.get("immutable_v4_chain")
    _require(
        isinstance(chain, Mapping)
        and set(chain)
        == {
            "audit_attempt_id", "authorization", "authorization_status",
            "prior_false_reject_incident", "prior_go_path_incident", "v4_auditor",
            "v4_auditor_test", "v4_sealer", "v4_sealer_test",
        }
        and chain.get("audit_attempt_id") == FROZEN_V4_AUDIT_ATTEMPT_ID
        and chain.get("authorization") == FROZEN_V4_AUTH_IDENTITY
        and chain.get("authorization_status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4"
        and chain.get("prior_false_reject_incident")
        == {
            "path": V2_FALSE_REJECT_INCIDENT_RELATIVE,
            "size_bytes": V2_FALSE_REJECT_INCIDENT_SIZE_BYTES,
            "sha256": V2_FALSE_REJECT_INCIDENT_SHA256,
        }
        and chain.get("prior_go_path_incident")
        == {
            "path": V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
            "size_bytes": V4_PATH_MISPUBLISH_INCIDENT_SIZE_BYTES,
            "sha256": V4_PATH_MISPUBLISH_INCIDENT_SHA256,
        }
        and all(
            isinstance(chain.get(role.removeprefix("superseded_")), Mapping)
            and set(chain[role.removeprefix("superseded_")])
            == {"path", "size_bytes", "sha256"}
            and int(chain[role.removeprefix("superseded_")]["size_bytes"])
            == int(expected["size_bytes"])
            and chain[role.removeprefix("superseded_")]["sha256"]
            == expected["sha256"]
            and Path(str(chain[role.removeprefix("superseded_")]["path"])).resolve()
            == Path(str(expected["path"])).resolve()
            for role, expected in FROZEN_V4_SOURCE_IDENTITIES.items()
        ),
        "V4 review truncation incident immutable-chain mismatch",
    )
    mispublication = incident.get("mispublication")
    missing_paths = [
        f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005500.json",
        f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005600.json",
        f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005700.json",
    ]
    _require(
        isinstance(mispublication, Mapping)
        and mispublication.get("actual_rejected_review") == FROZEN_V4_REVIEW_IDENTITY
        and mispublication.get("actual_review_afterstate_canonical_bytes")
        == FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_BYTES
        and mispublication.get("actual_review_afterstate_canonical_sha256")
        == FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_SHA256
        and mispublication.get("actual_review_progress_inventory_count") == 101
        and mispublication.get("actual_review_progress_inventory_canonical_sha256")
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
        and mispublication.get("missing_progress_paths") == missing_paths
        and mispublication.get("required_afterstate_canonical_bytes")
        == V4_RECOVERED_AFTERSTATE_CANONICAL_BYTES
        and mispublication.get("required_afterstate_canonical_sha256")
        == V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        and mispublication.get("required_progress_inventory_count") == 104
        and mispublication.get("required_progress_inventory_canonical_sha256")
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256
        and mispublication.get("intended_review")
        == {"path": V4_REVIEW_RELATIVE, **FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY}
        and mispublication.get("transport_marker")
        == {
            "count": 1,
            "line_number_one_based": 388,
            "literal": "\u2026169 tokens truncated\u2026",
            "unicode_ellipsis_codepoint": "U+2026",
        },
        "V4 review truncation incident mispublication facts mismatch",
    )
    duplicate = mispublication.get("duplicate_key_semantics")
    _require(
        mispublication.get("duplicate_key_object_path")
        == f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005400.json"
        and duplicate
        == {
            "actual_last_wins_sha256": "c4f678c57b239960b802157f064d66333ebfa78c9122c1a1943aa517ff0024c1",
            "expected_sha256": "47e6b80980b9377c754efc7574e7f203dde9e59bf52fbeaf5b96a05ad9c1bce6",
            "key": "sha256",
            "raw_duplicate_count": 2,
        },
        "V4 review duplicate-key incident facts mismatch",
    )
    _require(
        canonical_payload_sha256(incident.get("postfailure_state"))
        == V5_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
        "V4 review truncation postfailure-state digest mismatch",
    )
    failure = incident.get("failure_boundary")
    root_cause = incident.get("root_cause")
    required_v5 = incident.get("required_v5_supersession")
    _require(
        isinstance(failure, Mapping)
        and failure.get("canonical_v4_go_present") is False
        and failure.get("data_or_parquet_value_read_started") is False
        and failure.get("full_offline_replay_started") is False
        and failure.get("live_postrun_auditor_started") is False
        and failure.get("network_requests_observed") == 0
        and failure.get("postrun_audit_output_files_written") == 0
        and isinstance(root_cause, Mapping)
        and root_cause.get("category")
        == "PATCH_TRANSPORT_TRUNCATION_OF_LARGE_REVIEW_PAYLOAD"
        and root_cause.get("invalid_review_can_authorize_v4_go") is False
        and root_cause.get("review_json_parseable_but_semantically_invalid") is True
        and root_cause.get("transport_inserted_literal_summary_marker") is True,
        "V4 review truncation no-GO/no-live boundary mismatch",
    )
    _require(
        isinstance(required_v5, Mapping)
        and required_v5.get("artifact_paths_root_relative")
        == {
            "authorization": V5_AUTH_RELATIVE,
            "independent_go": V5_GO_RELATIVE,
            "independent_review": V5_REVIEW_RELATIVE,
        }
        and required_v5.get("code_paths_repo_relative")
        == {
            "auditor": "scripts/audit_noaa_gfs_multiseason_raw_postrun_v5.py",
            "auditor_test": "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py",
            "authorization_sealer": "scripts/seal_noaa_gfs_multiseason_postrun_audit_v5.py",
            "authorization_sealer_test": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py",
        }
        and required_v5.get("compact_controls_required") is True
        and required_v5.get("full_recovery_progress_inventory_must_exist_only_in_immutable_v4_authorization")
        is True
        and required_v5.get("invalid_v4_review_must_be_bound_as_rejected_history")
        is True
        and required_v5.get("maximum_review_or_go_size_bytes") == 12_000
        and required_v5.get("v4_data_and_replay_logic_must_remain_ast_identical")
        is True
        and required_v5.get("status")
        == "REQUIRED_NOT_YET_AUTHORIZED_FOR_PUBLICATION_OR_EXECUTION",
        "V4 review truncation required V5 supersession mismatch",
    )
    _require(
        isinstance(correction_payload, Mapping)
        and set(correction_payload)
        == {
            "actual_review", "authoritative_canonical_bytes", "authoritative_sha256",
            "canonicalization", "corrected_json_pointer", "erroneous_sha256",
            "parsed_inventory_count", "source_json_pointer",
            "this_is_the_only_authoritative_override",
        }
        and correction_payload.get("actual_review") == FROZEN_V4_REVIEW_IDENTITY
        and correction_payload.get("authoritative_canonical_bytes")
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_BYTES
        and correction_payload.get("authoritative_sha256")
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
        and correction_payload.get("corrected_json_pointer")
        == "/mispublication/actual_review_progress_inventory_canonical_sha256"
        and correction_payload.get("erroneous_sha256")
        == V5_ERRONEOUS_PROGRESS_INVENTORY_SHA256
        and correction_payload.get("parsed_inventory_count") == 101
        and correction_payload.get("source_json_pointer")
        == "/recovered_afterstate/recovery_progress/inventory"
        and correction_payload.get("this_is_the_only_authoritative_override") is True,
        "V4 review truncation correction pointer/value mismatch",
    )
    _require(
        isinstance(erroneous, Mapping)
        and set(erroneous)
        == {
            "artifact_type", "erroneous_json_pointer", "path", "sha256",
            "size_bytes", "status",
        }
        and erroneous.get("path") == V5_TRANSPORT_INCIDENT_RELATIVE
        and erroneous.get("size_bytes") == V5_TRANSPORT_INCIDENT_SIZE_BYTES
        and erroneous.get("sha256") == V5_TRANSPORT_INCIDENT_SHA256
        and erroneous.get("status") == incident.get("status")
        and erroneous.get("erroneous_json_pointer")
        == correction_payload.get("corrected_json_pointer"),
        "V4 review truncation correction primary binding mismatch",
    )
    _require(
        isinstance(required_binding, Mapping)
        and required_binding.get(
            "correction_must_be_applied_before_any_v5_semantic_use_of_erroneous_incident"
        )
        is True
        and required_binding.get("erroneous_incident_alone_must_fail") is True
        and required_binding.get("full_progress_inventory_must_not_be_duplicated_in_v5_controls")
        is True
        and required_binding.get("invalid_v4_review_must_remain_rejected_history") is True
        and required_binding.get("must_bind_correction_record") is True
        and required_binding.get("must_bind_erroneous_incident") is True
        and required_binding.get("status")
        == "REQUIRED_NOT_YET_AUTHORIZED_FOR_PUBLICATION_OR_EXECUTION",
        "V4 review truncation correction V5 binding mismatch",
    )
    return {
        "incident": raw_incident,
        "correction": correction,
        "corrected_incident": corrected,
        "incident_identity": dict(incident_expected),
        "correction_identity": dict(correction_expected),
        "exact_one_override_applied": True,
        "incident_created": incident_created,
        "correction_created": correction_created,
    }


def _v5_recovered_afterstate_commitment(
    afterstate: Mapping[str, Any],
) -> dict[str, Any]:
    encoded = _v4_canonical_bytes(afterstate)
    progress = afterstate.get("recovery_progress")
    inventory = progress.get("inventory") if isinstance(progress, Mapping) else None
    _require(
        len(encoded) == V4_RECOVERED_AFTERSTATE_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        and isinstance(inventory, list)
        and len(inventory) == 104,
        "V5 full recovered afterstate cannot satisfy its compact commitment",
    )
    commitment = {
        "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
        "canonical_size_bytes": len(encoded),
        "recovery_progress_file_count": int(progress["file_count"]),
        "recovery_progress_inventory_count": len(inventory),
        "recovery_progress_inventory_canonical_sha256": str(
            progress["inventory_canonical_sha256"]
        ),
        "canonical_output_count": int(afterstate["canonical_output_count"]),
        "transaction_total_recursive_files": int(
            afterstate["transaction"]["total_recursive_files"]
        ),
        "recovery_history_file_count": int(
            afterstate["recovery_history_file_count"]
        ),
        "raw_active_lock_present": afterstate["raw_active_lock_present"],
        "decoded_active_lock_present": afterstate["decoded_active_lock_present"],
    }
    commitment_bytes = _v4_canonical_bytes(commitment)
    _require(
        commitment == V5_RECOVERED_AFTERSTATE_COMMITMENT
        and len(commitment_bytes)
        == V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_BYTES
        and hashlib.sha256(commitment_bytes).hexdigest()
        == V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256,
        "V5 compact recovered-afterstate commitment mismatch",
    )
    return commitment


def _v5_frozen_v4_required_command() -> list[str]:
    return _v4_required_command(
        ROOT_DEFAULT,
        path_under(ROOT_DEFAULT, V4_AUTH_RELATIVE),
        path_under(ROOT_DEFAULT, V4_GO_RELATIVE),
    )


def _v5_load_and_validate_v4_authorization_base(
    root: Path, declared_base: Any
) -> dict[str, Any]:
    _require(
        isinstance(declared_base, Mapping)
        and set(declared_base) == V5_AUTHORIZATION_BASE_KEYS,
        "V5 V4-authorization base summary schema mismatch",
    )
    actual_identity = _v4_verify_exact_root_identity(
        root,
        declared_base.get("identity"),
        FROZEN_V4_AUTH_IDENTITY,
        label="frozen V4 authorization base",
    )
    authorization = load_json(path_under(root, V4_AUTH_RELATIVE))
    encoded = _v4_canonical_bytes(authorization)
    _require(
        set(authorization) == V4_AUTH_KEYS
        and len(encoded) == FROZEN_V4_AUTH_PAYLOAD_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == FROZEN_V4_AUTH_PAYLOAD_CANONICAL_SHA256,
        "frozen V4 authorization schema/canonical digest mismatch",
    )
    authorization_created = _v4_created_utc(
        authorization.get("created_utc"), label="frozen V4 authorization"
    )
    _require(
        authorization.get("schema_version") == 4
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V4"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4"
        and authorization.get("audit_attempt_id") == FROZEN_V4_AUDIT_ATTEMPT_ID
        and authorization.get("recovery_attempt_id") == V4_RECOVERY_ATTEMPT_ID
        and authorization.get("runtime_identity_sha256")
        == V4_RUNTIME_IDENTITY_SHA256
        and authorization.get("independent_review_required") is True
        and authorization.get("independent_go_required") is True
        and authorization.get("required_command")
        == _v5_frozen_v4_required_command(),
        "frozen V4 authorization header/command mismatch",
    )
    _v4_validate_policy(authorization, label="frozen V4 authorization")
    _v4_validate_path_mispublish_incident(root, authorization.get("incident"))
    _v4_validate_false_reject_incident(
        root, authorization.get("v2_false_reject_incident")
    )
    frozen_v3 = _v4_validate_frozen_v3_chain(root)
    _require(
        authorization.get("v3_path_mispublish_state") == frozen_v3["state"],
        "frozen V4 authorization V3 inherited state mismatch",
    )
    for field, expected, path in (
        (
            "superseded_v2_auditor",
            V4_SUPERSEDED_V2_AUDITOR_IDENTITY,
            SUPERSEDED_V2_AUDITOR,
        ),
        (
            "superseded_v2_auditor_test",
            V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY,
            SUPERSEDED_V2_AUDITOR_TEST,
        ),
    ):
        _v4_require_exact_declared_identity(
            authorization.get(field), expected, label=f"frozen V4 {field}"
        )
        _v4_verify_external_identity(
            authorization[field],
            path,
            label=f"frozen V4 {field}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    for role, expected in FROZEN_V3_SOURCE_IDENTITIES.items():
        _v4_require_exact_declared_identity(
            authorization.get(role), expected, label=f"frozen V4 {role}"
        )
        _require(
            frozen_v3["source_identities"][role] == expected,
            f"frozen V4 inherited V3 source mismatch: {role}",
        )
    v4_field_paths = {
        "v4_auditor": FROZEN_V4_AUDITOR,
        "v4_auditor_test": FROZEN_V4_AUDITOR_TEST,
        "v4_sealer": FROZEN_V4_SEALER,
        "v4_sealer_test": FROZEN_V4_SEALER_TEST,
    }
    v4_sources: dict[str, dict[str, Any]] = {}
    for field, path in v4_field_paths.items():
        frozen_role = f"superseded_{field}"
        expected = FROZEN_V4_SOURCE_IDENTITIES[frozen_role]
        _v4_require_exact_declared_identity(
            authorization.get(field), expected, label=f"frozen {field}"
        )
        v4_sources[frozen_role] = _v4_verify_external_identity(
            authorization[field],
            path,
            label=f"frozen {field}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    controls = authorization.get("v2_recovery_controls")
    _require(
        isinstance(controls, Mapping)
        and set(controls) == set(V4_V2_CONTROL_IDENTITIES),
        "frozen V4 V2-control schema mismatch",
    )
    for role, expected in V4_V2_CONTROL_IDENTITIES.items():
        _v4_verify_exact_root_identity(
            root, controls[role], expected, label=f"frozen V4 V2 control {role}"
        )
    support = authorization.get("v2_recovery_support")
    _require(
        isinstance(support, Mapping) and set(support) == set(V4_V2_SUPPORT_PATHS),
        "frozen V4 V2-support schema mismatch",
    )
    for role, expected_path in V4_V2_SUPPORT_PATHS.items():
        _v4_verify_external_identity(
            support[role],
            expected_path,
            label=f"frozen V4 V2 support {role}",
            expected_size_sha256=V4_V2_SUPPORT_SIZE_SHA256[role],
        )
    original = authorization.get("original_v1_provenance")
    _require(
        isinstance(original, Mapping)
        and dict(original) == V4_ORIGINAL_V1_PROVENANCE,
        "frozen V4 original V1 provenance mismatch",
    )
    raw_record = original["original_raw_authorization_v1"]
    prelaunch_record = original["original_independent_prelaunch_audit_v1"]
    _v4_verify_exact_root_identity(
        root,
        raw_record,
        V4_ORIGINAL_V1_PROVENANCE["original_raw_authorization_v1"],
        label="frozen V4 original raw authorization",
    )
    _v4_require_exact_declared_identity(
        prelaunch_record,
        V4_ORIGINAL_V1_PROVENANCE["original_independent_prelaunch_audit_v1"],
        label="frozen V4 original independent prelaunch audit",
    )
    verify_identity(
        prelaunch_record,
        path_under(root, str(prelaunch_record["path"])),
        root=root,
        label="frozen V4 original independent prelaunch audit",
        allowed_extra_fields=("status",),
    )
    raw_authorization = load_json(path_under(root, str(raw_record["path"])))
    runtime_identity = raw_authorization.get("runtime_identity")
    runtime_bytes = _v4_canonical_bytes(runtime_identity)
    _require(
        raw_authorization.get("independent_prelaunch_audit") == prelaunch_record
        and isinstance(runtime_identity, Mapping)
        and len(runtime_bytes) == 1_405
        and hashlib.sha256(runtime_bytes).hexdigest() == V4_RUNTIME_IDENTITY_SHA256
        and raw_authorization.get("runtime_identity_sha256")
        == V4_RUNTIME_IDENTITY_SHA256,
        "frozen V4 original V1/runtime transitive binding mismatch",
    )
    afterstate = _v4_recovered_afterstate_from_authorization(authorization)
    _require(
        afterstate == frozen_v3["afterstate"],
        "frozen V4/V3 full recovered afterstate mismatch",
    )
    _v4_validate_zero_snapshot(
        authorization.get("preaudit_zero_mutation_snapshot"),
        incident_postfailure_sha256=V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
    )
    v4_evidence_bound = {
        "v2_auditor": authorization["superseded_v2_auditor"],
        "v2_auditor_test": authorization["superseded_v2_auditor_test"],
        **{role: authorization[role] for role in FROZEN_V3_SOURCE_IDENTITIES},
        **{
            field: authorization[field]
            for field in ("v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test")
        },
    }
    _v4_validate_test_evidence(
        authorization.get("test_evidence"),
        authorization_created=authorization_created,
        bound_identities=v4_evidence_bound,
    )
    test_evidence_sha = canonical_payload_sha256(authorization["test_evidence"])
    base = {
        "identity": actual_identity,
        "schema_version": 4,
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4",
        "audit_attempt_id": FROZEN_V4_AUDIT_ATTEMPT_ID,
        "recovery_attempt_id": V4_RECOVERY_ATTEMPT_ID,
        "payload_canonical_sha256": FROZEN_V4_AUTH_PAYLOAD_CANONICAL_SHA256,
        "test_evidence_canonical_sha256": test_evidence_sha,
        "recovered_afterstate_canonical_sha256": (
            V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
    }
    _require(
        test_evidence_sha == FROZEN_V4_TEST_EVIDENCE_CANONICAL_SHA256
        and dict(declared_base) == base,
        "V5 declared V4 authorization base summary mismatch",
    )
    return {
        "authorization": authorization,
        "authorization_created": authorization_created,
        "identity": actual_identity,
        "base": base,
        "afterstate": afterstate,
        "commitment": _v5_recovered_afterstate_commitment(afterstate),
        "v4_source_identities": v4_sources,
        "frozen_v3": frozen_v3,
    }


def _v5_load_rejected_v4_review(
    root: Path, v4_base: Mapping[str, Any]
) -> dict[str, Any]:
    review_path = path_under(root, V4_REVIEW_RELATIVE)
    actual_identity = _v4_verify_exact_root_identity(
        root,
        FROZEN_V4_REVIEW_IDENTITY,
        FROZEN_V4_REVIEW_IDENTITY,
        label="rejected malformed V4 independent review",
    )
    raw = review_path.read_text(encoding="utf-8")
    marker = "\u2026169 tokens truncated\u2026"
    lines = raw.splitlines()
    _require(
        raw.count(marker) == 1
        and len(lines) >= 388
        and marker in lines[387],
        "rejected V4 review transport marker mismatch",
    )
    duplicates: list[tuple[str, Any, Any]] = []

    def _last_wins_with_duplicate_recording(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append((key, result[key], value))
            result[key] = value
        return result

    try:
        review = json.loads(raw, object_pairs_hook=_last_wins_with_duplicate_recording)
    except Exception as exc:
        raise AuditFailure(f"rejected V4 review is no longer parseable: {exc}") from exc
    _require(
        isinstance(review, Mapping)
        and set(review) == V4_REVIEW_KEYS
        and len(duplicates) == 1
        and duplicates[0][0] == "sha256"
        and duplicates[0][1]
        == (
            "47e6b80980b9377c754efc7574e7f203dde9e59bf52f"
            "\u2026169 tokens truncated\u2026"
            f"es__decode_recovery_v2__20260810T211017500046Z__005700.json"
        )
        and duplicates[0][2]
        == "c4f678c57b239960b802157f064d66333ebfa78c9122c1a1943aa517ff0024c1",
        "rejected V4 review duplicate-key evidence mismatch",
    )
    encoded = _v4_canonical_bytes(review)
    afterstate = review.get("recovered_afterstate")
    afterstate_bytes = _v4_canonical_bytes(afterstate)
    progress = afterstate.get("recovery_progress") if isinstance(afterstate, Mapping) else None
    inventory = progress.get("inventory") if isinstance(progress, Mapping) else None
    inventory_bytes = _v4_canonical_bytes(inventory)
    _require(
        len(encoded) == FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_SHA256
        and len(afterstate_bytes) == FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_BYTES
        and hashlib.sha256(afterstate_bytes).hexdigest()
        == FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_SHA256
        and isinstance(inventory, list)
        and len(inventory) == 101
        and len(inventory_bytes)
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_BYTES
        and hashlib.sha256(inventory_bytes).hexdigest()
        == FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "rejected V4 review parsed payload/afterstate/inventory mismatch",
    )
    inventory_by_path = {
        str(record.get("path")): record
        for record in inventory
        if isinstance(record, Mapping)
    }
    missing = {
        f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005500.json",
        f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005600.json",
        f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005700.json",
    }
    _require(
        all(path not in inventory_by_path for path in missing)
        and inventory_by_path[
            f"decoded/recovery_progress/decoded_messages__{V4_RECOVERY_ATTEMPT_ID}__005400.json"
        ]["sha256"]
        == "c4f678c57b239960b802157f064d66333ebfa78c9122c1a1943aa517ff0024c1",
        "rejected V4 review missing-progress/effective-sha mismatch",
    )
    authorization = v4_base["authorization"]
    review_created = _v4_created_utc(
        review.get("created_utc"), label="rejected V4 independent review"
    )
    _require(
        review.get("schema_version") == 4
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V4"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V4"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == FROZEN_V4_AUDIT_ATTEMPT_ID
        and review.get("authorization") == FROZEN_V4_AUTH_IDENTITY
        and review.get("incident") == authorization.get("incident")
        and review.get("v2_false_reject_incident")
        == authorization.get("v2_false_reject_incident")
        and review.get("v3_path_mispublish_state")
        == authorization.get("v3_path_mispublish_state")
        and review.get("v2_recovery_controls")
        == authorization.get("v2_recovery_controls")
        and review.get("auditor_execution_started") is False
        and review.get("independent_go_required") is True
        and review_created >= v4_base["authorization_created"],
        "rejected V4 review header/crosslink mismatch",
    )
    _v4_validate_policy(review, label="rejected V4 independent review")
    for field in (
        "superseded_v2_auditor", "superseded_v2_auditor_test",
        *FROZEN_V3_SOURCE_IDENTITIES.keys(),
        "v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test",
    ):
        _require(
            review.get(field) == authorization.get(field),
            f"rejected V4 review source crosslink mismatch: {field}",
        )
    checks = review.get("independent_checks")
    recheck = review.get("test_evidence_recheck")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == V4_REVIEW_CHECKS
        and all(checks.get(key) is True for key in V4_REVIEW_CHECKS)
        and isinstance(recheck, Mapping)
        and set(recheck) == V4_REVIEW_TEST_RECHECK_KEYS
        and recheck.get("authorization_test_evidence_canonical_sha256")
        == FROZEN_V4_TEST_EVIDENCE_CANONICAL_SHA256
        and all(
            recheck.get(key) is True
            for key in V4_REVIEW_TEST_RECHECK_KEYS
            if key != "authorization_test_evidence_canonical_sha256"
        ),
        "rejected V4 review independent checks/evidence mismatch",
    )
    intended = json.loads(json.dumps(review, ensure_ascii=False))
    intended["recovered_afterstate"] = v4_base["afterstate"]
    intended_bytes = _v4_canonical_bytes(intended)
    pretty = (
        json.dumps(intended, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _require(
        len(intended_bytes) == FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_BYTES
        and hashlib.sha256(intended_bytes).hexdigest()
        == FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_SHA256
        and len(pretty) == FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY["size_bytes"]
        and hashlib.sha256(pretty).hexdigest()
        == FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY["sha256"],
        "intended V4 review reconstruction mismatch",
    )
    v4_go_path = path_under(root, V4_GO_RELATIVE)
    require_no_symlink_chain(v4_go_path, root, label="forbidden canonical V4 GO path")
    _require(
        not os.path.lexists(str(v4_go_path)),
        "canonical V4 GO must remain absent before truncation state is declared",
    )
    state = {
        "authorization": v4_base["identity"],
        "independent_review": actual_identity,
        "canonical_go": {"path": V4_GO_RELATIVE, "present": False},
        "actual_review_payload_canonical_sha256": (
            FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_SHA256
        ),
        "intended_review_payload_canonical_sha256": (
            FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_SHA256
        ),
        "actual_recovered_afterstate_canonical_sha256": (
            FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_SHA256
        ),
        "expected_recovered_afterstate_canonical_sha256": (
            V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
        "actual_progress_inventory_count": 101,
        "expected_progress_inventory_count": 104,
        "intended_review_pretty_serialization": dict(
            FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY
        ),
        "v4_auditor_execution_started": False,
        "v4_auditor_execution_authorized": False,
    }
    _require(set(state) == V5_TRUNCATION_STATE_KEYS, "V5 truncation state schema mismatch")
    return {
        "review": review,
        "review_identity": actual_identity,
        "intended_review": intended,
        "state": state,
        "duplicate_count": len(duplicates),
        "transport_marker_count": raw.count(marker),
        "review_created": review_created,
    }


def _v5_validate_transport_truncation_state(
    declared: Any, actual: Mapping[str, Any]
) -> None:
    _require(
        isinstance(declared, Mapping)
        and set(declared) == V5_TRUNCATION_STATE_KEYS
        and dict(declared) == dict(actual),
        "V5 V4-review transport-truncation state mismatch",
    )


def _v5_validate_zero_snapshot(root: Path, snapshot: Any) -> None:
    _require(
        isinstance(snapshot, Mapping) and set(snapshot) == V5_ZERO_SNAPSHOT_KEYS,
        "V5 preaudit zero-mutation snapshot schema mismatch",
    )
    _require(
        snapshot.get("incident_postfailure_state_canonical_sha256")
        == V5_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256
        and snapshot.get("recovered_afterstate_commitment_canonical_sha256")
        == V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        and snapshot.get("active_locks")
        == {
            "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
            "decoded": {
                "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
                "present": False,
            },
        }
        and snapshot.get("v5_control_temporary_files")
        == {
            "checked_destination_relative_paths": list(
                _v5_control_destination_relatives()
            ),
            "matching_temporary_files": [],
        }
        and snapshot.get("postrun_audit_output_files_present") == 0
        and snapshot.get("network_requests") == 0
        and snapshot.get("audit_files_written") == 0
        and snapshot.get("labels_read") is False
        and snapshot.get("arrays_2024_read") is False
        and snapshot.get("arrays_2025_read") is False
        and snapshot.get("models_fit") == 0
        and snapshot.get("submission_csv_created") is False,
        "V5 preaudit zero-mutation snapshot values mismatch",
    )
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(
            str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock"))
        )
        and _v5_control_temporary_files(root) == []
        and _v5_postrun_report_candidates(root) == [],
        "V5 preaudit zero-mutation snapshot does not match current filesystem",
    )


def _v5_validate_test_evidence(
    evidence: Any,
    *,
    authorization_created: datetime,
    bound_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(
        isinstance(evidence, Mapping) and set(evidence) == V5_TEST_EVIDENCE_KEYS,
        "V5 immutable test-evidence schema mismatch",
    )
    evidence_created = _v4_created_utc(
        evidence.get("created_utc"), label="V5 test evidence"
    )
    _require(
        evidence_created <= authorization_created
        and evidence.get("schema_version") == 1
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_TEST_EVIDENCE"
        and evidence.get("status")
        == "PASS_FROZEN_V5_AUDITOR_AND_SEALER_TESTS"
        and evidence.get("pytest_isolation")
        == "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS"
        and evidence.get("required_test_names") == list(V5_REQUIRED_TEST_NAMES)
        and evidence.get("required_test_names_present") is True
        and evidence.get("all_exit_codes_zero") is True
        and evidence.get("bound_identities") == bound_identities,
        "V5 immutable test-evidence header/binding mismatch",
    )
    source_compile = evidence.get("source_compile")
    _require(
        isinstance(source_compile, Mapping)
        and set(source_compile) == {"method", "result", "files"}
        and source_compile.get("method") == "compile_exact_source_no_pyc"
        and source_compile.get("result") == "PASS"
        and source_compile.get("files")
        == [
            bound_identities["v5_auditor"],
            bound_identities["v5_auditor_test"],
            bound_identities["v5_sealer"],
            bound_identities["v5_sealer_test"],
        ],
        "V5 exact-source compile evidence mismatch",
    )
    expected_commands = {
        "v5_auditor_tests": _v5_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py"
        ),
        "v5_sealer_tests": _v5_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py"
        ),
        "frozen_v4_auditor_tests": _v5_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py"
        ),
        "frozen_v4_sealer_tests": _v5_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py"
        ),
    }
    runs = evidence.get("test_runs")
    _require(
        isinstance(runs, Mapping) and set(runs) == V5_TEST_RUN_KEYS,
        "V5 immutable pytest run map mismatch",
    )
    for role, expected_command in expected_commands.items():
        run = runs[role]
        _require(
            isinstance(run, Mapping)
            and set(run) == V4_TEST_RUN_RECORD_KEYS
            and run.get("command") == expected_command
            and run.get("exit_code") == 0
            and isinstance(run.get("summary"), str)
            and "passed" in str(run["summary"])
            and "failed" not in str(run["summary"]).casefold()
            and "error" not in str(run["summary"]).casefold()
            and isinstance(run.get("stdout_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stdout_sha256"])) is not None
            and isinstance(run.get("stderr_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stderr_sha256"])) is not None
            and run.get("network_guard_installed") is True
            and run.get("cacheprovider_disabled") is True,
            f"V5 immutable pytest run mismatch: {role}",
        )
    _require(
        evidence.get("production_shape_regression")
        == {
            "v4_authorization_base_validated": True,
            "v4_transport_truncation_incident_validated": True,
            "compact_afterstate_commitment_validated": True,
            "full_afterstate_reconstructed_in_memory": True,
            "original_v1_four_key_identity_passed": True,
            "transaction_reached": True,
            "raw_reached": True,
            "decoded_reached": True,
            "offline_replay_reached": True,
            "synthetic_fixture": True,
        }
        and evidence.get("real_seven_spawn_regression")
        == {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        }
        and evidence.get("stdout_only_regression") is True
        and evidence.get("network_guard")
        == {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        }
        and evidence.get("network_requests") == 0
        and evidence.get("audit_files_written") == 0
        and evidence.get("labels_read") is False
        and evidence.get("arrays_2024_read") is False
        and evidence.get("arrays_2025_read") is False
        and evidence.get("models_fit") == 0
        and evidence.get("submission_csv_created") is False,
        "V5 immutable test-evidence safety/regression mismatch",
    )


def _v5_assert_compact_control(payload: Mapping[str, Any], *, label: str) -> None:
    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            _require(
                "inventory" not in value
                and "recovered_afterstate" not in value
                and "recovery_progress" not in value,
                f"{label} embeds a forbidden full-afterstate field",
            )
            for nested in value.values():
                _walk(nested)
        elif isinstance(value, list):
            _require(
                len(value) != 104,
                f"{label} embeds a forbidden full 104-entry inventory",
            )
            for nested in value:
                _walk(nested)

    _walk(payload)


def _v5_validate_predata_authority(
    root: Path, authorization_path: Path, go_path: Path
) -> dict[str, Any]:
    """Validate the complete compact V5 chain before any output/data dereference."""

    expected_authorization_path = path_under(root, V5_AUTH_RELATIVE)
    expected_review_path = path_under(root, V5_REVIEW_RELATIVE)
    expected_go_path = path_under(root, V5_GO_RELATIVE)
    _require(
        Path(os.path.abspath(authorization_path)) == expected_authorization_path
        and Path(os.path.abspath(go_path)) == expected_go_path,
        "V5 authorization/GO paths are not canonical",
    )
    for path, label in (
        (expected_authorization_path, "V5 authorization"),
        (expected_review_path, "V5 independent review"),
        (expected_go_path, "V5 independent GO"),
    ):
        require_no_symlink_chain(path, root, label=label)
        _require(
            os.path.lexists(str(path)) and path.is_file() and not _linklike(path),
            f"{label} is absent, non-regular or link-like",
        )
    _require(
        expected_review_path.stat().st_size <= 12_000
        and expected_go_path.stat().st_size <= 12_000,
        "V5 compact review or GO exceeds the 12KB transport limit",
    )
    namespace_start = _v5_validate_control_namespace(root)
    _require(_v5_control_temporary_files(root) == [], "V5 control temporary files present")
    _require(
        _v5_postrun_report_candidates(root) == [],
        "persisted V2/V3/V4/V5 postrun report candidate is forbidden",
    )
    authorization = _v5_load_strict_json(
        expected_authorization_path, label="V5 authorization"
    )
    review = _v5_load_strict_json(
        expected_review_path, label="V5 independent review"
    )
    go = _v5_load_strict_json(expected_go_path, label="V5 independent GO")
    _require(set(authorization) == V5_AUTH_KEYS, "V5 authorization schema mismatch")
    _require(set(review) == V5_REVIEW_KEYS, "V5 independent review schema mismatch")
    _require(set(go) == V5_GO_KEYS, "V5 independent GO schema mismatch")
    for payload, label in (
        (authorization, "V5 authorization"),
        (review, "V5 independent review"),
        (go, "V5 independent GO"),
    ):
        _v5_assert_compact_control(payload, label=label)
    authorization_created = _v4_created_utc(
        authorization.get("created_utc"), label="V5 authorization"
    )
    review_created = _v4_created_utc(
        review.get("created_utc"), label="V5 independent review"
    )
    go_created = _v4_created_utc(go.get("created_utc"), label="V5 independent GO")
    attempt_id = authorization.get("audit_attempt_id")
    derived_attempt = "postrun_audit_v5__" + str(authorization["created_utc"]).translate(
        str.maketrans("", "", "-:.")
    )
    _require(
        authorization.get("schema_version") == 5
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V5"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5"
        and isinstance(attempt_id, str)
        and re.fullmatch(r"postrun_audit_v5__[0-9]{8}T[0-9]{12}Z", attempt_id)
        is not None
        and attempt_id == derived_attempt
        and authorization.get("recovery_attempt_id") == V4_RECOVERY_ATTEMPT_ID
        and authorization.get("runtime_identity_sha256")
        == V4_RUNTIME_IDENTITY_SHA256
        and authorization.get("independent_review_required") is True
        and authorization.get("independent_go_required") is True,
        "V5 authorization header/binding mismatch",
    )
    _v5_validate_policy(authorization, label="V5 authorization")
    incident_chain = _v5_validate_incident_chain(
        root, authorization.get("incident"), authorization.get("incident_correction")
    )
    v4_base = _v5_load_and_validate_v4_authorization_base(
        root, authorization.get("v4_authorization_base")
    )
    rejected_review = _v5_load_rejected_v4_review(root, v4_base)
    _require(
        v4_base["authorization_created"] <= rejected_review["review_created"]
        < incident_chain["incident_created"]
        < incident_chain["correction_created"]
        <= authorization_created,
        "V4/V5 incident and authority chronology mismatch",
    )
    _v5_validate_transport_truncation_state(
        authorization.get("v4_review_transport_truncation_state"),
        rejected_review["state"],
    )
    _require(
        authorization.get("recovered_afterstate_commitment")
        == v4_base["commitment"]
        == V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "V5 authorization compact afterstate commitment mismatch",
    )
    frozen_v4_actual: dict[str, dict[str, Any]] = {}
    frozen_v4_paths = {
        "superseded_v4_auditor": FROZEN_V4_AUDITOR,
        "superseded_v4_auditor_test": FROZEN_V4_AUDITOR_TEST,
        "superseded_v4_sealer": FROZEN_V4_SEALER,
        "superseded_v4_sealer_test": FROZEN_V4_SEALER_TEST,
    }
    for role, expected in FROZEN_V4_SOURCE_IDENTITIES.items():
        _v4_require_exact_declared_identity(
            authorization.get(role), expected, label=f"V5 {role}"
        )
        frozen_v4_actual[role] = _v4_verify_external_identity(
            authorization[role],
            frozen_v4_paths[role],
            label=f"V5 {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    v5_paths = {
        "v5_auditor": Path(__file__).resolve(),
        "v5_auditor_test": AUDITOR_TEST,
        "v5_sealer": V5_SEALER,
        "v5_sealer_test": V5_SEALER_TEST,
    }
    v5_actual = {
        role: _v4_verify_external_identity(
            authorization.get(role), path, label=f"V5 executing {role}"
        )
        for role, path in v5_paths.items()
    }
    _v5_validate_zero_snapshot(
        root, authorization.get("preaudit_zero_mutation_snapshot")
    )
    _require(
        authorization.get("required_command")
        == _v5_required_command(root, expected_authorization_path, expected_go_path),
        "V5 authorization required command mismatch",
    )
    evidence_bound = {
        "v4_authorization": v4_base["identity"],
        "v4_review_transport_truncation_incident": (
            incident_chain["incident_identity"]
        ),
        "v4_review_transport_truncation_incident_correction": (
            incident_chain["correction_identity"]
        ),
        **frozen_v4_actual,
        **v5_actual,
    }
    _v5_validate_test_evidence(
        authorization.get("test_evidence"),
        authorization_created=authorization_created,
        bound_identities=evidence_bound,
    )
    _require(
        review.get("schema_version") == 5
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V5"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V5"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == attempt_id
        and review.get("auditor_execution_started") is False
        and review.get("independent_go_required") is True
        and review_created >= authorization_created,
        "V5 independent review header/verdict mismatch",
    )
    _v5_validate_policy(review, label="V5 independent review")
    _require(
        go.get("schema_version") == 5
        and go.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V5"
        and go.get("status") == "GO_V5_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY"
        and go.get("audit_attempt_id") == attempt_id
        and go.get("recovery_attempt_id") == V4_RECOVERY_ATTEMPT_ID
        and go.get("recovery_rerun_authorized") is False
        and go.get("required_command") == authorization.get("required_command")
        and go_created >= review_created,
        "V5 independent GO header/binding mismatch",
    )
    _v5_validate_policy(go, label="V5 independent GO")
    authorization_actual = identity(expected_authorization_path, root)
    review_actual = identity(expected_review_path, root)
    common_fields = (
        "incident", "incident_correction", "v4_authorization_base",
        "v4_review_transport_truncation_state", *FROZEN_V4_SOURCE_IDENTITIES.keys(),
        "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
        "recovered_afterstate_commitment",
    )
    _require(
        review.get("authorization") == authorization_actual
        and go.get("authorization") == authorization_actual
        and go.get("independent_review") == review_actual
        and all(
            review.get(field) == authorization.get(field)
            and go.get(field) == authorization.get(field)
            for field in common_fields
        ),
        "V5 authorization/review/GO cross-binding mismatch",
    )
    checks = review.get("independent_checks")
    recheck = review.get("test_evidence_recheck")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == V5_REVIEW_CHECKS
        and all(checks.get(key) is True for key in V5_REVIEW_CHECKS),
        "V5 independent review check set/verdict mismatch",
    )
    _require(
        isinstance(recheck, Mapping)
        and set(recheck) == V4_REVIEW_TEST_RECHECK_KEYS
        and recheck.get("authorization_test_evidence_canonical_sha256")
        == canonical_payload_sha256(authorization["test_evidence"])
        and all(
            recheck.get(key) is True
            for key in V4_REVIEW_TEST_RECHECK_KEYS
            if key != "authorization_test_evidence_canonical_sha256"
        ),
        "V5 independent test-evidence recheck mismatch",
    )
    _require(
        _v5_validate_control_namespace(root) == namespace_start
        and _v5_control_temporary_files(root) == []
        and _v5_postrun_report_candidates(root) == [],
        "V5 predata control namespace changed",
    )
    return {
        "authorization": authorization,
        "authorization_identity": authorization_actual,
        "review": review,
        "review_identity": review_actual,
        "go": go,
        "go_identity": identity(expected_go_path, root),
        "incident_chain": incident_chain,
        "v4_base": v4_base,
        "rejected_v4_review": rejected_review,
        "recovered_afterstate": v4_base["afterstate"],
        "recovered_afterstate_commitment": v4_base["commitment"],
        "frozen_v4_source_identities": frozen_v4_actual,
        "v5_source_identities": v5_actual,
        "control_namespace": namespace_start,
    }


def _v5_validate_postauthority_control_state(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate compact controls and their reconstructed state without data reads."""

    authorization = authority["authorization"]
    afterstate = authority["recovered_afterstate"]
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(
            str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock"))
        ),
        "V5 recovered afterstate has an active lock",
    )
    commitment = _v5_recovered_afterstate_commitment(afterstate)
    _require(
        commitment == authority["recovered_afterstate_commitment"]
        and _v5_control_temporary_files(root) == []
        and _v5_postrun_report_candidates(root) == [],
        "V5 afterstate/commitment/temp/report recheck mismatch",
    )
    incident_chain = _v5_validate_incident_chain(
        root, authorization["incident"], authorization["incident_correction"]
    )
    _require(
        incident_chain["incident_identity"]
        == authority["incident_chain"]["incident_identity"]
        and incident_chain["correction_identity"]
        == authority["incident_chain"]["correction_identity"],
        "V5 incident/correction changed during audit",
    )
    v4_base = _v5_load_and_validate_v4_authorization_base(
        root, authorization["v4_authorization_base"]
    )
    _require(
        v4_base["afterstate"] == afterstate
        and v4_base["commitment"] == commitment,
        "V5 reconstructed V4 afterstate changed during audit",
    )
    rejected_review = _v5_load_rejected_v4_review(root, v4_base)
    _require(
        rejected_review["state"]
        == authorization["v4_review_transport_truncation_state"],
        "V5 rejected V4 review state changed during audit",
    )
    for role, record, relative in (
        ("authorization", authority["authorization_identity"], V5_AUTH_RELATIVE),
        ("independent review", authority["review_identity"], V5_REVIEW_RELATIVE),
        ("independent GO", authority["go_identity"], V5_GO_RELATIVE),
    ):
        verify_identity(
            record,
            path_under(root, relative),
            root=root,
            label=f"V5 postaudit {role}",
        )
    for role, path in (
        ("v5_auditor", Path(__file__).resolve()),
        ("v5_auditor_test", AUDITOR_TEST),
        ("v5_sealer", V5_SEALER),
        ("v5_sealer_test", V5_SEALER_TEST),
    ):
        _v4_verify_external_identity(
            authorization[role], path, label=f"V5 postaudit {role}"
        )
    for role, expected in FROZEN_V4_SOURCE_IDENTITIES.items():
        path = {
            "superseded_v4_auditor": FROZEN_V4_AUDITOR,
            "superseded_v4_auditor_test": FROZEN_V4_AUDITOR_TEST,
            "superseded_v4_sealer": FROZEN_V4_SEALER,
            "superseded_v4_sealer_test": FROZEN_V4_SEALER_TEST,
        }[role]
        _v4_verify_external_identity(
            authorization[role],
            path,
            label=f"V5 postaudit {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    _v5_validate_zero_snapshot(
        root, authorization["preaudit_zero_mutation_snapshot"]
    )
    _require(
        _v5_validate_control_namespace(root) == authority["control_namespace"],
        "V5 postauthority control namespace changed",
    )
    return commitment


def _v5_validate_postauthority_afterstate(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    v4_authorization = authority["v4_base"]["authorization"]
    afterstate = authority["recovered_afterstate"]
    commitment = _v5_validate_postauthority_control_state(root, authority)
    for role, record in v4_authorization["canonical_outputs"].items():
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V5 inherited canonical output {role}",
        )
    for role, record in v4_authorization["recovery_transaction"].items():
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V5 inherited recovery transaction {role}",
        )
    history = v4_authorization["recovery_history"]
    for role, record in history["completion_locks"].items():
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V5 inherited recovery completion lock {role}",
        )
    verify_identity(
        history["postcommit_input_audit"],
        path_under(root, str(history["postcommit_input_audit"]["path"])),
        root=root,
        label="V5 inherited recovery postcommit input audit",
    )
    history_dir = path_under(root, "decoded/recovery_history")
    _require(
        history_dir.is_dir() and not _linklike(history_dir),
        "V5 recovery-history directory invalid",
    )
    expected_history_paths = {
        str(record["path"]) for record in history["completion_locks"].values()
    } | {str(history["postcommit_input_audit"]["path"])}
    actual_history_entries = list(history_dir.iterdir())
    _require(
        len(actual_history_entries) == 3
        and all(path.is_file() and not _linklike(path) for path in actual_history_entries)
        and {path.relative_to(root).as_posix() for path in actual_history_entries}
        == expected_history_paths,
        "V5 recovery-history exact filesystem closure mismatch",
    )
    transaction_root = path_under(root, "raw/output_transactions")
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "V5 output-transaction root invalid",
    )
    recovery_authorization_v2 = load_json(
        path_under(root, V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"])
    )
    remnants = recovery_authorization_v2.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list) and len(remnants) == 2,
        "V5 incident-bound preplan-remnant declaration mismatch",
    )
    expected_transaction_files = {
        str(v4_authorization["recovery_transaction"]["plan"]["path"]),
        str(v4_authorization["recovery_transaction"]["commit"]["path"]),
    }
    for index, record in enumerate(remnants):
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "size_bytes", "sha256"},
            f"V5 preplan remnant identity invalid: {index}",
        )
        verify_identity(
            record,
            path_under(root, str(record["path"])),
            root=root,
            label=f"V5 incident-bound preplan remnant {index}",
        )
        expected_transaction_files.add(str(record["path"]))
    expected_transaction_dirs = {
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged/raw",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    actual_transaction_files: set[str] = set()
    actual_transaction_dirs: set[str] = set()
    for path in transaction_root.rglob("*"):
        _require(not _linklike(path), "V5 transaction tree contains a link/junction")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            actual_transaction_files.add(relative)
        elif path.is_dir():
            actual_transaction_dirs.add(relative)
        else:
            raise AuditFailure("V5 transaction tree contains a special object")
    _require(
        actual_transaction_files == expected_transaction_files
        and actual_transaction_dirs == expected_transaction_dirs
        and len(actual_transaction_files) == 4,
        "V5 transaction recursive exact filesystem closure mismatch",
    )
    progress_dir = path_under(root, "decoded/recovery_progress")
    _require(
        progress_dir.is_dir() and not _linklike(progress_dir),
        "V5 recovery-progress directory invalid",
    )
    actual_progress_paths = sorted(progress_dir.iterdir(), key=lambda path: path.name)
    _require(
        len(actual_progress_paths) == 104
        and all(path.is_file() and not _linklike(path) for path in actual_progress_paths),
        "V5 recovery-progress filesystem closure mismatch",
    )
    actual_inventory = [identity(path, root) for path in actual_progress_paths]
    _require(
        actual_inventory == v4_authorization["recovery_progress"]["inventory"],
        "V5 recovery-progress inventory rehash mismatch",
    )
    inventory_bytes = _v4_canonical_bytes(actual_inventory)
    _require(
        len(inventory_bytes) == V4_PROGRESS_INVENTORY_CANONICAL_BYTES
        and hashlib.sha256(inventory_bytes).hexdigest()
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "V5 recovery-progress canonical digest mismatch",
    )
    _require(
        _v5_validate_postauthority_control_state(root, authority) == commitment,
        "V5 postauthority compact control state changed",
    )
    return {
        "recovered_afterstate_commitment": commitment,
        "recovered_afterstate_canonical_sha256": (
            V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
        "canonical_output_count": 8,
        "transaction_recursive_file_count": 4,
        "transaction_recursive_directory_count": 7,
        "recovery_history_file_count": 3,
        "recovery_progress_file_count": 104,
        "active_locks_present": 0,
        "v5_control_temporary_files_present": 0,
        "canonical_v4_go_present": 0,
        "rejected_v4_review_rehashed": True,
        "incident_correction_rehashed": True,
        "control_namespace_exact": True,
    }


def _v7_required_command(
    root: Path, authorization_path: Path, go_path: Path
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), "-B", "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v7",
        "--root", str(root.resolve()),
        "--authorization", str(authorization_path.resolve()),
        "--independent-go", str(go_path.resolve()),
    ]


def _v7_require_canonical_runtime_entrypoint(
    root: Path, authorization_path: Path, go_path: Path
) -> None:
    _require(
        __spec__ is not None
        and __spec__.name == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v7"
        and __package__ == "scripts",
        "V7 production audit requires the canonical -m module entrypoint",
    )
    _require(
        sys.dont_write_bytecode is True
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None,
        "V7 production audit bytecode/cache environment mismatch",
    )
    expected_argv = [
        str(Path(__file__).resolve()), "--root", str(root.resolve()),
        "--authorization", str(authorization_path.resolve()),
        "--independent-go", str(go_path.resolve()),
    ]
    actual_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    _require(actual_argv == expected_argv, "V7 production argv is not exact canonical order")
    actual_orig_argv = [str(Path(sys.orig_argv[0]).resolve()), *sys.orig_argv[1:]]
    _require(
        actual_orig_argv == _v7_required_command(root, authorization_path, go_path),
        "V7 production interpreter/module argv is not exact authorized command",
    )


def _v7_test_command(relative: str) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), "-B", "-c",
        V6_PYTEST_NETWORK_GUARD_SOURCE, "-q", "-p", "no:cacheprovider", relative,
    ]


def _v7_control_destination_relatives() -> tuple[str, ...]:
    return (
        FROZEN_V3_AUTH_RELATIVE, FROZEN_V3_REVIEW_RELATIVE,
        FROZEN_V3_CANONICAL_GO_RELATIVE, FROZEN_V3_MISPLACED_GO_RELATIVE,
        V4_AUTH_RELATIVE, V4_REVIEW_RELATIVE, V4_GO_RELATIVE,
        V5_AUTH_RELATIVE, V5_REVIEW_RELATIVE, V5_GO_RELATIVE,
        V6_AUTH_RELATIVE, V6_REVIEW_RELATIVE, V6_GO_RELATIVE,
        V7_AUTH_RELATIVE, V7_REVIEW_RELATIVE, V7_GO_RELATIVE,
        V2_FALSE_REJECT_INCIDENT_RELATIVE, V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        V5_TRANSPORT_INCIDENT_RELATIVE, V5_TRANSPORT_CORRECTION_RELATIVE,
        V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
        V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
    )


def _v7_control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in _v7_control_destination_relatives():
        destination = path_under(root, relative)
        parent = destination.parent
        _require(parent.is_dir() and not _linklike(parent), "V7 control parent invalid")
        folded = destination.name.casefold()
        for sibling in parent.iterdir():
            name = sibling.name.casefold()
            if sibling.name == destination.name:
                continue
            if (
                name.startswith(f".{folded}.")
                or name.startswith(f"{folded}.")
                or name == f"{folded}.tmp"
            ):
                matches.append(sibling.relative_to(root).as_posix())
    return sorted(set(matches))


def _v7_postrun_report_candidates(root: Path) -> list[str]:
    return [
        relative
        for relative in V7_POSTRUN_REPORT_CANDIDATES
        if os.path.lexists(str(root / relative))
    ]


def _v7_validate_control_namespace(root: Path) -> dict[str, list[str]]:
    forbidden = (
        FROZEN_V3_CANONICAL_GO_RELATIVE, V4_GO_RELATIVE,
        V6_AUTH_RELATIVE, V6_REVIEW_RELATIVE, V6_GO_RELATIVE,
    )
    _require(
        all(not os.path.lexists(str(path_under(root, relative))) for relative in forbidden),
        "canonical V3/V4 GO and all V6 controls must remain absent",
    )
    expected = {
        "prereg": {
            Path(FROZEN_V3_AUTH_RELATIVE).name,
            Path(FROZEN_V3_MISPLACED_GO_RELATIVE).name,
            Path(V4_AUTH_RELATIVE).name,
            Path(V5_AUTH_RELATIVE).name,
            Path(V7_AUTH_RELATIVE).name,
        },
        "independent_redteam": {
            Path(FROZEN_V3_REVIEW_RELATIVE).name,
            Path(V4_REVIEW_RELATIVE).name,
            Path(V5_REVIEW_RELATIVE).name,
            Path(V5_GO_RELATIVE).name,
            Path(V7_REVIEW_RELATIVE).name,
            Path(V7_GO_RELATIVE).name,
        },
        "incidents": {
            Path(V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
            Path(V5_TRANSPORT_INCIDENT_RELATIVE).name,
            Path(V5_TRANSPORT_CORRECTION_RELATIVE).name,
            Path(V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE).name,
        },
    }
    prefixes = {
        "prereg": "decode_recovery_postrun_audit_",
        "independent_redteam": "track_a_decode_recovery_postrun_auditor_",
        "incidents": "track_a_decode_recovery_postrun_auditor_",
    }
    result: dict[str, list[str]] = {}
    for relative, expected_names in expected.items():
        parent = path_under(root, relative)
        _require(parent.is_dir() and not _linklike(parent), f"V7 namespace invalid: {relative}")
        matched: list[str] = []
        for entry in parent.iterdir():
            scan_name = entry.name.casefold().lstrip(".")
            if scan_name.startswith(prefixes[relative]):
                _require(entry.is_file() and not _linklike(entry), "V7 namespace object invalid")
                matched.append(entry.name)
        _require(
            set(matched) == expected_names
            and len({name.casefold() for name in matched}) == len(matched),
            f"V7 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched)
    return result


def _v7_validate_policy(payload: Mapping[str, Any], *, label: str) -> None:
    _v6_validate_policy(payload, label=label)


def _v7_assert_compact_control(payload: Mapping[str, Any], *, label: str) -> None:
    _v6_assert_compact_control(payload, label=label)


def _v7_validate_zero_snapshot(root: Path, payload: Any) -> None:
    _require(
        isinstance(payload, Mapping) and set(payload) == V7_ZERO_SNAPSHOT_KEYS,
        "V7 zero-mutation snapshot schema mismatch",
    )
    _require(
        payload.get("incident_postfailure_state_canonical_sha256")
        == V7_TIMESTAMP_INCIDENT_POSTFAILURE_CANONICAL_SHA256
        and payload.get("recovered_afterstate_commitment_canonical_sha256")
        == V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        and payload.get("active_locks")
        == {
            "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
            "decoded": {"path": "decoded/DECODE_RECOVERY_ACTIVE.lock", "present": False},
        }
        and payload.get("v7_control_temporary_files")
        == {
            "checked_destination_relative_paths": list(_v7_control_destination_relatives()),
            "matching_temporary_files": [],
        }
        and payload.get("postrun_audit_output_files_present") == 0
        and payload.get("network_requests") == 0
        and payload.get("audit_files_written") == 0
        and payload.get("labels_read") is False
        and payload.get("arrays_2024_read") is False
        and payload.get("arrays_2025_read") is False
        and payload.get("models_fit") == 0
        and payload.get("submission_csv_created") is False,
        "V7 zero-mutation snapshot value mismatch",
    )
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock")))
        and all(
            not os.path.lexists(str(path_under(root, relative)))
            for relative in (V6_AUTH_RELATIVE, V6_REVIEW_RELATIVE, V6_GO_RELATIVE)
        )
        and _v7_control_temporary_files(root) == []
        and _v7_postrun_report_candidates(root) == [],
        "V7 current zero-mutation filesystem mismatch",
    )


def _v7_validate_timestamp_chain(
    *,
    timestamp_incident: Mapping[str, Any],
    direct_payload: Mapping[str, Any],
    false_reject: Mapping[str, Any],
    v5_base: Mapping[str, Any],
    authorization: Mapping[str, Any],
    review: Mapping[str, Any],
    go: Mapping[str, Any],
) -> dict[str, Any]:
    declared_historical_raw = str(
        timestamp_incident["payload"]["historical_timestamp"]["value"]
    )
    declared_v5_authorization_raw = str(
        timestamp_incident["payload"]["historical_timestamp"]
        ["v5_authorization_created_utc"]
    )
    raw_values = {
        "historical_9e74": direct_payload.get("created_utc"),
        "v5_authorization": v5_base["authorization"].get("created_utc"),
        "v5_review": v5_base["review"].get("created_utc"),
        "v5_go": v5_base["go"].get("created_utc"),
        "v5_false_reject": false_reject["payload"].get("created_utc"),
        "v6_timestamp_false_reject": timestamp_incident["payload"].get("created_utc"),
        "v7_authorization": authorization.get("created_utc"),
        "v7_review": review.get("created_utc"),
        "v7_go": go.get("created_utc"),
    }
    _require(
        raw_values["historical_9e74"] == declared_historical_raw
        and raw_values["v5_authorization"] == declared_v5_authorization_raw
        == FROZEN_V5_AUTH_CREATED_UTC
        and raw_values["v5_review"] == FROZEN_V5_REVIEW_CREATED_UTC
        and raw_values["v5_go"] == FROZEN_V5_GO_CREATED_UTC
        and raw_values["v5_false_reject"] == "2026-08-11T02:51:47.476539Z"
        and raw_values["v6_timestamp_false_reject"]
        == "2026-08-11T04:22:34.325428Z",
        "V7 helper-returned raw timestamp binding mismatch",
    )
    parsed = {
        role: _v7_parse_rfc3339_utc_100ns(value, label=role)
        for role, value in raw_values.items()
    }
    ordered_roles = tuple(raw_values)
    _require(
        all(
            parsed[left]["ticks_100ns"] < parsed[right]["ticks_100ns"]
            for left, right in zip(ordered_roles, ordered_roles[1:])
        ),
        "V7 historical/control timestamp chronology mismatch",
    )
    return {"raw": raw_values, "parsed": parsed}


def _v7_validate_test_evidence(
    evidence: Any,
    *,
    authorization_created: Mapping[str, Any],
    bound_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_bound_keys = {
        "v5_authorization", "v5_independent_review", "v5_independent_go",
        "v5_historical_identity_false_reject_incident",
        "direct_file_failure_incident",
        "v6_auth_sealer_timestamp_precision_false_reject_incident",
        "superseded_v6_auditor", "superseded_v6_auditor_test",
        "superseded_v6_sealer", "superseded_v6_sealer_test",
        "v7_auditor", "v7_auditor_test", "v7_sealer", "v7_sealer_test",
    }
    _require(
        isinstance(evidence, Mapping)
        and set(evidence) == V7_TEST_EVIDENCE_KEYS
        and set(bound_identities) == expected_bound_keys,
        "V7 immutable test-evidence schema/bound-role mismatch",
    )
    created = _v7_parse_rfc3339_utc_100ns(
        evidence.get("created_utc"), label="V7 test evidence"
    )
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_TEST_EVIDENCE"
        and evidence.get("status") == "PASS_FROZEN_V7_AUDITOR_AND_SEALER_TESTS"
        and created["ticks_100ns"] <= int(authorization_created["ticks_100ns"])
        and evidence.get("bound_identities") == bound_identities
        and evidence.get("pytest_isolation")
        == "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS"
        and evidence.get("required_test_names") == list(V7_REQUIRED_TEST_NAMES)
        and evidence.get("required_test_names_present") is True
        and evidence.get("all_exit_codes_zero") is True,
        "V7 immutable test-evidence header/binding mismatch",
    )
    compile_record = evidence.get("source_compile")
    _require(
        isinstance(compile_record, Mapping)
        and set(compile_record) == {"method", "result", "files"}
        and compile_record.get("method") == "compile_exact_source_no_pyc"
        and compile_record.get("result") == "PASS"
        and compile_record.get("files")
        == [
            bound_identities["v7_auditor"],
            bound_identities["v7_auditor_test"],
            bound_identities["v7_sealer"],
            bound_identities["v7_sealer_test"],
        ],
        "V7 exact-source compile evidence mismatch",
    )
    expected_commands = {
        "v7_auditor_tests": _v7_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py"
        ),
        "v7_sealer_tests": _v7_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py"
        ),
        "frozen_v6_auditor_tests": _v7_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py"
        ),
        "frozen_v6_sealer_tests": _v7_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py"
        ),
    }
    runs = evidence.get("test_runs")
    _require(
        isinstance(runs, Mapping) and set(runs) == V7_TEST_RUN_KEYS,
        "V7 immutable pytest run map mismatch",
    )
    for role, expected_command in expected_commands.items():
        run = runs[role]
        _require(
            isinstance(run, Mapping)
            and set(run) == V4_TEST_RUN_RECORD_KEYS
            and run.get("command") == expected_command
            and run.get("exit_code") == 0
            and isinstance(run.get("summary"), str)
            and "passed" in str(run["summary"])
            and "failed" not in str(run["summary"]).casefold()
            and "error" not in str(run["summary"]).casefold()
            and isinstance(run.get("stdout_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stdout_sha256"])) is not None
            and isinstance(run.get("stderr_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stderr_sha256"])) is not None
            and run.get("network_guard_installed") is True
            and run.get("cacheprovider_disabled") is True,
            f"V7 immutable pytest run mismatch: {role}",
        )
    _require(
        evidence.get("production_shape_regression")
        == {
            "actual_root_historical_predata_validated": True,
            "actual_root_collect_build_self_validator_validated": True,
            "actual_timestamp_incident_validated_predata": True,
            "compact_afterstate_commitment_validated": True,
            "decoded_reached": True,
            "direct_file_failure_incident_9e74_validated_predata": True,
            "frozen_v6_timestamp_false_reject_reproduced": True,
            "full_afterstate_reconstructed_in_memory": True,
            "historical_current_role_distinction_validated": True,
            "offline_replay_reached": True,
            "original_v1_four_key_identity_passed": True,
            "raw_reached": True,
            "rfc3339_100ns_parser_zero_through_seven_exact": True,
            "synthetic_fixture": True,
            "transaction_reached": True,
            "v5_authority_base_validated": True,
            "v5_false_reject_incident_validated": True,
            "v6_auth_sealer_timestamp_false_reject_incident_validated": True,
            "v6_and_v5_production_data_and_replay_core_ast_identical": True,
        }
        and evidence.get("real_seven_spawn_regression")
        == {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v7",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v7",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        }
        and evidence.get("stdout_only_regression") is True
        and evidence.get("network_guard")
        == {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        }
        and evidence.get("network_requests") == 0
        and evidence.get("audit_files_written") == 0
        and evidence.get("labels_read") is False
        and evidence.get("arrays_2024_read") is False
        and evidence.get("arrays_2025_read") is False
        and evidence.get("models_fit") == 0
        and evidence.get("submission_csv_created") is False,
        "V7 immutable test-evidence safety/regression mismatch",
    )


def _v7_validate_predata_authority(
    root: Path, authorization_path: Path, go_path: Path
) -> dict[str, Any]:
    """Close V7 timestamp/history authority before any data dereference."""

    expected_authorization_path = path_under(root, V7_AUTH_RELATIVE)
    expected_review_path = path_under(root, V7_REVIEW_RELATIVE)
    expected_go_path = path_under(root, V7_GO_RELATIVE)
    _require(
        Path(os.path.abspath(authorization_path)) == expected_authorization_path
        and Path(os.path.abspath(go_path)) == expected_go_path,
        "V7 authorization/GO paths are not canonical",
    )
    for path, label in (
        (expected_authorization_path, "V7 authorization"),
        (expected_review_path, "V7 independent review"),
        (expected_go_path, "V7 independent GO"),
    ):
        require_no_symlink_chain(path, root, label=label)
        _require(
            os.path.lexists(str(path)) and path.is_file() and not _linklike(path),
            f"{label} is absent, non-regular or link-like",
        )
    _require(
        expected_review_path.stat().st_size <= 12_000
        and expected_go_path.stat().st_size <= 12_000,
        "V7 compact review or GO exceeds the 12KB transport limit",
    )
    namespace_start = _v7_validate_control_namespace(root)
    _require(_v7_control_temporary_files(root) == [], "V7 control temporary files present")
    _require(
        _v7_postrun_report_candidates(root) == [],
        "persisted V2/V3/V4/V5/V6/V7 postrun report candidate is forbidden",
    )
    authorization = _v5_load_strict_json(
        expected_authorization_path, label="V7 authorization"
    )
    review = _v5_load_strict_json(expected_review_path, label="V7 independent review")
    go = _v5_load_strict_json(expected_go_path, label="V7 independent GO")
    _require(set(authorization) == V7_AUTH_KEYS, "V7 authorization schema mismatch")
    _require(set(review) == V7_REVIEW_KEYS, "V7 independent review schema mismatch")
    _require(set(go) == V7_GO_KEYS, "V7 independent GO schema mismatch")
    for payload, label in (
        (authorization, "V7 authorization"),
        (review, "V7 independent review"),
        (go, "V7 independent GO"),
    ):
        _v7_assert_compact_control(payload, label=label)
    authorization_created = _v7_parse_rfc3339_utc_100ns(
        authorization.get("created_utc"), label="V7 authorization"
    )
    review_created = _v7_parse_rfc3339_utc_100ns(
        review.get("created_utc"), label="V7 review"
    )
    go_created = _v7_parse_rfc3339_utc_100ns(go.get("created_utc"), label="V7 GO")
    attempt_id = authorization.get("audit_attempt_id")
    derived_attempt = "postrun_audit_v7__" + str(authorization["created_utc"]).translate(
        str.maketrans("", "", "-:.")
    )
    _require(
        authorization.get("schema_version") == 7
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V7"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V7"
        and isinstance(attempt_id, str)
        and re.fullmatch(r"postrun_audit_v7__[0-9]{8}T[0-9]{12}Z", attempt_id)
        is not None
        and attempt_id == derived_attempt
        and authorization.get("recovery_attempt_id") == FROZEN_V5_RECOVERY_ATTEMPT_ID
        and authorization.get("runtime_identity_sha256") == V4_RUNTIME_IDENTITY_SHA256
        and authorization.get("independent_review_required") is True
        and authorization.get("independent_go_required") is True,
        "V7 authorization header/binding mismatch",
    )
    _v7_validate_policy(authorization, label="V7 authorization")

    # Documentary incidents and their exact raw timestamps are the first
    # semantic authority roles, before V5 afterstate reconstruction or data.
    timestamp_incident = _v7_validate_timestamp_false_reject_incident(
        root, authorization.get("incident")
    )
    false_reject_record = {
        "path": V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": V6_HISTORICAL_FALSE_REJECT_INCIDENT_SIZE_BYTES,
        "sha256": V6_HISTORICAL_FALSE_REJECT_INCIDENT_SHA256,
    }
    false_reject = _v6_validate_v5_false_reject_incident(
        root, false_reject_record
    )
    direct_path = path_under(root, RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE)
    direct_identity = identity(direct_path, root)
    direct_payload = _v5_load_strict_json(
        direct_path, label="immutable historical direct-file failure incident"
    )
    historical_contract = _v6_validate_historical_9e74_payload(
        direct_payload, direct_identity
    )
    expected_historical_contract = _v6_expected_historical_contract(direct_identity)
    _require(
        historical_contract == expected_historical_contract,
        "V7 internally reconstructed historical contract mismatch",
    )
    _v6_validate_historical_contract(
        authorization.get("historical_failed_head_identity_contract"),
        expected_historical_contract,
    )
    v5_base = _v6_load_and_validate_v5_authority_base(
        root, authorization.get("v5_authority_base")
    )
    timestamp_chain = _v7_validate_timestamp_chain(
        timestamp_incident=timestamp_incident,
        direct_payload=direct_payload,
        false_reject=false_reject,
        v5_base=v5_base,
        authorization=authorization,
        review=review,
        go=go,
    )
    _require(
        authorization.get("recovered_afterstate_commitment")
        == v5_base["commitment"]
        == V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "V7 recovered-afterstate commitment mismatch",
    )
    frozen_v6_actual: dict[str, dict[str, Any]] = {}
    frozen_v6_paths = {
        "superseded_v6_auditor": FROZEN_V6_AUDITOR,
        "superseded_v6_auditor_test": FROZEN_V6_AUDITOR_TEST,
        "superseded_v6_sealer": FROZEN_V6_SEALER,
        "superseded_v6_sealer_test": FROZEN_V6_SEALER_TEST,
    }
    for role, expected in FROZEN_V6_SOURCE_IDENTITIES.items():
        _v4_require_exact_declared_identity(
            authorization.get(role), expected, label=f"V7 {role}"
        )
        frozen_v6_actual[role] = _v4_verify_external_identity(
            authorization[role], frozen_v6_paths[role], label=f"V7 {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    v7_paths = {
        "v7_auditor": Path(__file__).resolve(),
        "v7_auditor_test": AUDITOR_TEST,
        "v7_sealer": V7_SEALER,
        "v7_sealer_test": V7_SEALER_TEST,
    }
    v7_actual = {
        role: _v4_verify_external_identity(
            authorization.get(role), path, label=f"V7 executing {role}"
        )
        for role, path in v7_paths.items()
    }
    _v7_validate_zero_snapshot(
        root, authorization.get("preaudit_zero_mutation_snapshot")
    )
    _require(
        authorization.get("required_command")
        == _v7_required_command(root, expected_authorization_path, expected_go_path),
        "V7 authorization required command mismatch",
    )
    evidence_bound = {
        "v5_authorization": FROZEN_V5_CONTROL_IDENTITIES["authorization"],
        "v5_independent_review": FROZEN_V5_CONTROL_IDENTITIES["independent_review"],
        "v5_independent_go": FROZEN_V5_CONTROL_IDENTITIES["independent_go"],
        "v5_historical_identity_false_reject_incident": false_reject["identity"],
        "direct_file_failure_incident": direct_identity,
        "v6_auth_sealer_timestamp_precision_false_reject_incident": (
            timestamp_incident["identity"]
        ),
        **frozen_v6_actual,
        **v7_actual,
    }
    _v7_validate_test_evidence(
        authorization.get("test_evidence"),
        authorization_created=authorization_created,
        bound_identities=evidence_bound,
    )
    _require(
        review.get("schema_version") == 7
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V7"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V7"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == attempt_id
        and review.get("auditor_execution_started") is False
        and review.get("independent_go_required") is True,
        "V7 independent review header/verdict mismatch",
    )
    _v7_validate_policy(review, label="V7 independent review")
    _require(
        go.get("schema_version") == 7
        and go.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V7"
        and go.get("status") == "GO_V7_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY"
        and go.get("audit_attempt_id") == attempt_id
        and go.get("recovery_attempt_id") == FROZEN_V5_RECOVERY_ATTEMPT_ID
        and go.get("recovery_rerun_authorized") is False
        and go.get("required_command") == authorization.get("required_command"),
        "V7 independent GO header/binding mismatch",
    )
    _v7_validate_policy(go, label="V7 independent GO")
    authorization_actual = identity(expected_authorization_path, root)
    review_actual = identity(expected_review_path, root)
    common_fields = (
        "incident", "v5_authority_base", "historical_failed_head_identity_contract",
        *FROZEN_V6_SOURCE_IDENTITIES.keys(), "v7_auditor", "v7_auditor_test",
        "v7_sealer", "v7_sealer_test", "recovered_afterstate_commitment",
    )
    _require(
        review.get("authorization") == authorization_actual
        and go.get("authorization") == authorization_actual
        and go.get("independent_review") == review_actual
        and all(
            review.get(field) == authorization.get(field)
            and go.get(field) == authorization.get(field)
            for field in common_fields
        ),
        "V7 authorization/review/GO cross-binding mismatch",
    )
    checks = review.get("independent_checks")
    recheck = review.get("test_evidence_recheck")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == V7_REVIEW_CHECKS
        and all(checks.get(key) is True for key in V7_REVIEW_CHECKS),
        "V7 independent review check set/verdict mismatch",
    )
    _require(
        isinstance(recheck, Mapping)
        and set(recheck) == V4_REVIEW_TEST_RECHECK_KEYS
        and recheck.get("authorization_test_evidence_canonical_sha256")
        == canonical_payload_sha256(authorization["test_evidence"])
        and all(
            recheck.get(key) is True
            for key in V4_REVIEW_TEST_RECHECK_KEYS
            if key != "authorization_test_evidence_canonical_sha256"
        ),
        "V7 independent test-evidence recheck mismatch",
    )
    _require(
        _v7_validate_control_namespace(root) == namespace_start
        and _v7_control_temporary_files(root) == []
        and _v7_postrun_report_candidates(root) == [],
        "V7 predata control namespace changed",
    )
    return {
        "authorization": authorization,
        "authorization_identity": authorization_actual,
        "review": review,
        "review_identity": review_actual,
        "go": go,
        "go_identity": identity(expected_go_path, root),
        "timestamp_incident": timestamp_incident,
        "false_reject_incident": false_reject,
        "historical_contract": historical_contract,
        "timestamp_chain": timestamp_chain,
        "v5_base": v5_base,
        "recovered_afterstate": v5_base["recovered_afterstate"],
        "recovered_afterstate_commitment": v5_base["commitment"],
        "frozen_v6_source_identities": frozen_v6_actual,
        "v7_source_identities": v7_actual,
        "control_namespace": namespace_start,
    }


def _v7_parse_rfc3339_utc_100ns(value: Any, *, label: str) -> dict[str, Any]:
    """Parse exact UTC RFC3339 text into lossless integer 100ns ticks."""

    _require(isinstance(value, str), f"{label} timestamp is not a string")
    match = re.fullmatch(
        r"([0-9]{4})-([0-9]{2})-([0-9]{2})T"
        r"([0-9]{2}):([0-9]{2}):([0-9]{2})"
        r"(?:\.([0-9]{1,7}))?Z",
        value,
    )
    _require(match is not None, f"{label} timestamp is invalid")
    assert match is not None
    year, month, day, hour, minute, second = (
        int(part) for part in match.groups()[:6]
    )
    fraction = match.group(7) or ""
    try:
        whole = datetime(
            year, month, day, hour, minute, second, tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise AuditFailure(f"{label} timestamp is invalid") from exc
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = whole - epoch
    whole_seconds = delta.days * 86_400 + delta.seconds
    fraction_ticks = int(fraction.ljust(7, "0") or "0")
    return {
        "raw": value,
        "fractional_digits": len(fraction),
        "ticks_100ns": whole_seconds * 10_000_000 + fraction_ticks,
    }


def _v7_validate_timestamp_false_reject_incident(
    root: Path, record: Any
) -> dict[str, Any]:
    """Strictly bind the immutable V6 authorization-sealer false reject."""

    expected = {
        "path": V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": V7_TIMESTAMP_FALSE_REJECT_INCIDENT_SIZE_BYTES,
        "sha256": V7_TIMESTAMP_FALSE_REJECT_INCIDENT_SHA256,
    }
    _v4_require_exact_declared_identity(
        record, expected, label="V6 authorization-sealer timestamp incident"
    )
    actual = _v4_verify_exact_root_identity(
        root,
        record,
        expected,
        label="V6 authorization-sealer timestamp incident",
    )
    payload = _v5_load_strict_json(
        path_under(root, V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE),
        label="V6 authorization-sealer timestamp incident",
    )
    _require(
        set(payload)
        == {
            "schema_version", "artifact_type", "status", "created_utc",
            "authority_scope", "failed_execution", "immutable_v6_candidate",
            "historical_timestamp", "root_cause", "execution_boundary",
            "postfailure_state", "prohibitions", "required_v7_supersession",
        }
        and payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_AUTH_SEALER_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT"
        and payload.get("status")
        == "SEALED_V6_AUTH_SEALER_TIMESTAMP_PRECISION_FALSE_REJECT_BEFORE_AUTH_PUBLICATION_V7_SUPERSESSION_REQUIRED",
        "V6 authorization-sealer timestamp incident schema/header mismatch",
    )
    canonical = _v4_canonical_bytes(payload)
    _require(
        len(canonical) == V7_TIMESTAMP_FALSE_REJECT_INCIDENT_CANONICAL_BYTES
        and hashlib.sha256(canonical).hexdigest()
        == V7_TIMESTAMP_FALSE_REJECT_INCIDENT_CANONICAL_SHA256,
        "V6 authorization-sealer timestamp incident canonical identity mismatch",
    )
    nested_identities = {
        "authority_scope": (149, "72a088aaac3fb8ecb776c022576da447a77c04d8516562fc2a54b8346a0d5738"),
        "failed_execution": (1_612, "c36985541fdc5738872193856ce1b97af471c7931a406e828bffabe01a0e1a07"),
        "immutable_v6_candidate": (1_146, "d70b17da20d6447b72ea8bdb3bce394b2e0281f618dde769bfd1a6e0e471bbf1"),
        "historical_timestamp": (442, "75f21ea35144de130af28014d4128ad0c810441c94fdf4c5b1234db8cd4e8deb"),
        "root_cause": (607, "9528aee14be053cc444066316154e3ad821761991db40b24bba1675f425e845f"),
        "execution_boundary": (1_855, "c53400e0e13bea1edcfec1dbcdcbc1a7747824afd7a1250dfdebfb391e98f3ef"),
        "postfailure_state": (1_825, V7_TIMESTAMP_INCIDENT_POSTFAILURE_CANONICAL_SHA256),
        "prohibitions": (156, "db2f96e4f611b8d79815b93fc416a93a06ca266343fea488679a1b995687a003"),
        "required_v7_supersession": (1_331, "c55f99b8ce126ed8ccf82c8796e1dda2abcb621c2112ca0136052b12d5ffdebc"),
    }
    for key, (size_bytes, sha256) in nested_identities.items():
        value = payload.get(key)
        _require(isinstance(value, Mapping), f"timestamp incident {key} is invalid")
        encoded = _v4_canonical_bytes(value)
        _require(
            len(encoded) == size_bytes
            and hashlib.sha256(encoded).hexdigest() == sha256,
            f"timestamp incident {key} canonical identity mismatch",
        )
    _require(
        payload["authority_scope"]
        == {
            "documentary_only": True,
            "v6_retry_authorized": False,
            "v6_review_go_or_live_audit_authorized": False,
            "v7_implementation_or_execution_authorized": False,
        }
        and payload["prohibitions"]
        == {
            "mutate_v1_through_v6": False,
            "network_or_recovery_rerun": False,
            "publish_v6_auth_review_or_go": False,
            "retry_v6_sealer": False,
            "run_v6_postrun_auditor": False,
        },
        "timestamp incident authority/prohibition semantics mismatch",
    )
    created = _v7_parse_rfc3339_utc_100ns(
        payload.get("created_utc"), label="timestamp incident created_utc"
    )
    historical = payload["historical_timestamp"]
    _require(
        set(historical)
        == {
            "incident", "field_path", "value", "fractional_second_digits",
            "precision_unit", "rfc3339_utc", "immutable",
            "v5_authorization_created_utc", "chronology_valid",
        }
        and historical.get("incident")
        == {
            "path": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
            "size_bytes": 7_732,
            "sha256": RECOVERY_SEALER_ENTRYPOINT_INCIDENT_SHA256,
        }
        and historical.get("field_path") == "created_utc"
        and historical.get("value") == "2026-08-10T18:45:00.5807528Z"
        and historical.get("fractional_second_digits") == 7
        and historical.get("precision_unit") == "100_NANOSECONDS"
        and historical.get("rfc3339_utc") is True
        and historical.get("immutable") is True
        and historical.get("v5_authorization_created_utc")
        == FROZEN_V5_AUTH_CREATED_UTC
        and historical.get("chronology_valid") is True,
        "timestamp incident historical raw-string binding mismatch",
    )
    historical_created = _v7_parse_rfc3339_utc_100ns(
        historical["value"], label="historical 9e74 created_utc"
    )
    v5_authorization_created = _v7_parse_rfc3339_utc_100ns(
        historical["v5_authorization_created_utc"],
        label="frozen V5 authorization created_utc",
    )
    _require(
        historical_created["ticks_100ns"]
        < v5_authorization_created["ticks_100ns"]
        < created["ticks_100ns"],
        "timestamp incident internal chronology mismatch",
    )
    return {
        "identity": actual,
        "payload": payload,
        "created": created,
        "historical_created": historical_created,
        "v5_authorization_created": v5_authorization_created,
    }


def _v6_required_command(root: Path, authorization_path: Path, go_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), "-B", "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
        "--root", str(root.resolve()),
        "--authorization", str(authorization_path.resolve()),
        "--independent-go", str(go_path.resolve()),
    ]


def _v6_require_canonical_runtime_entrypoint(
    root: Path, authorization_path: Path, go_path: Path
) -> None:
    _require(
        __spec__ is not None
        and __spec__.name == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6"
        and __package__ == "scripts",
        "V6 production audit requires the canonical -m module entrypoint",
    )
    _require(
        sys.dont_write_bytecode is True
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None,
        "V6 production audit bytecode/cache environment mismatch",
    )
    expected_argv = [
        str(Path(__file__).resolve()),
        "--root",
        str(root.resolve()),
        "--authorization",
        str(authorization_path.resolve()),
        "--independent-go",
        str(go_path.resolve()),
    ]
    actual_argv = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    _require(actual_argv == expected_argv, "V6 production argv is not exact canonical order")
    actual_orig_argv = [
        str(Path(sys.orig_argv[0]).resolve()), *sys.orig_argv[1:]
    ]
    _require(
        actual_orig_argv == _v6_required_command(root, authorization_path, go_path),
        "V6 production interpreter/module argv is not exact authorized command",
    )


def _v6_test_command(relative: str) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), "-B", "-c",
        V6_PYTEST_NETWORK_GUARD_SOURCE, "-q", "-p", "no:cacheprovider", relative,
    ]


def _v6_control_destination_relatives() -> tuple[str, ...]:
    return (
        FROZEN_V3_AUTH_RELATIVE, FROZEN_V3_REVIEW_RELATIVE,
        FROZEN_V3_CANONICAL_GO_RELATIVE, FROZEN_V3_MISPLACED_GO_RELATIVE,
        V4_AUTH_RELATIVE, V4_REVIEW_RELATIVE, V4_GO_RELATIVE,
        V5_AUTH_RELATIVE, V5_REVIEW_RELATIVE, V5_GO_RELATIVE,
        V6_AUTH_RELATIVE, V6_REVIEW_RELATIVE, V6_GO_RELATIVE,
        V2_FALSE_REJECT_INCIDENT_RELATIVE, V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        V5_TRANSPORT_INCIDENT_RELATIVE, V5_TRANSPORT_CORRECTION_RELATIVE,
        V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
    )


def _v6_control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in _v6_control_destination_relatives():
        destination = path_under(root, relative)
        parent = destination.parent
        _require(parent.is_dir() and not _linklike(parent), "V6 control parent invalid")
        folded = destination.name.casefold()
        for sibling in parent.iterdir():
            name = sibling.name.casefold()
            if sibling.name == destination.name:
                continue
            if (
                name.startswith(f".{folded}.")
                or name.startswith(f"{folded}.")
                or name == f"{folded}.tmp"
            ):
                matches.append(sibling.relative_to(root).as_posix())
    return sorted(set(matches))


def _v6_postrun_report_candidates(root: Path) -> list[str]:
    return [
        relative
        for relative in V6_POSTRUN_REPORT_CANDIDATES
        if os.path.lexists(str(root / relative))
    ]


def _v6_validate_control_namespace(root: Path) -> dict[str, list[str]]:
    _require(
        not os.path.lexists(str(path_under(root, FROZEN_V3_CANONICAL_GO_RELATIVE)))
        and not os.path.lexists(str(path_under(root, V4_GO_RELATIVE))),
        "canonical V3/V4 GO must remain absent",
    )
    expected = {
        "prereg": {
            Path(FROZEN_V3_AUTH_RELATIVE).name,
            Path(FROZEN_V3_MISPLACED_GO_RELATIVE).name,
            Path(V4_AUTH_RELATIVE).name,
            Path(V5_AUTH_RELATIVE).name,
            Path(V6_AUTH_RELATIVE).name,
        },
        "independent_redteam": {
            Path(FROZEN_V3_REVIEW_RELATIVE).name,
            Path(V4_REVIEW_RELATIVE).name,
            Path(V5_REVIEW_RELATIVE).name,
            Path(V5_GO_RELATIVE).name,
            Path(V6_REVIEW_RELATIVE).name,
            Path(V6_GO_RELATIVE).name,
        },
        "incidents": {
            Path(V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
            Path(V5_TRANSPORT_INCIDENT_RELATIVE).name,
            Path(V5_TRANSPORT_CORRECTION_RELATIVE).name,
            Path(V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE).name,
        },
    }
    prefixes = {
        "prereg": "decode_recovery_postrun_audit_",
        "independent_redteam": "track_a_decode_recovery_postrun_auditor_",
        "incidents": "track_a_decode_recovery_postrun_auditor_",
    }
    result: dict[str, list[str]] = {}
    for relative, expected_names in expected.items():
        parent = path_under(root, relative)
        _require(parent.is_dir() and not _linklike(parent), f"V6 namespace invalid: {relative}")
        matched: list[str] = []
        for entry in parent.iterdir():
            scan_name = entry.name.casefold().lstrip(".")
            if scan_name.startswith(prefixes[relative]):
                _require(entry.is_file() and not _linklike(entry), "V6 namespace object invalid")
                matched.append(entry.name)
        _require(
            set(matched) == expected_names
            and len({name.casefold() for name in matched}) == len(matched),
            f"V6 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched)
    return result


def _v6_validate_policy(payload: Mapping[str, Any], *, label: str) -> None:
    _require(
        payload.get("max_spawn_processes") == 7
        and payload.get("network_requests_allowed") == 0
        and payload.get("audit_files_written_allowed") == 0
        and payload.get("labels_read_allowed") is False
        and payload.get("arrays_2024_read_allowed") is False
        and payload.get("arrays_2025_read_allowed") is False
        and payload.get("models_fit_allowed") == 0
        and payload.get("submission_csv_allowed") is False
        and payload.get("full_offline_redecode_required") is True
        and payload.get("stdout_only") is True,
        f"{label} policy mismatch",
    )


def _v6_expected_historical_contract(identity_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "incident": dict(identity_record),
        "historical_incident_contract": {
            "failed_head_identities": {
                "canonical_size_bytes": V6_HISTORICAL_FAILED_HEADS_CANONICAL_BYTES,
                "canonical_sha256": V6_HISTORICAL_FAILED_HEADS_CANONICAL_SHA256,
            },
            "verified_zero_state_after_failure": {
                "canonical_size_bytes": V6_HISTORICAL_ZERO_STATE_CANONICAL_BYTES,
                "canonical_sha256": V6_HISTORICAL_ZERO_STATE_CANONICAL_SHA256,
            },
            "absent_control_paths": {
                "canonical_size_bytes": V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_BYTES,
                "canonical_sha256": V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_SHA256,
            },
        },
        "validation_mode": (
            "IMMUTABLE_9E74_HISTORICAL_CONTRACT_PREDATA_NO_CURRENT_ROLE_DEREFERENCE"
        ),
        "validation_moved_before_parquet_and_data_reads": True,
        "current_role_path_dereference_forbidden": True,
    }


def _v6_validate_historical_contract(payload: Any, expected: Mapping[str, Any]) -> None:
    _require(
        isinstance(payload, Mapping)
        and set(payload) == V6_HISTORICAL_CONTRACT_KEYS
        and isinstance(payload.get("historical_incident_contract"), Mapping)
        and set(payload["historical_incident_contract"])
        == V6_HISTORICAL_COMMITMENT_KEYS
        and all(
            isinstance(value, Mapping) and set(value) == V6_CANONICAL_COMMITMENT_KEYS
            for value in payload["historical_incident_contract"].values()
        )
        and payload == expected,
        "V6 historical failed-head identity contract mismatch",
    )


def _v6_load_and_validate_v5_authority_base(
    root: Path, base: Any
) -> dict[str, Any]:
    _require(
        isinstance(base, Mapping) and set(base) == V6_AUTHORITY_BASE_KEYS,
        "V6 frozen V5 authority-base schema mismatch",
    )
    for role, expected in FROZEN_V5_CONTROL_IDENTITIES.items():
        _v4_require_exact_declared_identity(
            base.get(role), expected, label=f"V6 frozen V5 {role}"
        )
        _v4_verify_exact_root_identity(
            root, base[role], expected, label=f"V6 frozen V5 {role}"
        )
    authorization_path = path_under(root, V5_AUTH_RELATIVE)
    review_path = path_under(root, V5_REVIEW_RELATIVE)
    go_path = path_under(root, V5_GO_RELATIVE)
    authorization = _v5_load_strict_json(authorization_path, label="frozen V5 authorization")
    review = _v5_load_strict_json(review_path, label="frozen V5 independent review")
    go = _v5_load_strict_json(go_path, label="frozen V5 independent GO")
    for payload, expected_bytes, expected_sha, label in (
        (authorization, FROZEN_V5_AUTH_PAYLOAD_CANONICAL_BYTES,
         FROZEN_V5_AUTH_PAYLOAD_CANONICAL_SHA256, "authorization"),
        (review, FROZEN_V5_REVIEW_PAYLOAD_CANONICAL_BYTES,
         FROZEN_V5_REVIEW_PAYLOAD_CANONICAL_SHA256, "independent review"),
        (go, FROZEN_V5_GO_PAYLOAD_CANONICAL_BYTES,
         FROZEN_V5_GO_PAYLOAD_CANONICAL_SHA256, "independent GO"),
    ):
        encoded = _v4_canonical_bytes(payload)
        _require(
            len(encoded) == expected_bytes
            and hashlib.sha256(encoded).hexdigest() == expected_sha,
            f"frozen V5 {label} canonical payload mismatch",
        )
    _require(
        set(authorization) == V5_AUTH_KEYS
        and set(review) == V5_REVIEW_KEYS
        and set(go) == V5_GO_KEYS,
        "frozen V5 control schema mismatch",
    )
    auth_created = _v4_created_utc(
        authorization.get("created_utc"), label="frozen V5 authorization"
    )
    review_created = _v4_created_utc(
        review.get("created_utc"), label="frozen V5 independent review"
    )
    go_created = _v4_created_utc(go.get("created_utc"), label="frozen V5 independent GO")
    _require(
        authorization.get("schema_version") == 5
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V5"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5"
        and authorization.get("audit_attempt_id") == FROZEN_V5_AUDIT_ATTEMPT_ID
        and authorization.get("recovery_attempt_id") == FROZEN_V5_RECOVERY_ATTEMPT_ID
        and authorization.get("runtime_identity_sha256") == V4_RUNTIME_IDENTITY_SHA256
        and authorization.get("independent_review_required") is True
        and authorization.get("independent_go_required") is True
        and review.get("schema_version") == 5
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V5"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V5"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == FROZEN_V5_AUDIT_ATTEMPT_ID
        and review.get("auditor_execution_started") is False
        and review.get("independent_go_required") is True
        and go.get("schema_version") == 5
        and go.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V5"
        and go.get("status") == "GO_V5_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY"
        and go.get("audit_attempt_id") == FROZEN_V5_AUDIT_ATTEMPT_ID
        and go.get("recovery_attempt_id") == FROZEN_V5_RECOVERY_ATTEMPT_ID
        and go.get("recovery_rerun_authorized") is False
        and auth_created < review_created < go_created,
        "frozen V5 authority-chain header/chronology mismatch",
    )
    _v6_validate_policy(authorization, label="frozen V5 authorization")
    _v6_validate_policy(review, label="frozen V5 independent review")
    _v6_validate_policy(go, label="frozen V5 independent GO")
    authorization_identity = identity(authorization_path, root)
    review_identity = identity(review_path, root)
    _require(
        review.get("authorization") == authorization_identity
        and go.get("authorization") == authorization_identity
        and go.get("independent_review") == review_identity
        and go.get("required_command") == authorization.get("required_command")
        and authorization.get("required_command")
        == _v5_required_command(
            ROOT_DEFAULT,
            path_under(ROOT_DEFAULT, V5_AUTH_RELATIVE),
            path_under(ROOT_DEFAULT, V5_GO_RELATIVE),
        ),
        "frozen V5 control cross-binding/command mismatch",
    )
    v4_base = _v5_load_and_validate_v4_authorization_base(
        root, authorization.get("v4_authorization_base")
    )
    incident_chain = _v5_validate_incident_chain(
        root, authorization.get("incident"), authorization.get("incident_correction")
    )
    rejected = _v5_load_rejected_v4_review(root, v4_base)
    _v5_validate_transport_truncation_state(
        authorization.get("v4_review_transport_truncation_state"), rejected["state"]
    )
    _require(
        authorization.get("recovered_afterstate_commitment")
        == v4_base["commitment"]
        == V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "frozen V5 recovered-afterstate commitment mismatch",
    )
    frozen_v4_actual: dict[str, dict[str, Any]] = {}
    for role, expected in FROZEN_V4_SOURCE_IDENTITIES.items():
        path = {
            "superseded_v4_auditor": FROZEN_V4_AUDITOR,
            "superseded_v4_auditor_test": FROZEN_V4_AUDITOR_TEST,
            "superseded_v4_sealer": FROZEN_V4_SEALER,
            "superseded_v4_sealer_test": FROZEN_V4_SEALER_TEST,
        }[role]
        _v4_require_exact_declared_identity(
            authorization.get(role), expected, label=f"frozen V5 {role}"
        )
        frozen_v4_actual[role] = _v4_verify_external_identity(
            authorization[role], path, label=f"frozen V5 {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    frozen_v5_actual: dict[str, dict[str, Any]] = {}
    for role, expected in FROZEN_V5_SOURCE_IDENTITIES.items():
        auth_role = role.removeprefix("superseded_")
        path = {
            "v5_auditor": FROZEN_V5_AUDITOR,
            "v5_auditor_test": FROZEN_V5_AUDITOR_TEST,
            "v5_sealer": FROZEN_V5_SEALER,
            "v5_sealer_test": FROZEN_V5_SEALER_TEST,
        }[auth_role]
        declared = authorization.get(auth_role)
        _require(
            isinstance(declared, Mapping)
            and int(declared.get("size_bytes", -1)) == int(expected["size_bytes"])
            and declared.get("sha256") == expected["sha256"]
            and os.path.normcase(str(declared.get("path")))
            == os.path.normcase(str(expected["path"])),
            f"frozen V5 source declaration mismatch: {role}",
        )
        frozen_v5_actual[role] = _v4_verify_external_identity(
            declared, path, label=f"frozen V5 source {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    evidence_bound = {
        "v4_authorization": v4_base["identity"],
        "v4_review_transport_truncation_incident": incident_chain["incident_identity"],
        "v4_review_transport_truncation_incident_correction": incident_chain["correction_identity"],
        **frozen_v4_actual,
        "v5_auditor": frozen_v5_actual["superseded_v5_auditor"],
        "v5_auditor_test": frozen_v5_actual["superseded_v5_auditor_test"],
        "v5_sealer": frozen_v5_actual["superseded_v5_sealer"],
        "v5_sealer_test": frozen_v5_actual["superseded_v5_sealer_test"],
    }
    _v5_validate_test_evidence(
        authorization.get("test_evidence"),
        authorization_created=auth_created,
        bound_identities=evidence_bound,
    )
    _require(
        canonical_payload_sha256(authorization["test_evidence"])
        == FROZEN_V5_TEST_EVIDENCE_CANONICAL_SHA256
        and review.get("independent_checks")
        == {key: True for key in V5_REVIEW_CHECKS}
        and review.get("test_evidence_recheck", {}).get(
            "authorization_test_evidence_canonical_sha256"
        ) == FROZEN_V5_TEST_EVIDENCE_CANONICAL_SHA256,
        "frozen V5 evidence/review mismatch",
    )
    common_fields = (
        "incident", "incident_correction", "v4_authorization_base",
        "v4_review_transport_truncation_state", *FROZEN_V4_SOURCE_IDENTITIES.keys(),
        "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
        "recovered_afterstate_commitment",
    )
    _require(
        all(
            review.get(field) == authorization.get(field)
            and go.get(field) == authorization.get(field)
            for field in common_fields
        ),
        "frozen V5 authorization/review/GO semantic cross-binding mismatch",
    )
    expected_base = {
        "authorization": FROZEN_V5_CONTROL_IDENTITIES["authorization"],
        "independent_review": FROZEN_V5_CONTROL_IDENTITIES["independent_review"],
        "independent_go": FROZEN_V5_CONTROL_IDENTITIES["independent_go"],
        "audit_attempt_id": FROZEN_V5_AUDIT_ATTEMPT_ID,
        "recovery_attempt_id": FROZEN_V5_RECOVERY_ATTEMPT_ID,
        "authorization_status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5",
        "review_status": "PASS_PENDING_INDEPENDENT_GO_V5",
        "go_status": "GO_V5_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
        "recovered_afterstate_canonical_sha256": V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
    }
    _require(base == expected_base, "V6 frozen V5 authority-base value mismatch")
    return {
        "base": expected_base,
        "authorization": authorization,
        "review": review,
        "go": go,
        "v4_base": v4_base,
        "recovered_afterstate": v4_base["afterstate"],
        "commitment": v4_base["commitment"],
        "authorization_created": auth_created,
        "review_created": review_created,
        "go_created": go_created,
        "frozen_v5_sources": frozen_v5_actual,
    }


def _v6_validate_zero_snapshot(root: Path, payload: Any) -> None:
    _require(
        isinstance(payload, Mapping) and set(payload) == V6_ZERO_SNAPSHOT_KEYS,
        "V6 zero-mutation snapshot schema mismatch",
    )
    _require(
        payload.get("incident_postfailure_state_canonical_sha256")
        == "866d6564abb4d00cb1cb2961677cf767b9934ba44829cee3fec0ff46e9961690"
        and payload.get("recovered_afterstate_commitment_canonical_sha256")
        == V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        and payload.get("active_locks")
        == {
            "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
            "decoded": {"path": "decoded/DECODE_RECOVERY_ACTIVE.lock", "present": False},
        }
        and payload.get("v6_control_temporary_files")
        == {
            "checked_destination_relative_paths": list(
                _v6_control_destination_relatives()
            ),
            "matching_temporary_files": [],
        }
        and payload.get("postrun_audit_output_files_present") == 0
        and payload.get("network_requests") == 0
        and payload.get("audit_files_written") == 0
        and payload.get("labels_read") is False
        and payload.get("arrays_2024_read") is False
        and payload.get("arrays_2025_read") is False
        and payload.get("models_fit") == 0
        and payload.get("submission_csv_created") is False,
        "V6 zero-mutation snapshot value mismatch",
    )
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock")))
        and _v6_control_temporary_files(root) == []
        and _v6_postrun_report_candidates(root) == [],
        "V6 current zero-mutation filesystem mismatch",
    )


def _v6_validate_test_evidence(
    evidence: Any,
    *,
    authorization_created: datetime,
    bound_identities: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(
        isinstance(evidence, Mapping) and set(evidence) == V6_TEST_EVIDENCE_KEYS,
        "V6 immutable test-evidence schema mismatch",
    )
    created = _v4_created_utc(evidence.get("created_utc"), label="V6 test evidence")
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_TEST_EVIDENCE"
        and evidence.get("status") == "PASS_FROZEN_V6_AUDITOR_AND_SEALER_TESTS"
        and created <= authorization_created
        and evidence.get("bound_identities") == bound_identities
        and evidence.get("pytest_isolation")
        == "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS"
        and evidence.get("required_test_names") == list(V6_REQUIRED_TEST_NAMES)
        and evidence.get("required_test_names_present") is True
        and evidence.get("all_exit_codes_zero") is True,
        "V6 immutable test-evidence header/binding mismatch",
    )
    compile_record = evidence.get("source_compile")
    _require(
        isinstance(compile_record, Mapping)
        and set(compile_record) == {"method", "result", "files"}
        and compile_record.get("method") == "compile_exact_source_no_pyc"
        and compile_record.get("result") == "PASS"
        and compile_record.get("files")
        == [
            bound_identities["v6_auditor"],
            bound_identities["v6_auditor_test"],
            bound_identities["v6_sealer"],
            bound_identities["v6_sealer_test"],
        ],
        "V6 exact-source compile evidence mismatch",
    )
    expected_commands = {
        "v6_auditor_tests": _v6_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py"
        ),
        "v6_sealer_tests": _v6_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py"
        ),
        "frozen_v5_auditor_tests": _v6_test_command(
            "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py"
        ),
        "frozen_v5_sealer_tests": _v6_test_command(
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py"
        ),
    }
    runs = evidence.get("test_runs")
    _require(
        isinstance(runs, Mapping) and set(runs) == V6_TEST_RUN_KEYS,
        "V6 immutable pytest run map mismatch",
    )
    for role, expected_command in expected_commands.items():
        run = runs[role]
        _require(
            isinstance(run, Mapping)
            and set(run) == V4_TEST_RUN_RECORD_KEYS
            and run.get("command") == expected_command
            and run.get("exit_code") == 0
            and isinstance(run.get("summary"), str)
            and "passed" in str(run["summary"])
            and "failed" not in str(run["summary"]).casefold()
            and "error" not in str(run["summary"]).casefold()
            and isinstance(run.get("stdout_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stdout_sha256"])) is not None
            and isinstance(run.get("stderr_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(run["stderr_sha256"])) is not None
            and run.get("network_guard_installed") is True
            and run.get("cacheprovider_disabled") is True,
            f"V6 immutable pytest run mismatch: {role}",
        )
    _require(
        evidence.get("production_shape_regression")
        == {
            "actual_root_historical_predata_validated": True,
            "compact_afterstate_commitment_validated": True,
            "decoded_reached": True,
            "direct_file_failure_incident_9e74_validated_predata": True,
            "full_afterstate_reconstructed_in_memory": True,
            "historical_current_role_distinction_validated": True,
            "offline_replay_reached": True,
            "original_v1_four_key_identity_passed": True,
            "raw_reached": True,
            "synthetic_fixture": True,
            "transaction_reached": True,
            "v5_authority_base_validated": True,
            "v5_false_reject_incident_validated": True,
            "v5_production_data_and_replay_core_ast_identical": True,
        }
        and evidence.get("real_seven_spawn_regression")
        == {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        }
        and evidence.get("stdout_only_regression") is True
        and evidence.get("network_guard")
        == {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        }
        and evidence.get("network_requests") == 0
        and evidence.get("audit_files_written") == 0
        and evidence.get("labels_read") is False
        and evidence.get("arrays_2024_read") is False
        and evidence.get("arrays_2025_read") is False
        and evidence.get("models_fit") == 0
        and evidence.get("submission_csv_created") is False,
        "V6 immutable test-evidence safety/regression mismatch",
    )


def _v6_assert_compact_control(payload: Mapping[str, Any], *, label: str) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            _require(
                "inventory" not in value
                and "recovered_afterstate" not in value
                and "recovery_progress" not in value,
                f"{label} embeds a forbidden full-afterstate field",
            )
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            _require(len(value) != 104, f"{label} embeds a forbidden 104-entry list")
            for nested in value:
                walk(nested)

    walk(payload)


def _v6_validate_predata_authority(
    root: Path, authorization_path: Path, go_path: Path
) -> dict[str, Any]:
    """Close V6 authority and V1 historical literals before any data dereference."""

    expected_authorization_path = path_under(root, V6_AUTH_RELATIVE)
    expected_review_path = path_under(root, V6_REVIEW_RELATIVE)
    expected_go_path = path_under(root, V6_GO_RELATIVE)
    _require(
        Path(os.path.abspath(authorization_path)) == expected_authorization_path
        and Path(os.path.abspath(go_path)) == expected_go_path,
        "V6 authorization/GO paths are not canonical",
    )
    for path, label in (
        (expected_authorization_path, "V6 authorization"),
        (expected_review_path, "V6 independent review"),
        (expected_go_path, "V6 independent GO"),
    ):
        require_no_symlink_chain(path, root, label=label)
        _require(
            os.path.lexists(str(path)) and path.is_file() and not _linklike(path),
            f"{label} is absent, non-regular or link-like",
        )
    _require(
        expected_review_path.stat().st_size <= 12_000
        and expected_go_path.stat().st_size <= 12_000,
        "V6 compact review or GO exceeds the 12KB transport limit",
    )
    namespace_start = _v6_validate_control_namespace(root)
    _require(_v6_control_temporary_files(root) == [], "V6 control temporary files present")
    _require(
        _v6_postrun_report_candidates(root) == [],
        "persisted V2/V3/V4/V5/V6 postrun report candidate is forbidden",
    )
    authorization = _v5_load_strict_json(
        expected_authorization_path, label="V6 authorization"
    )
    review = _v5_load_strict_json(expected_review_path, label="V6 independent review")
    go = _v5_load_strict_json(expected_go_path, label="V6 independent GO")
    _require(set(authorization) == V6_AUTH_KEYS, "V6 authorization schema mismatch")
    _require(set(review) == V6_REVIEW_KEYS, "V6 independent review schema mismatch")
    _require(set(go) == V6_GO_KEYS, "V6 independent GO schema mismatch")
    for payload, label in (
        (authorization, "V6 authorization"),
        (review, "V6 independent review"),
        (go, "V6 independent GO"),
    ):
        _v6_assert_compact_control(payload, label=label)
    authorization_created = _v4_created_utc(
        authorization.get("created_utc"), label="V6 authorization"
    )
    review_created = _v4_created_utc(review.get("created_utc"), label="V6 review")
    go_created = _v4_created_utc(go.get("created_utc"), label="V6 GO")
    attempt_id = authorization.get("audit_attempt_id")
    derived_attempt = "postrun_audit_v6__" + str(authorization["created_utc"]).translate(
        str.maketrans("", "", "-:.")
    )
    _require(
        authorization.get("schema_version") == 6
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V6"
        and authorization.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V6"
        and isinstance(attempt_id, str)
        and re.fullmatch(r"postrun_audit_v6__[0-9]{8}T[0-9]{12}Z", attempt_id)
        is not None
        and attempt_id == derived_attempt
        and authorization.get("recovery_attempt_id") == FROZEN_V5_RECOVERY_ATTEMPT_ID
        and authorization.get("runtime_identity_sha256") == V4_RUNTIME_IDENTITY_SHA256
        and authorization.get("independent_review_required") is True
        and authorization.get("independent_go_required") is True,
        "V6 authorization header/binding mismatch",
    )
    _v6_validate_policy(authorization, label="V6 authorization")

    # This exact historical contract is intentionally first among semantic
    # bindings: it performs only incident JSON/identity work, before V5 base
    # reconstruction, canonical-output rehash, schema inspection, or replay.
    false_reject = _v6_validate_v5_false_reject_incident(
        root, authorization.get("incident")
    )
    direct_path = path_under(root, RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE)
    direct_identity = identity(direct_path, root)
    direct_payload = _v5_load_strict_json(
        direct_path, label="immutable historical direct-file failure incident"
    )
    historical_contract = _v6_validate_historical_9e74_payload(
        direct_payload, direct_identity
    )
    expected_historical_contract = _v6_expected_historical_contract(direct_identity)
    _require(
        historical_contract == expected_historical_contract,
        "V6 internally reconstructed historical contract mismatch",
    )
    _v6_validate_historical_contract(
        authorization.get("historical_failed_head_identity_contract"),
        expected_historical_contract,
    )

    v5_base = _v6_load_and_validate_v5_authority_base(
        root, authorization.get("v5_authority_base")
    )
    _require(
        v5_base["go_created"] < false_reject["created"] < authorization_created,
        "V5 GO/false-reject/V6 authorization chronology mismatch",
    )
    _require(
        authorization.get("recovered_afterstate_commitment")
        == v5_base["commitment"]
        == V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "V6 recovered-afterstate commitment mismatch",
    )
    frozen_v5_actual: dict[str, dict[str, Any]] = {}
    for role, expected in FROZEN_V5_SOURCE_IDENTITIES.items():
        path = {
            "superseded_v5_auditor": FROZEN_V5_AUDITOR,
            "superseded_v5_auditor_test": FROZEN_V5_AUDITOR_TEST,
            "superseded_v5_sealer": FROZEN_V5_SEALER,
            "superseded_v5_sealer_test": FROZEN_V5_SEALER_TEST,
        }[role]
        _v4_require_exact_declared_identity(
            authorization.get(role), expected, label=f"V6 {role}"
        )
        frozen_v5_actual[role] = _v4_verify_external_identity(
            authorization[role], path, label=f"V6 {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    v6_paths = {
        "v6_auditor": Path(__file__).resolve(),
        "v6_auditor_test": AUDITOR_TEST,
        "v6_sealer": V6_SEALER,
        "v6_sealer_test": V6_SEALER_TEST,
    }
    v6_actual = {
        role: _v4_verify_external_identity(
            authorization.get(role), path, label=f"V6 executing {role}"
        )
        for role, path in v6_paths.items()
    }
    _v6_validate_zero_snapshot(
        root, authorization.get("preaudit_zero_mutation_snapshot")
    )
    _require(
        authorization.get("required_command")
        == _v6_required_command(root, expected_authorization_path, expected_go_path),
        "V6 authorization required command mismatch",
    )
    evidence_bound = {
        "v5_authorization": FROZEN_V5_CONTROL_IDENTITIES["authorization"],
        "v5_independent_review": FROZEN_V5_CONTROL_IDENTITIES["independent_review"],
        "v5_independent_go": FROZEN_V5_CONTROL_IDENTITIES["independent_go"],
        "v5_historical_identity_false_reject_incident": false_reject["identity"],
        "direct_file_failure_incident": direct_identity,
        **frozen_v5_actual,
        **v6_actual,
    }
    _v6_validate_test_evidence(
        authorization.get("test_evidence"),
        authorization_created=authorization_created,
        bound_identities=evidence_bound,
    )
    _require(
        review.get("schema_version") == 6
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V6"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V6"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == attempt_id
        and review.get("auditor_execution_started") is False
        and review.get("independent_go_required") is True
        and review_created > authorization_created,
        "V6 independent review header/verdict mismatch",
    )
    _v6_validate_policy(review, label="V6 independent review")
    _require(
        go.get("schema_version") == 6
        and go.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V6"
        and go.get("status") == "GO_V6_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY"
        and go.get("audit_attempt_id") == attempt_id
        and go.get("recovery_attempt_id") == FROZEN_V5_RECOVERY_ATTEMPT_ID
        and go.get("recovery_rerun_authorized") is False
        and go.get("required_command") == authorization.get("required_command")
        and go_created > review_created,
        "V6 independent GO header/binding mismatch",
    )
    _v6_validate_policy(go, label="V6 independent GO")
    authorization_actual = identity(expected_authorization_path, root)
    review_actual = identity(expected_review_path, root)
    common_fields = (
        "incident", "v5_authority_base", "historical_failed_head_identity_contract",
        *FROZEN_V5_SOURCE_IDENTITIES.keys(), "v6_auditor", "v6_auditor_test",
        "v6_sealer", "v6_sealer_test", "recovered_afterstate_commitment",
    )
    _require(
        review.get("authorization") == authorization_actual
        and go.get("authorization") == authorization_actual
        and go.get("independent_review") == review_actual
        and all(
            review.get(field) == authorization.get(field)
            and go.get(field) == authorization.get(field)
            for field in common_fields
        ),
        "V6 authorization/review/GO cross-binding mismatch",
    )
    checks = review.get("independent_checks")
    recheck = review.get("test_evidence_recheck")
    _require(
        isinstance(checks, Mapping)
        and set(checks) == V6_REVIEW_CHECKS
        and all(checks.get(key) is True for key in V6_REVIEW_CHECKS),
        "V6 independent review check set/verdict mismatch",
    )
    _require(
        isinstance(recheck, Mapping)
        and set(recheck) == V4_REVIEW_TEST_RECHECK_KEYS
        and recheck.get("authorization_test_evidence_canonical_sha256")
        == canonical_payload_sha256(authorization["test_evidence"])
        and all(
            recheck.get(key) is True
            for key in V4_REVIEW_TEST_RECHECK_KEYS
            if key != "authorization_test_evidence_canonical_sha256"
        ),
        "V6 independent test-evidence recheck mismatch",
    )
    _require(
        _v6_validate_control_namespace(root) == namespace_start
        and _v6_control_temporary_files(root) == []
        and _v6_postrun_report_candidates(root) == [],
        "V6 predata control namespace changed",
    )
    return {
        "authorization": authorization,
        "authorization_identity": authorization_actual,
        "review": review,
        "review_identity": review_actual,
        "go": go,
        "go_identity": identity(expected_go_path, root),
        "incident": false_reject,
        "historical_contract": historical_contract,
        "v5_base": v5_base,
        "recovered_afterstate": v5_base["recovered_afterstate"],
        "recovered_afterstate_commitment": v5_base["commitment"],
        "frozen_v5_source_identities": frozen_v5_actual,
        "v6_source_identities": v6_actual,
        "control_namespace": namespace_start,
    }


def _v6_validate_postauthority_control_state(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = authority["authorization"]
    afterstate = authority["recovered_afterstate"]
    _require(
        not os.path.lexists(str(path_under(root, "raw/RAW_LAUNCH_ACTIVE.lock")))
        and not os.path.lexists(str(path_under(root, "decoded/DECODE_RECOVERY_ACTIVE.lock"))),
        "V6 recovered afterstate has an active lock",
    )
    commitment = _v5_recovered_afterstate_commitment(afterstate)
    _require(
        commitment == authority["recovered_afterstate_commitment"]
        and _v6_control_temporary_files(root) == []
        and _v6_postrun_report_candidates(root) == [],
        "V6 afterstate/commitment/temp/report recheck mismatch",
    )
    incident = _v6_validate_v5_false_reject_incident(root, authorization["incident"])
    _require(
        incident["identity"] == authority["incident"]["identity"],
        "V6 false-reject incident changed during audit",
    )
    direct_path = path_under(root, RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE)
    direct_identity = identity(direct_path, root)
    historical = _v6_validate_historical_9e74_payload(
        _v5_load_strict_json(direct_path, label="postaudit historical 9e74"),
        direct_identity,
    )
    _v6_validate_historical_contract(
        authorization["historical_failed_head_identity_contract"], historical
    )
    v5_base = _v6_load_and_validate_v5_authority_base(
        root, authorization["v5_authority_base"]
    )
    _require(
        v5_base["recovered_afterstate"] == afterstate
        and v5_base["commitment"] == commitment,
        "V6 reconstructed V5 afterstate changed during audit",
    )
    for role, record, relative in (
        ("authorization", authority["authorization_identity"], V6_AUTH_RELATIVE),
        ("independent review", authority["review_identity"], V6_REVIEW_RELATIVE),
        ("independent GO", authority["go_identity"], V6_GO_RELATIVE),
    ):
        verify_identity(
            record, path_under(root, relative), root=root, label=f"V6 postaudit {role}"
        )
    for role, path in (
        ("v6_auditor", Path(__file__).resolve()),
        ("v6_auditor_test", AUDITOR_TEST),
        ("v6_sealer", V6_SEALER),
        ("v6_sealer_test", V6_SEALER_TEST),
    ):
        _v4_verify_external_identity(
            authorization[role], path, label=f"V6 postaudit {role}"
        )
    for role, expected in FROZEN_V5_SOURCE_IDENTITIES.items():
        path = {
            "superseded_v5_auditor": FROZEN_V5_AUDITOR,
            "superseded_v5_auditor_test": FROZEN_V5_AUDITOR_TEST,
            "superseded_v5_sealer": FROZEN_V5_SEALER,
            "superseded_v5_sealer_test": FROZEN_V5_SEALER_TEST,
        }[role]
        _v4_verify_external_identity(
            authorization[role], path, label=f"V6 postaudit {role}",
            expected_size_sha256=(int(expected["size_bytes"]), str(expected["sha256"])),
        )
    _v6_validate_zero_snapshot(root, authorization["preaudit_zero_mutation_snapshot"])
    _require(
        _v6_validate_control_namespace(root) == authority["control_namespace"],
        "V6 postauthority control namespace changed",
    )
    return commitment


def _v6_validate_postauthority_afterstate(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    v4_authorization = authority["v5_base"]["v4_base"]["authorization"]
    afterstate = authority["recovered_afterstate"]
    commitment = _v6_validate_postauthority_control_state(root, authority)
    for role, record in v4_authorization["canonical_outputs"].items():
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V6 inherited canonical output {role}",
        )
    for role, record in v4_authorization["recovery_transaction"].items():
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V6 inherited recovery transaction {role}",
        )
    history = v4_authorization["recovery_history"]
    for role, record in history["completion_locks"].items():
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V6 inherited recovery completion lock {role}",
        )
    verify_identity(
        history["postcommit_input_audit"],
        path_under(root, str(history["postcommit_input_audit"]["path"])),
        root=root, label="V6 inherited recovery postcommit input audit",
    )
    history_dir = path_under(root, "decoded/recovery_history")
    expected_history_paths = {
        str(record["path"]) for record in history["completion_locks"].values()
    } | {str(history["postcommit_input_audit"]["path"])}
    actual_history_entries = list(history_dir.iterdir())
    _require(
        history_dir.is_dir() and not _linklike(history_dir)
        and len(actual_history_entries) == 3
        and all(path.is_file() and not _linklike(path) for path in actual_history_entries)
        and {path.relative_to(root).as_posix() for path in actual_history_entries}
        == expected_history_paths,
        "V6 recovery-history exact filesystem closure mismatch",
    )
    transaction_root = path_under(root, "raw/output_transactions")
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "V6 output-transaction root invalid",
    )
    recovery_authorization_v2 = load_json(
        path_under(root, V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"])
    )
    remnants = recovery_authorization_v2.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list) and len(remnants) == 2,
        "V6 incident-bound preplan-remnant declaration mismatch",
    )
    expected_transaction_files = {
        str(v4_authorization["recovery_transaction"]["plan"]["path"]),
        str(v4_authorization["recovery_transaction"]["commit"]["path"]),
    }
    for index, record in enumerate(remnants):
        _require(
            isinstance(record, Mapping) and set(record) == {"path", "size_bytes", "sha256"},
            f"V6 preplan remnant identity invalid: {index}",
        )
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V6 incident-bound preplan remnant {index}",
        )
        expected_transaction_files.add(str(record["path"]))
    expected_transaction_dirs = {
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged/raw",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    actual_transaction_files: set[str] = set()
    actual_transaction_dirs: set[str] = set()
    for path in transaction_root.rglob("*"):
        _require(not _linklike(path), "V6 transaction tree contains a link/junction")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            actual_transaction_files.add(relative)
        elif path.is_dir():
            actual_transaction_dirs.add(relative)
        else:
            raise AuditFailure("V6 transaction tree contains a special object")
    _require(
        actual_transaction_files == expected_transaction_files
        and actual_transaction_dirs == expected_transaction_dirs
        and len(actual_transaction_files) == 4,
        "V6 transaction recursive exact filesystem closure mismatch",
    )
    progress_dir = path_under(root, "decoded/recovery_progress")
    actual_progress_paths = sorted(progress_dir.iterdir(), key=lambda path: path.name)
    _require(
        progress_dir.is_dir() and not _linklike(progress_dir)
        and len(actual_progress_paths) == 104
        and all(path.is_file() and not _linklike(path) for path in actual_progress_paths),
        "V6 recovery-progress filesystem closure mismatch",
    )
    actual_inventory = [identity(path, root) for path in actual_progress_paths]
    inventory_bytes = _v4_canonical_bytes(actual_inventory)
    _require(
        actual_inventory == v4_authorization["recovery_progress"]["inventory"]
        and len(inventory_bytes) == V4_PROGRESS_INVENTORY_CANONICAL_BYTES
        and hashlib.sha256(inventory_bytes).hexdigest()
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "V6 recovery-progress inventory/canonical digest mismatch",
    )
    _require(
        _v6_validate_postauthority_control_state(root, authority) == commitment,
        "V6 postauthority compact control state changed",
    )
    return {
        "recovered_afterstate_commitment": commitment,
        "recovered_afterstate_canonical_sha256": V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
        "canonical_output_count": 8,
        "transaction_recursive_file_count": 4,
        "transaction_recursive_directory_count": 7,
        "recovery_history_file_count": 3,
        "recovery_progress_file_count": 104,
        "active_locks_present": 0,
        "v6_control_temporary_files_present": 0,
        "canonical_v4_go_present": 0,
        "historical_9e74_rehashed": True,
        "v5_false_reject_incident_rehashed": True,
        "control_namespace_exact": True,
    }


def _v7_validate_postauthority_afterstate(
    root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    v4_authorization = authority["v5_base"]["v4_base"]["authorization"]
    afterstate = authority["recovered_afterstate"]
    commitment = _v7_validate_postauthority_control_state(root, authority)
    for role, record in v4_authorization["canonical_outputs"].items():
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V7 inherited canonical output {role}",
        )
    for role, record in v4_authorization["recovery_transaction"].items():
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V7 inherited recovery transaction {role}",
        )
    history = v4_authorization["recovery_history"]
    for role, record in history["completion_locks"].items():
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V7 inherited recovery completion lock {role}",
        )
    verify_identity(
        history["postcommit_input_audit"],
        path_under(root, str(history["postcommit_input_audit"]["path"])),
        root=root, label="V7 inherited recovery postcommit input audit",
    )
    history_dir = path_under(root, "decoded/recovery_history")
    expected_history_paths = {
        str(record["path"]) for record in history["completion_locks"].values()
    } | {str(history["postcommit_input_audit"]["path"])}
    actual_history_entries = list(history_dir.iterdir())
    _require(
        history_dir.is_dir() and not _linklike(history_dir)
        and len(actual_history_entries) == 3
        and all(path.is_file() and not _linklike(path) for path in actual_history_entries)
        and {path.relative_to(root).as_posix() for path in actual_history_entries}
        == expected_history_paths,
        "V7 recovery-history exact filesystem closure mismatch",
    )
    transaction_root = path_under(root, "raw/output_transactions")
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "V7 output-transaction root invalid",
    )
    recovery_authorization_v2 = load_json(
        path_under(root, V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"])
    )
    remnants = recovery_authorization_v2.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list) and len(remnants) == 2,
        "V7 incident-bound preplan-remnant declaration mismatch",
    )
    expected_transaction_files = {
        str(v4_authorization["recovery_transaction"]["plan"]["path"]),
        str(v4_authorization["recovery_transaction"]["commit"]["path"]),
    }
    for index, record in enumerate(remnants):
        _require(
            isinstance(record, Mapping) and set(record) == {"path", "size_bytes", "sha256"},
            f"V7 preplan remnant identity invalid: {index}",
        )
        verify_identity(
            record, path_under(root, str(record["path"])), root=root,
            label=f"V7 incident-bound preplan remnant {index}",
        )
        expected_transaction_files.add(str(record["path"]))
    expected_transaction_dirs = {
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged/raw",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    actual_transaction_files: set[str] = set()
    actual_transaction_dirs: set[str] = set()
    for path in transaction_root.rglob("*"):
        _require(not _linklike(path), "V7 transaction tree contains a link/junction")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            actual_transaction_files.add(relative)
        elif path.is_dir():
            actual_transaction_dirs.add(relative)
        else:
            raise AuditFailure("V7 transaction tree contains a special object")
    _require(
        actual_transaction_files == expected_transaction_files
        and actual_transaction_dirs == expected_transaction_dirs
        and len(actual_transaction_files) == 4,
        "V7 transaction recursive exact filesystem closure mismatch",
    )
    progress_dir = path_under(root, "decoded/recovery_progress")
    actual_progress_paths = sorted(progress_dir.iterdir(), key=lambda path: path.name)
    _require(
        progress_dir.is_dir() and not _linklike(progress_dir)
        and len(actual_progress_paths) == 104
        and all(path.is_file() and not _linklike(path) for path in actual_progress_paths),
        "V7 recovery-progress filesystem closure mismatch",
    )
    actual_inventory = [identity(path, root) for path in actual_progress_paths]
    inventory_bytes = _v4_canonical_bytes(actual_inventory)
    _require(
        actual_inventory == v4_authorization["recovery_progress"]["inventory"]
        and len(inventory_bytes) == V4_PROGRESS_INVENTORY_CANONICAL_BYTES
        and hashlib.sha256(inventory_bytes).hexdigest()
        == V4_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "V7 recovery-progress inventory/canonical digest mismatch",
    )
    _require(
        _v7_validate_postauthority_control_state(root, authority) == commitment,
        "V7 postauthority compact control state changed",
    )
    return {
        "recovered_afterstate_commitment": commitment,
        "recovered_afterstate_canonical_sha256": V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
        "canonical_output_count": 8,
        "transaction_recursive_file_count": 4,
        "transaction_recursive_directory_count": 7,
        "recovery_history_file_count": 3,
        "recovery_progress_file_count": 104,
        "active_locks_present": 0,
        "v7_control_temporary_files_present": 0,
        "canonical_v4_go_present": 0,
        "historical_9e74_rehashed": True,
        "v5_false_reject_incident_rehashed": True,
        "control_namespace_exact": True,
    }


def audit(
    root: Path,
    *,
    contract: AuditContract = PRODUCTION_CONTRACT,
    runner_path: Path = RUNNER_DEFAULT,
    authorization_path: Path | None = None,
    independent_go_path: Path | None = None,
    _test_message_decoder: Any | None = None,
) -> dict[str, Any]:
    """Audit a quiescent producer tree without changing it or using a network."""

    _require(not _linklike(root), "producer root link is forbidden")
    _require(not _linklike(runner_path), "runner path link is forbidden")
    root = root.resolve()
    runner_path = runner_path.resolve()
    _require(
        not (contract == PRODUCTION_CONTRACT and _test_message_decoder is not None),
        "production audit forbids decoder override",
    )
    _require(root.is_dir(), f"producer root is absent: {root}")
    if contract == PRODUCTION_CONTRACT:
        _require(
            _V7_CANONICAL_MAIN_ACTIVE,
            "V7 production audit is callable only from the canonical guarded main",
        )
    authority: dict[str, Any] | None = None
    v7_afterstate_start: dict[str, Any] | None = None
    v7_afterstate_end: dict[str, Any] | None = None
    if contract == PRODUCTION_CONTRACT or authorization_path is not None or independent_go_path is not None:
        _require(
            authorization_path is not None and independent_go_path is not None,
            "V7 production audit requires authorization and independent GO",
        )
        authority = _v7_validate_predata_authority(
            root, authorization_path.resolve(), independent_go_path.resolve()
        )
        v7_afterstate_start = _v7_validate_postauthority_afterstate(root, authority)
    require_no_symlink_chain(root / "raw", root, label="raw control path")
    _audit_active_lock(root)
    control_temporary_gate_start = (
        _audit_recovery_control_temporary_files_absent(root)
    )
    schema_preflight = preflight_zero_target_parquet_schemas(root, contract)
    provenance = _audit_provenance(root, runner_path, contract)
    provenance["runner_path"] = str(runner_path)
    transaction = _audit_transaction_and_launch(root, contract)
    census_frame, census_rows = _audit_census(provenance["field_census_path"], contract)
    raw = _audit_raw_ranges(
        root,
        census_rows,
        transaction["outputs"]["raw_parquet"],
        transaction["outputs"]["raw_csv"],
        contract,
    )
    events = _audit_request_events(root, raw["raw_info"], contract)
    decoded = _audit_decoded(
        root,
        census_frame,
        provenance["coordinate"],
        transaction["outputs"],
        contract,
    )
    replay = _audit_full_offline_redecode(
        raw["raw_info"],
        provenance["coordinate"],
        transaction["outputs"],
        contract,
        decoder=_test_message_decoder or decode_offline_grib_message,
        synthetic_decoder_override=_test_message_decoder is not None,
        expected_runtime_identity_sha256=provenance["runtime_identity_sha256"],
    )
    access = _audit_canonical_metadata(
        root, provenance, transaction, events, decoded, raw["raw_info"], contract
    )
    external_identity_unique_file_count = access.get(
        "external_identity_unique_file_count",
        provenance["external_identity_unique_file_count"],
    )
    external_identity_unique_file_bytes = access.get(
        "external_identity_unique_file_bytes",
        provenance["external_identity_unique_file_bytes"],
    )
    _audit_active_lock(root)
    control_temporary_gate_end = (
        _audit_recovery_control_temporary_files_absent(root)
    )
    if authority is not None:
        v7_afterstate_end = _v7_validate_postauthority_afterstate(root, authority)
        _require(
            v7_afterstate_end == v7_afterstate_start,
            "V7 recovered afterstate commitment changed during full audit",
        )
    return {
        "artifact_type": "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V7",
        "schema_version": 7,
        "status": "PASS_PRODUCER_POSTRUN_READONLY_AUDIT_V7",
        "audit_mode": "OFFLINE_READ_ONLY_NO_NETWORK",
        "root": str(root),
        "provenance": {
            "authorization": provenance["authorization_identity"],
            "independent_go": provenance["go_identity"],
            "effective_preflight_manifest": provenance["preflight_identity"],
            "effective_preflight_amendment": provenance["preflight_amendment_identity"],
            "independent_prelaunch_audit": provenance["independent_prelaunch_identity"],
            "independent_prelaunch_audit_base_identity": provenance[
                "independent_prelaunch_identity_base"
            ],
            "runner": provenance["runner_identity"],
            "runner_test": provenance["runner_test_identity"],
            "runtime_identity_sha256": provenance["runtime_identity_sha256"],
            "embedded_identity_rehashes": provenance["embedded_identity_rehashes"],
            "external_identity_unique_file_count": external_identity_unique_file_count,
            "external_identity_unique_file_bytes": external_identity_unique_file_bytes,
            "auditor_source": identity(Path(__file__).resolve()),
            "auditor_test": identity(AUDITOR_TEST),
            "executing_v7_auditor": (
                authority["v7_source_identities"]["v7_auditor"]
                if authority is not None
                else identity(Path(__file__).resolve())
            ),
            "executing_v7_auditor_test": (
                authority["v7_source_identities"]["v7_auditor_test"]
                if authority is not None
                else identity(AUDITOR_TEST)
            ),
            "executing_v7_sealer": (
                authority["v7_source_identities"]["v7_sealer"]
                if authority is not None else None
            ),
            "executing_v7_sealer_test": (
                authority["v7_source_identities"]["v7_sealer_test"]
                if authority is not None else None
            ),
            "frozen_v2_auditor_baseline": V4_SUPERSEDED_V2_AUDITOR_IDENTITY,
            "frozen_v2_auditor_test_baseline": V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY,
            "frozen_v6_source_baseline": (
                authority["frozen_v6_source_identities"]
                if authority is not None else FROZEN_V6_SOURCE_IDENTITIES
            ),
            "v7_audit_authorization": (
                authority["authorization_identity"] if authority is not None else None
            ),
            "v7_independent_review": (
                authority["review_identity"] if authority is not None else None
            ),
            "v7_independent_go": (
                authority["go_identity"] if authority is not None else None
            ),
            "v5_authority_base": (
                authority["v5_base"]["base"] if authority is not None else None
            ),
            "v5_historical_identity_false_reject_incident": (
                authority["false_reject_incident"]["identity"]
                if authority is not None else None
            ),
            "v6_auth_sealer_timestamp_precision_false_reject_incident": (
                authority["timestamp_incident"]["identity"]
                if authority is not None else None
            ),
            "historical_direct_file_failure_incident": (
                authority["historical_contract"]["incident"]
                if authority is not None else None
            ),
            "historical_failed_head_identity_contract": (
                authority["historical_contract"]
                if authority is not None else None
            ),
            "recovered_afterstate_commitment": (
                authority["recovered_afterstate_commitment"]
                if authority is not None else None
            ),
            "zero_target_schema_preflight": {
                name: list(columns) for name, columns in schema_preflight.items()
            },
        },
        "launch_and_transaction": {
            "active_lock_absent_at_start_and_end": True,
            "recovery_control_temporary_file_gate": {
                "start": control_temporary_gate_start,
                "end": control_temporary_gate_end,
            },
            "producer_attempt_id": transaction["attempt_id"],
            "matching_complete_lock": transaction["complete_lock_identity"],
            "launch_history_lock_count": transaction["launch_history_lock_count"],
            "transaction_plan": transaction["plan_identity"],
            "transaction_commit": transaction["commit_identity"],
            "canonical_outputs_rehashed": len(OUTPUT_RELATIVE_PATHS),
            "transaction_inventory_recursively_closed": transaction[
                "transaction_inventory_recursively_closed"
            ],
            "planned_staged_files_present": transaction[
                "planned_staged_files_present"
            ],
        },
        "raw_closure": {
            "objects": contract.object_rows,
            "ranges": raw["raw_rows"],
            "bytes": raw["raw_bytes"],
            "sidecars": raw["raw_rows"],
            "request_attempt_starts": events["starts"],
            "request_attempt_completions": events["completions"],
            "request_attempt_errors": events["errors"],
            "request_attempts_without_terminal_event": events["indeterminate_starts"],
            "max_global_attempt_number": events["max_global_attempt_number"],
        },
        "decoded_closure": decoded,
        "full_offline_raw_redecode": replay,
        "access_and_zero_target_facts": access,
        "audit_network_requests": 0,
        "external_network_bytes": 0,
        "external_data_array_bytes": 0,
        "external_identity_unique_file_count": external_identity_unique_file_count,
        "external_identity_unique_file_bytes": external_identity_unique_file_bytes,
        "audit_labels_read": False,
        "audit_models_fit": 0,
        "audit_files_written": 0,
        "v7_authorized_afterstate_recheck": {
            "start": v7_afterstate_start,
            "end": v7_afterstate_end,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--independent-go", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _V7_CANONICAL_MAIN_ACTIVE
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        _require(
            args.authorization is not None and args.independent_go is not None,
            "V7 canonical command requires explicit --authorization and --independent-go",
        )
        authorization_path = args.authorization.resolve()
        independent_go_path = args.independent_go.resolve()
        _v7_require_canonical_runtime_entrypoint(
            root, authorization_path, independent_go_path
        )
        _V7_CANONICAL_MAIN_ACTIVE = True
        report = audit(
            root,
            contract=PRODUCTION_CONTRACT,
            runner_path=RUNNER_DEFAULT,
            authorization_path=authorization_path,
            independent_go_path=independent_go_path,
        )
    except AuditNotReady as exc:
        print(
            json.dumps(
                {
                    "artifact_type": "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V7",
                    "schema_version": 7,
                    "status": "CONDITIONAL_SKIP_V7_POSTRUN_AUDIT_ACTIVE",
                    "reason": str(exc),
                    "active_lock": exc.active_lock,
                    "audit_network_requests": 0,
                    "audit_files_written": 0,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "artifact_type": "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V7",
                    "schema_version": 7,
                    "status": "FAIL_PRODUCER_POSTRUN_AUDIT_V7",
                    "reason": str(exc),
                    "audit_network_requests": 0,
                    "audit_files_written": 0,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    finally:
        _V7_CANONICAL_MAIN_ACTIVE = False
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
