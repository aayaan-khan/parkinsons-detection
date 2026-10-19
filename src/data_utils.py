"""Loading and validating the UCI Parkinson's voice datasets."""

import io
from pathlib import Path
from typing import Optional, Union
import urllib.error
import urllib.request

import pandas as pd


OXFORD_FILENAME = "parkinsons.data"
TELEMONITORING_FILENAME = "parkinsons_updrs.data"

OXFORD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/parkinsons.data"
)
TELEMONITORING_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)
OXFORD_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/174/parkinsons"
TELEMONITORING_DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring"
)

DATA_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OXFORD_PATH = DATA_RAW_DIR / OXFORD_FILENAME
TELEMONITORING_PATH = DATA_RAW_DIR / TELEMONITORING_FILENAME

FEATURE_FAMILIES = {
    "frequency": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
    "jitter": [
        "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)",
        "MDVP:RAP",
        "MDVP:PPQ",
        "Jitter:DDP",
    ],
    "shimmer": [
        "MDVP:Shimmer",
        "MDVP:Shimmer(dB)",
        "Shimmer:APQ3",
        "Shimmer:APQ5",
        "MDVP:APQ",
        "Shimmer:DDA",
    ],
    "noise": ["NHR", "HNR"],
    "nonlinear": ["RPDE", "DFA", "D2", "spread1", "spread2", "PPE"],
}

_SUBJECT_PATTERN = r"(S\d+)"
_RECORDING_PATTERN = r"_(\d+)$"
_OXFORD_ROW_COUNT = 195
_OXFORD_SUBJECT_COUNT = 32
_OXFORD_CLASS_COUNTS = {0: 48, 1: 147}
_TELEMONITORING_ROW_COUNT = 5875
_TELEMONITORING_SUBJECT_COUNT = 42


def _require(condition: bool, message: str) -> None:
    """Raise a readable validation error regardless of Python optimization."""
    if not condition:
        raise ValueError(message)


def _read_local_or_download(
    path: Union[str, Path], url: str, dataset_name: str, dataset_page: str
) -> pd.DataFrame:
    """Read a local CSV or download and parse it without requiring a writable path."""
    data_path = Path(path)
    if data_path.exists():
        return pd.read_csv(data_path)

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            contents = response.read()
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"Could not read or download the {dataset_name} dataset. "
            f"Place the file at {data_path} or check your network connection. "
            f"Dataset information: {dataset_page}"
        ) from error

    dataframe = pd.read_csv(io.BytesIO(contents))
    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(contents)
    except OSError:
        # The parsed download remains usable when the target folder is read-only.
        pass

    return dataframe


def load_parkinsons(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and validate the Oxford Parkinsons voice dataset.

    Args:
        path: Optional local CSV path. Defaults to this project's
            ``data/raw/parkinsons.data``; a missing file is downloaded there.

    Returns:
        The Oxford dataset with parsed ``subject_id`` and ``recording_id``.
    """
    data_path = OXFORD_PATH if path is None else Path(path)
    dataframe = _read_local_or_download(
        data_path, OXFORD_URL, "Oxford Parkinsons", OXFORD_DATASET_PAGE
    )

    _require("name" in dataframe.columns, "Oxford dataset is missing the 'name' column.")
    _require(
        "status" in dataframe.columns,
        "Oxford dataset is missing the 'status' target column.",
    )
    dataframe["subject_id"] = dataframe["name"].str.extract(
        _SUBJECT_PATTERN, expand=False
    )
    dataframe["recording_id"] = (
        dataframe["name"].str.extract(_RECORDING_PATTERN, expand=False).astype("Int64")
    )

    _require(
        len(dataframe) == _OXFORD_ROW_COUNT,
        f"Oxford dataset must contain {_OXFORD_ROW_COUNT} rows; found {len(dataframe)}.",
    )
    _require(
        dataframe["subject_id"].nunique() == _OXFORD_SUBJECT_COUNT,
        f"Oxford dataset must contain {_OXFORD_SUBJECT_COUNT} subjects; "
        f"found {dataframe['subject_id'].nunique()}.",
    )
    _require(
        not dataframe.isna().any().any(),
        "Oxford dataset contains null values.",
    )
    _require(
        dataframe["status"].isin([0, 1]).all(),
        "Oxford 'status' must contain only 0 (healthy) and 1 (Parkinson's).",
    )
    _require(
        dataframe.groupby("subject_id")["status"].nunique().le(1).all(),
        "Oxford 'status' must be constant within each subject.",
    )
    class_counts = dataframe["status"].value_counts().to_dict()
    _require(
        class_counts == _OXFORD_CLASS_COUNTS,
        "Oxford status counts must be 48 healthy (0) and 147 PD (1); "
        f"found {class_counts}.",
    )

    return dataframe


def load_telemonitoring(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate the UCI Parkinsons Telemonitoring dataset.

    Args:
        path: Optional local CSV path. Defaults to this project's
            ``data/raw/parkinsons_updrs.data``; a missing file is downloaded there.

    Returns:
        The Telemonitoring dataset. It contains PD patients only and is for
        regression, not detection.
    """
    data_path = TELEMONITORING_PATH if path is None else Path(path)
    dataframe = _read_local_or_download(
        data_path,
        TELEMONITORING_URL,
        "Parkinsons Telemonitoring",
        TELEMONITORING_DATASET_PAGE,
    )

    _require(
        len(dataframe) == _TELEMONITORING_ROW_COUNT,
        f"Telemonitoring dataset must contain {_TELEMONITORING_ROW_COUNT} rows; "
        f"found {len(dataframe)}.",
    )
    _require(
        "subject#" in dataframe.columns,
        "Telemonitoring dataset is missing the 'subject#' grouping column.",
    )
    _require(
        dataframe["subject#"].nunique() == _TELEMONITORING_SUBJECT_COUNT,
        f"Telemonitoring dataset must contain {_TELEMONITORING_SUBJECT_COUNT} "
        f"subjects; found {dataframe['subject#'].nunique()}.",
    )

    return dataframe


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return Oxford voice features and verify they match the defined families."""
    excluded_columns = {"name", "subject_id", "recording_id", "status"}
    feature_columns = [
        column for column in dataframe.columns if column not in excluded_columns
    ]
    family_columns = {
        column for family in FEATURE_FAMILIES.values() for column in family
    }

    _require(
        len(feature_columns) == 22,
        f"Expected 22 Oxford feature columns; found {len(feature_columns)}.",
    )
    _require(
        set(feature_columns) == family_columns,
        "Oxford feature columns must exactly match the union of FEATURE_FAMILIES.",
    )
    return feature_columns


if __name__ == "__main__":
    oxford = load_parkinsons()
    telemonitoring = load_telemonitoring()
    get_feature_columns(oxford)

    print(f"Oxford shape: {oxford.shape}")
    print(f"Oxford unique subjects: {oxford['subject_id'].nunique()}")
    print(
        "Oxford class counts (status): "
        f"{oxford['status'].value_counts().sort_index().to_dict()}"
    )
    print(f"Telemonitoring shape: {telemonitoring.shape}")
    print(f"Telemonitoring unique subjects: {telemonitoring['subject#'].nunique()}")
    print("ALL CHECKS PASSED")