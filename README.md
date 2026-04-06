# Subtle Medical Coding Challenge

Tasks I and II of the Subtle Medical coding challenge: DICOM I/O and fast acquisition simulation.

---

## Project layout

```
subtle_medical_challenge/
├── subtle/
│   ├── dicom_io.py        # Core DICOM loading / saving utilities
│   └── simulation.py      # blurring3d() — fast acquisition simulation
├── scripts/
│   ├── dicom_to_hdf5.py   # Task I  – convert DICOM series → HDF5
│   ├── hdf5_to_dicom.py   # Task I  – convert HDF5 → DICOM series
│   └── simulate_fast_acq.py  # Task II – Gaussian blur + visualisation
├── tests/
│   ├── test_dicom_io.py
│   └── test_simulation.py
└── pyproject.toml
```

---

## Setup

[uv](https://docs.astral.sh/uv/) is used for environment and dependency management.
Package versions are pinned to releases before **2026-01-01** to mitigate
supply-chain risk from freshly-published packages (`exclude-newer` in `pyproject.toml`).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install all dependencies
cd subtle_medical_challenge
uv sync

# Activate the environment (optional — uv run works without activation)
source .venv/bin/activate
```

---

## Data

Download the first knee case from [mridata.org](http://old.mridata.org/fullysampled/knees)
(Dicoms column, ~67 MB zip).  Unzip it into a local directory, e.g. `data/case01/dicoms/`.

```
data/
└── case01/
    └── dicoms/   ← unzip the downloaded archive here
```

---

## Task I – DICOM ↔ HDF5 round-trip

### Script 1 — DICOM → HDF5

Reads all DICOM slices from a directory, sorts them by `SliceLocation`,
normalises pixel values to `[0, 1]` float32, and writes a compressed HDF5 file.

```bash
python scripts/dicom_to_hdf5.py \
    --input-dicom  data/case01/dicoms \
    --output-hdf5  data/case01/sharp.h5
```

### Script 2 — HDF5 → DICOM

Reads the HDF5 volume, rescales it back to the original dtype and value range
(using the template DICOM series), and writes new DICOM files.

```bash
python scripts/hdf5_to_dicom.py \
    --input-dicom  data/case01/dicoms \
    --input-hdf5   data/case01/sharp.h5 \
    --output-dicom data/case01/roundtrip_dicoms
```

Running both scripts in sequence on the same data must produce DICOM files that
are pixel-identical to the originals (the normalise → denormalise cycle is
exact to within float32 precision; integer rounding is applied before casting).

---

## Task II – Simulating a fast acquisition

Applies a 2-D Gaussian blur independently to each axial slice (σ = 5 pixels by
default) to simulate reduced in-plane resolution from k-space undersampling.
Saves the blurred volume to HDF5 and DICOM, and optionally writes a side-by-side
comparison of the central slice.

```bash
python scripts/simulate_fast_acq.py \
    --input-hdf5    data/case01/sharp.h5 \
    --input-dicom   data/case01/dicoms \
    --output-hdf5   data/case01/blurry.h5 \
    --output-dicom  data/case01/blurry_dicoms \
    --sigma         5.0 \
    --save-figure   figures/task2_central_slice.png
```

---

## Running the tests

```bash
uv run pytest -v
```

All tests are self-contained and require no external data.

---

## Design decisions

| Decision | Rationale |
|---|---|
| Slice-wise 2-D blur (not 3-D) | MRI acceleration acts in the in-plane phase-encode direction; through-plane resolution is unaffected |
| float64 arithmetic for rescaling | Prevents accumulated rounding error that would break the round-trip identity test |
| `exclude-newer = "2026-01-01"` | Packages released after this date are excluded to mitigate supply-chain risks from freshly-published malicious packages |
| gzip compression in HDF5 | ~2–4× smaller files with no information loss; transparent to readers |
