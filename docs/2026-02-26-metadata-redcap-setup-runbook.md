# Phenotype Files Update Runbook

How to update the phenotype CSV/XLSX files used by `physio-qc` metadata features.

## Source Files

| Server filename | Source | Link / how to get |
|---|---|---|
| `InvestigationOfTheLo_DATA.csv` | REDCap export | **Ask Sophia** to export from the "Investigation of the Lo..." REDCap project |
| `Group_InfoSession_Data_CC.csv` | SharePoint | [Group_InfoSession_Data.xlsx (CC sheet)](https://mcgill-my.sharepoint.com/:x:/r/personal/georgios_mitsis_mcgill_ca/_layouts/15/Doc.aspx?sourcedoc=%7B32379283-C335-4FC1-8424-229DD78FAE66%7D&file=Group_InfoSession_Data.xlsx&action=default&mobileredirect=true) |
| `Group_InfoSession_Data_LC.csv` | SharePoint | Same workbook as above (LC sheet) |
| `Testing_Schedule_Sheet1.csv` | SharePoint | [Testing_Schedule.xlsx](https://mcgill-my.sharepoint.com/:x:/r/personal/georgios_mitsis_mcgill_ca/_layouts/15/Doc.aspx?sourcedoc=%7BB05032AE-A7DE-4BB9-BA6D-01BB85B81207%7D&file=Testing_Schedule.xlsx&action=default&mobileredirect=true) |
| `LC_Experiments_Notes_v2_Session_A_Physio.csv` | SharePoint | [LC_Experiments_Notes_v2.xlsx (Session A)](https://mcgill-my.sharepoint.com/:x:/r/personal/georgios_mitsis_mcgill_ca/_layouts/15/Doc.aspx?sourcedoc=%7B8B4E7939-AA57-4800-8031-7076921DB59C%7D&file=LC_Experiments_Notes_v2.xlsx&action=default&mobileredirect=true) |
| `LC_Experiments_Notes_v2_Session_B_MRI.csv` | SharePoint | Same workbook as above (Session B) |
| `REDCap_variables_definitions.xlsx` | REDCap | **Ask Sophia** (same as the DATA export) |

## Server Paths

The app reads from:
```
/export02/projects/LCS/05_phenotype/redcap_exports/
```

A mirror also exists at `/export02/projects/LCS_Management/05_phenotype/redcap_exports/` — the script updates both.

## Update Script

`scripts/update_phenotype_files.sh` automates the full update. Here's what it does:

1. **Renames** downloaded files from `~/Downloads` (parenthesized names like `Group_InfoSession_Data(CC) (1).csv`) to the plain underscore names the app expects (`Group_InfoSession_Data_CC.csv`)
2. **Checks** which renamed files are present and ready
3. **Uploads** them via `scp` to a staging folder on the server (`/home/bic/oreng/phenotype_upload/`)
4. **Replaces** old files in both server target paths (removes old date-tagged versions, copies new plain-named ones)
5. **Verifies** the final folder contents

### How to use

```bash
# Step 1: Download latest files from SharePoint / REDCap to ~/Downloads

# Step 2: Sync code to server first (so config.py with plain filenames is deployed)
physio-qc-sync

# Step 3: Dry-run — shows what would be renamed and uploaded, touches nothing
./scripts/update_phenotype_files.sh

# Step 4: If dry-run looks correct, apply for real
./scripts/update_phenotype_files.sh --apply
```

The dry-run is safe — it only reads files and prints. The `--apply` flag is required to actually copy/upload/replace anything.

### If the script isn't available or you prefer manual steps

See the "Manual Procedure" section below.

## Manual Procedure

### 1) Download from sources

- SharePoint files: open each link above, download the relevant sheet as CSV
- REDCap: ask Sophia for the latest export

Downloaded files land in `~/Downloads` with parenthesized names like `Group_InfoSession_Data(CC) (1).csv`.

### 2) Rename locally

Parentheses and spaces must be converted to underscores to match what the app expects.

```bash
cd ~/Downloads

cp "Group_InfoSession_Data(CC) (1).csv"                      Group_InfoSession_Data_CC.csv
cp "Group_InfoSession_Data(LC) (1).csv"                      Group_InfoSession_Data_LC.csv
cp "Testing_Schedule(Sheet1) (1).csv"                        Testing_Schedule_Sheet1.csv
cp "LC_Experiments_Notes_v2(Session A (Physio)) (1).csv"     LC_Experiments_Notes_v2_Session_A_Physio.csv
cp "LC_Experiments_Notes_v2(Session B (MRI)) (1).csv"        LC_Experiments_Notes_v2_Session_B_MRI.csv
cp "InvestigationOfTheLo_DATA_<date>_<time>.csv"             InvestigationOfTheLo_DATA.csv
```

(Adjust `(1)` suffix depending on how many times you've downloaded — omit if first download.)

### 3) Upload to server staging

```bash
ssh oreng@gm-stor "mkdir -p /home/bic/oreng/phenotype_upload"

scp ~/Downloads/InvestigationOfTheLo_DATA.csv \
    ~/Downloads/Group_InfoSession_Data_CC.csv \
    ~/Downloads/Group_InfoSession_Data_LC.csv \
    ~/Downloads/Testing_Schedule_Sheet1.csv \
    ~/Downloads/LC_Experiments_Notes_v2_Session_A_Physio.csv \
    ~/Downloads/LC_Experiments_Notes_v2_Session_B_MRI.csv \
    oreng@gm-stor:/home/bic/oreng/phenotype_upload/
```

(Only include `REDCap_variables_definitions.xlsx` if Sophia provided a new version.)

### 4) Replace files on server

```bash
ssh oreng@gm-stor bash -s <<'EOF'
SRC=/home/bic/oreng/phenotype_upload
for DEST in /export02/projects/LCS/05_phenotype/redcap_exports \
            /export02/projects/LCS_Management/05_phenotype/redcap_exports; do
  rm -f "$DEST"/Group_InfoSession_Data_CC_*.csv "$DEST"/Group_InfoSession_Data_CC.csv
  rm -f "$DEST"/Group_InfoSession_Data_LC_*.csv "$DEST"/Group_InfoSession_Data_LC.csv
  rm -f "$DEST"/Testing_Schedule_Sheet1_*.csv "$DEST"/Testing_Schedule_Sheet1.csv
  rm -f "$DEST"/LC_Experiments_Notes_v2_Session_A_Physio_*.csv "$DEST"/LC_Experiments_Notes_v2_Session_A_Physio.csv
  rm -f "$DEST"/LC_Experiments_Notes_v2_Session_B_MRI_*.csv "$DEST"/LC_Experiments_Notes_v2_Session_B_MRI.csv
  rm -f "$DEST"/InvestigationOfTheLo_DATA_*.csv "$DEST"/InvestigationOfTheLo_DATA.csv
  rm -f "$DEST"/REDCap_variables_definitions_*.xlsx "$DEST"/REDCap_variables_definitions.xlsx
  cp "$SRC"/* "$DEST"/
done
echo "Done — both paths updated"
EOF
```

If permission denied on the `LCS` path, ask the folder owner (sloparco) to run the copy.

### 5) Verify

```bash
ssh oreng@gm-stor "ls -lh /export02/projects/LCS/05_phenotype/redcap_exports/"
```

Expected: plain-named files with today's date, no old date-tagged duplicates.

## Format Changes Log

### 2026-03-27: Session B (MRI) notes

New columns added (no code changes needed — the app reads headers dynamically):
- `Break TIme` (between "Break Taken?" and "If Break, how long?")
- `Phase-Contrast Angio`
- `TOF Angio`

Columns changed from free text to `TRUE`/`FALSE`:
- `Break Taken?` (was: "?", "??", "NA", etc.)
- `Blood Draw` (was: "?", "Skipped, wasnt needed", etc.)

No code changes required — the app only matches on `*Notes*` columns for task-specific notes.

## Validation

Start app on server:
```bash
cd /export02/users/oreng/physio-qc && uv run streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8511
```

Tunnel from local machine:
```bash
ssh -N -L 8511:127.0.0.1:8511 oreng@gm-stor
```

Check at `http://127.0.0.1:8511` → Metadata tab:
- Demographics populate (Sex, Gender, Age, BMI)
- Setup/task notes appear for Session A and B
- BMI sourced from group info files (CC first, then LC)

## Troubleshooting

- `No such file or directory`: the `/export02/...` paths exist on the server, not your Mac.
- `Permission denied` on scp to final folder: use the staging approach above.
- Missing notes: verify the participant has data in that column in the source CSV.
