# Metadata Setup Runbook (2026-02-26)

This runbook documents how to prepare and deploy the phenotype files used by the `physio-qc` metadata features.

## Scope

The metadata system now uses:
- participant demographics (sex, gender, age, BMI)
- neuropsych summary (including MoCA fields)
- questionnaire context/interpretation
- schedule researchers
- Session A/B experiment notes (overall + task notes)

Server target folder:
- `/export02/projects/LCS_Management/05_phenotype/redcap_exports/`

## Required Filenames (Exact)

The app expects these exact names in the server folder:

- `InvestigationOfTheLo_DATA_2026-02-10_1414.csv`
- `Group_InfoSession_Data(CC).csv`
- `Group_InfoSession_Data(LC).csv`
- `Testing_Schedule(Sheet1).csv`
- `LC_Experiments_Notes_v2(Session A (Physio)).csv`
- `LC_Experiments_Notes_v2(Session B (MRI)).csv`
- `REDCap_variables_definitions.xlsx`

## Local File Preparation

If your local LC file arrives as `Group_InfoSession_Data(LC) (1).csv`, rename it first:

```bash
cd /Users/orengurevitch/Downloads
mv "Group_InfoSession_Data(LC) (1).csv" "Group_InfoSession_Data(LC).csv"
```

## Upload Procedure (Safe)

Because `scp` to `/export02/.../redcap_exports` may fail with permission denied, use staging first.

### 1) Create a server staging folder

```bash
ssh oreng@gm-stor "mkdir -p /home/bic/oreng/redcap_upload_2026_02_26"
```

### 2) Upload from local machine to staging

```bash
scp "/Users/orengurevitch/Downloads/InvestigationOfTheLo_DATA_2026-02-10_1414.csv" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
scp "/Users/orengurevitch/Downloads/Group_InfoSession_Data(CC).csv" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
scp "/Users/orengurevitch/Downloads/Group_InfoSession_Data(LC).csv" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
scp "/Users/orengurevitch/Downloads/Testing_Schedule(Sheet1).csv" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
scp "/Users/orengurevitch/Downloads/LC_Experiments_Notes_v2(Session A (Physio)).csv" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
scp "/Users/orengurevitch/Downloads/LC_Experiments_Notes_v2(Session B (MRI)).csv" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
scp "/Users/orengurevitch/Downloads/REDCap_variables_definitions.xlsx" oreng@gm-stor:/home/bic/oreng/redcap_upload_2026_02_26/
```

### 3) Move staging files into final folder

```bash
ssh oreng@gm-stor "cp /home/bic/oreng/redcap_upload_2026_02_26/* /export02/projects/LCS_Management/05_phenotype/redcap_exports/"
```

If that fails due permissions, run with `sudo` (if you have it), or ask folder owner/admin to run that command.

### 4) Verify final folder

```bash
ssh oreng@gm-stor "ls -lh /export02/projects/LCS_Management/05_phenotype/redcap_exports/ | grep -E 'InvestigationOfTheLo|Group_InfoSession_Data|Testing_Schedule|LC_Experiments_Notes_v2|REDCap_variables_definitions'"
```

## How To Validate In App

Start app on server:

```bash
source /export02/users/oreng/.venv/bin/activate
cd /export02/users/oreng/physio-qc
python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8511
```

Tunnel from local machine:

```bash
ssh -N -L 8511:127.0.0.1:8511 oreng@gm-stor
```

Open:
- `http://127.0.0.1:8511`

Check in Metadata tab:
- top row fields populate (`Sex`, `Gender`, `Age`, `BMI`, `ECG Configuration`)
- `BMI` comes from group info files when available (CC first, then LC)
- Setup notes appear only in Metadata tab
- task tabs show task-specific notes (for example, task `rest` shows `Resting State - Notes` when present)

## Troubleshooting

- `No such file or directory` for `/export02/...` on local Mac:
  - expected; that path exists on server, not on local machine.
- `scp ... Permission denied` to final folder:
  - upload to `/home/bic/oreng/redcap_upload_2026_02_26/` first, then copy server-side.
- notes missing for a task:
  - verify participant has notes for that specific task column in Session A/B notes file.
- wrong/empty BMI:
  - verify `BMI` value in `Group_InfoSession_Data(CC).csv` or `Group_InfoSession_Data(LC).csv`.
