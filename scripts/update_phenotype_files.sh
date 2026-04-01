#!/usr/bin/env bash
# Update phenotype CSV/XLSX files on the server from ~/Downloads.
#
# Usage:
#   ./scripts/update_phenotype_files.sh          # dry-run (shows what would happen)
#   ./scripts/update_phenotype_files.sh --apply   # actually upload and replace
#
# See docs/2026-02-26-metadata-redcap-setup-runbook.md for full context.

set -euo pipefail

DL="$HOME/Downloads"
SERVER="oreng@gm-stor"
STAGING="/home/bic/oreng/phenotype_upload"
TARGETS=(
    "/export02/projects/LCS/05_phenotype/redcap_exports"
    "/export02/projects/LCS_Management/05_phenotype/redcap_exports"
)

# Download patterns (basename without extension) -> server filename
# Each entry: "download_base_pattern|server_filename"
ENTRIES=(
    "Group_InfoSession_Data(CC)|Group_InfoSession_Data_CC.csv"
    "Group_InfoSession_Data(LC)|Group_InfoSession_Data_LC.csv"
    "Testing_Schedule(Sheet1)|Testing_Schedule_Sheet1.csv"
    "LC_Experiments_Notes_v2(Session A (Physio))|LC_Experiments_Notes_v2_Session_A_Physio.csv"
    "LC_Experiments_Notes_v2(Session B (MRI))|LC_Experiments_Notes_v2_Session_B_MRI.csv"
)

PLAIN_FILES=(
    "Group_InfoSession_Data_CC.csv"
    "Group_InfoSession_Data_LC.csv"
    "Testing_Schedule_Sheet1.csv"
    "LC_Experiments_Notes_v2_Session_A_Physio.csv"
    "LC_Experiments_Notes_v2_Session_B_MRI.csv"
    "InvestigationOfTheLo_DATA.csv"
    "REDCap_variables_definitions.xlsx"
)

DRY_RUN=true
[[ "${1:-}" == "--apply" ]] && DRY_RUN=false

# --- Step 1: Rename downloads ---
echo "=== Step 1: Rename downloads ==="
for entry in "${ENTRIES[@]}"; do
    pattern="${entry%%|*}"
    target="${entry##*|}"
    found=""
    for suffix in "" " (1)" " (2)" " (3)"; do
        candidate="$DL/${pattern}${suffix}.csv"
        [[ -f "$candidate" ]] && found="$candidate"
    done
    if [[ -n "$found" ]]; then
        echo "  $(basename "$found") -> $target"
        $DRY_RUN || cp "$found" "$DL/$target"
    fi
done

# Handle date-tagged REDCap file
redcap_src=$(ls -t "$DL"/InvestigationOfTheLo_DATA_????-??-??_????.csv 2>/dev/null | head -1 || true)
if [[ -n "${redcap_src:-}" ]]; then
    echo "  $(basename "$redcap_src") -> InvestigationOfTheLo_DATA.csv"
    $DRY_RUN || cp "$redcap_src" "$DL/InvestigationOfTheLo_DATA.csv"
fi

# --- Step 2: Check which files are ready ---
echo ""
echo "=== Step 2: Files ready to upload ==="
upload_files=()
for f in "${PLAIN_FILES[@]}"; do
    if [[ -f "$DL/$f" ]]; then
        echo "  OK  $f"
        upload_files+=("$DL/$f")
    else
        echo "  --  $f (not found, skipping)"
    fi
done

if [[ ${#upload_files[@]} -eq 0 ]]; then
    echo "No files to upload."
    exit 1
fi

if $DRY_RUN; then
    echo ""
    echo "=== Dry run complete. Run with --apply to upload. ==="
    exit 0
fi

# --- Step 3: Upload to staging ---
echo ""
echo "=== Step 3: Upload to staging ==="
ssh "$SERVER" "mkdir -p $STAGING"
scp "${upload_files[@]}" "$SERVER:$STAGING/"

# --- Step 4: Replace on server ---
echo ""
echo "=== Step 4: Replace files on server ==="
for dest in "${TARGETS[@]}"; do
    echo "  Updating $dest ..."
    ssh "$SERVER" bash <<EOF
cd "$dest" 2>/dev/null || { echo "  SKIP $dest (not accessible)"; exit 0; }
rm -f Group_InfoSession_Data_CC_*.csv Group_InfoSession_Data_CC.csv
rm -f Group_InfoSession_Data_LC_*.csv Group_InfoSession_Data_LC.csv
rm -f Testing_Schedule_Sheet1_*.csv Testing_Schedule_Sheet1.csv
rm -f LC_Experiments_Notes_v2_Session_A_Physio_*.csv LC_Experiments_Notes_v2_Session_A_Physio.csv
rm -f LC_Experiments_Notes_v2_Session_B_MRI_*.csv LC_Experiments_Notes_v2_Session_B_MRI.csv
rm -f InvestigationOfTheLo_DATA_*.csv InvestigationOfTheLo_DATA.csv
rm -f REDCap_variables_definitions_*.xlsx REDCap_variables_definitions.xlsx
cp $STAGING/* "$dest/"
echo "  Done: $dest"
EOF
done

# --- Step 5: Verify ---
echo ""
echo "=== Step 5: Verify ==="
ssh "$SERVER" "ls -lh ${TARGETS[0]}/"

echo ""
echo "=== All done. Validate in app. ==="
