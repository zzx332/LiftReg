DATA_ROOT="/home/zzx/data/pair_CT_DSA_10"

ROI="liver vertebrae_C1 vertebrae_C2 vertebrae_C3 vertebrae_C4 vertebrae_C5 vertebrae_C6 vertebrae_C7 \
vertebrae_T1 vertebrae_T2 vertebrae_T3 vertebrae_T4 vertebrae_T5 vertebrae_T6 vertebrae_T7 vertebrae_T8 \
vertebrae_T9 vertebrae_T10 vertebrae_T11 vertebrae_T12 vertebrae_L1 vertebrae_L2 vertebrae_L3 \
vertebrae_L4 vertebrae_L5 sacrum"

for case_dir in "$DATA_ROOT"/*/; do
    case_name=$(basename "$case_dir")
    ct_nii="$case_dir/CT/ct.nii.gz"
    out_dir="$case_dir/Segmentation"
    out_seg="$out_dir/seg.nii.gz"

    if [[ ! -f "$ct_nii" ]]; then
        echo "[SKIP] $case_name: no ct.nii.gz"
        continue
    fi

    mkdir -p "$out_dir"
    echo "[RUN] $case_name"

    TotalSegmentator -i "$ct_nii" -o "$out_seg" --ml -rs $ROI
done