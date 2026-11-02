#!/bin/bash
# Submit exp2 train jobs, then CATH and AF-CATH test jobs after each train succeeds.
# Usage:
#   ./submit_exp2.sh          # train + dependent tests (default)
#   ./submit_exp2.sh train    # same
#   ./submit_exp2.sh test     # both evals only (no dependency)

set -euo pipefail

mode="${1:-train}"

submit_job() {
    local command=$1
    local exp_name=$2
    local dataset_config=$3
    local model=$4
    local feature_name=$5
    local seed=$6
    local out_dir=$7
    local extra_wrap=$8
    shift 8
    local job_id
    job_id=$(sbatch --parsable \
        -p p.hpcl91,p.hpcl94g \
        --job-name="bvrs2" \
        --gres=gpu:h100:1 \
        --cpus-per-task 10 \
        --ntasks-per-core=2 \
        --mem-per-cpu 30G \
        --time 48:00:00 \
        --kill-on-invalid-dep=yes \
        --output="${out_dir}/slurm-%j.out" \
        --error="${out_dir}/slurm-%j.out" \
        "$@" \
        --wrap "bioverse ${command} configs/base.yaml configs/torch.yaml configs/${dataset_config}.yaml configs/${model}.yaml configs/${feature_name}_features.yaml seed=${seed} name=${exp_name} globals.workers=10 trainer.workers=10 ${extra_wrap}")
    echo "${job_id%%;*}"
}

for seed in 1; do
    for model in gvp pifold mpnn; do
        for dataset_config in cath42 afcath affull; do
            case $dataset_config in
                cath42) dataset=inv ;;
                afcath) dataset=afcath ;;
                affull) dataset=affull ;;
            esac
            for feature in gvp mpnn pifold; do
                exp_name=exp2_${model}_${feature}_${dataset}_${seed}
                out_dir="results/${exp_name}"
                mkdir -p "${out_dir}"
                # CATH eval writes results/<exp>/test_results.yaml (historical path).
                # AF-CATH eval writes to a subdirectory so it does not overwrite CATH.
                cath_wrap="benchmark=B_INVC42"
                afcath_dir="${out_dir}/eval_afcath"
                mkdir -p "${afcath_dir}"
                afcath_wrap="benchmark=B_AFCATH trainer.root=${afcath_dir} trainer.restore_from=../checkpoint"
                if [ "$mode" = "test" ]; then
                    submit_job test "$exp_name" "$dataset_config" "$model" "$feature" "$seed" "$out_dir" \
                        "$cath_wrap" >/dev/null
                    submit_job test "$exp_name" "$dataset_config" "$model" "$feature" "$seed" "$afcath_dir" \
                        "$afcath_wrap" >/dev/null
                else
                    train_id=$(submit_job train "$exp_name" "$dataset_config" "$model" "$feature" "$seed" "$out_dir" "")
                    submit_job test "$exp_name" "$dataset_config" "$model" "$feature" "$seed" "$out_dir" \
                        "$cath_wrap" --dependency="afterok:${train_id}" >/dev/null
                    submit_job test "$exp_name" "$dataset_config" "$model" "$feature" "$seed" "$afcath_dir" \
                        "$afcath_wrap" --dependency="afterok:${train_id}" >/dev/null
                fi
            done
        done
    done
done
