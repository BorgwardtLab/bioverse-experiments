#!/bin/bash
# Submit exp1 train jobs, then a test job that starts after each train succeeds.
# Usage:
#   ./submit_exp1.sh          # train + dependent test (default)
#   ./submit_exp1.sh train    # same
#   ./submit_exp1.sh test     # test only (no dependency)

set -euo pipefail

mode="${1:-train}"

submit_job() {
    local command=$1
    local exp_name=$2
    local framework=$3
    local model=$4
    local benchmark=$5
    local seed=$6
    local out_dir=$7
    shift 7
    local job_id
    job_id=$(sbatch --parsable \
        -p p.hpcl91,p.hpcl94g \
        --job-name="bvrs1" \
        --gres=gpu:h100:1 \
        --cpus-per-task 5 \
        --ntasks-per-core=2 \
        --mem-per-cpu 8G \
        --time 48:00:00 \
        --kill-on-invalid-dep=yes \
        --output="${out_dir}/slurm-%j.out" \
        --error="${out_dir}/slurm-%j.out" \
        "$@" \
        --wrap "bioverse ${command} configs/base.yaml configs/${framework}.yaml configs/${benchmark}.yaml configs/${model}.yaml seed=${seed} name=${exp_name}")
    echo "${job_id%%;*}"
}

for seed in {1..5}; do
    for framework in torch tensorflow; do
        for model in graph voxel point; do
            for benchmark in qm9 md17 cath42 dock gym tmalign enzyme rna; do
                exp_name="exp1_${framework}_${model}_${benchmark}_${seed}"
                out_dir="results/${exp_name}"
                mkdir -p "${out_dir}"
                if [ "$mode" = "test" ]; then
                    submit_job test "$exp_name" "$framework" "$model" "$benchmark" "$seed" "$out_dir" >/dev/null
                else
                    train_id=$(submit_job train "$exp_name" "$framework" "$model" "$benchmark" "$seed" "$out_dir")
                    submit_job test "$exp_name" "$framework" "$model" "$benchmark" "$seed" "$out_dir" \
                        --dependency="afterok:${train_id}" >/dev/null
                fi
            done
        done
    done
done
