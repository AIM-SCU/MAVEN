#!/usr/bin/env bash

# # Make sure conda commands work in this shell:
# eval "$(conda shell.bash hook)"
source /workspace/miniconda3/etc/profile.d/conda.sh

# 1) Deactivate current env
conda deactivate

# 2) Activate evaluation env
conda activate ec

# 3) Run your actual evaluation script
############################################
export HF_HOME=/workspace/hf_home
echo $HF_HOME
############################################
cd /workspace/t2v_self/evaluation/video/EvalCrafter
bash t2v_eval_video_metrics.sh /workspace/t2v_self/evaluation/video/EvalCrafter \
    /workspace/t2v_self/evaluation/video/EvalCrafter/my_test_videos \
    /workspace/t2v_self/evaluation/video/EvalCrafter/my_test_prompts

# 4) Deactivate that env
conda deactivate

# 5) Reactivate your main environment
conda activate autogen

