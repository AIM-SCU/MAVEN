EC_path=$1
dir_videos=$2
dir_prompts=$3
gpu_id=${4:-0}  # Default to 0 if not provided

# Activate ec conda environment for EvalCrafter
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate ec

# IS
cd $EC_path
cd ./metrics
### IS.py does not need prompts to evaluate?
python3 is.py --dir_videos $dir_videos --gpu_id $gpu_id


# # VQA_A and VQA_T
cd $EC_path
cd ./metrics/DOVER
### evaluate_a_set_of_videos.py does not need prompts to evaluate?
python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos --gpu_id $gpu_id


# CLIP-Score
cd $EC_path
cd ./metrics/Scores_with_CLIP
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_score' --dir_prompts $dir_prompts --gpu_id $gpu_id

# Face Consistency
cd $EC_path
cd ./metrics/Scores_with_CLIP
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'face_consistency_score' --dir_prompts $dir_prompts --gpu_id $gpu_id

# SD-Score
cd $EC_path
cd ./metrics/Scores_with_CLIP
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'sd_score' --dir_prompts $dir_prompts --gpu_id $gpu_id

# BLIP-BLUE
## TODO: need to update [transformers] to latest using [pip install -U transformers], also comment out the [cached_download] from 'huggingface_hub'
## need to find a way to not do this removal of [cached_download] in the future
cd $EC_path
cd ./metrics/Scores_with_CLIP
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'blip_bleu' --dir_prompts $dir_prompts --gpu_id $gpu_id

# CLIP-Temp
cd $EC_path
cd ./metrics/Scores_with_CLIP
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_temp_score' --dir_prompts $dir_prompts --gpu_id $gpu_id



# Flow-Score
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score' --dir_prompts $dir_prompts --gpu_id $gpu_id



# Warping Error
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'warping_error' --dir_prompts $dir_prompts --gpu_id $gpu_id


# # Final results - pass GPU ID to use correct results directory
cd $EC_path
python eval_from_metrics_2.py --gpu_id $gpu_id

# Deactivate conda environment
conda deactivate 



