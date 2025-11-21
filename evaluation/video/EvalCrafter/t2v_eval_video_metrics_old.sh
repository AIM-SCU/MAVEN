EC_path=$1 
dir_videos=$2
dir_prompts=$3

# [need for speicific platform] pip install spatial_correlation_sampler
# pip install spatial_correlation_sampler==0.4.0

# Celebrity ID Score
# detect human faces which is not helpful for our case
# cd $EC_path
# cd ./metrics/deepface
# python3 celebrity_id_score.py --dir_videos  $dir_videos --dir_prompts $dir_prompts

# IS
cd $EC_path
cd ./metrics
### IS.py does not need prompts to evaluate?
python3 is.py --dir_videos $dir_videos 


# # OCR Score
## detect text in video (which is not helpful for our case)
# cd $EC_path
# cd ./metrics
# ### OCR_score.py needs [dir_path_face]?
# python3 ocr_score.py --dir_videos $dir_videos --metric 'ocr_score' --dir_prompts $dir_prompts

# # VQA_A and VQA_T
cd $EC_path
cd ./metrics/DOVER
### evaluate_a_set_of_videos.py does not need prompts to evaluate?
python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos


# CLIP-Score 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_score' --dir_prompts $dir_prompts

# Face Consistency 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'face_consistency_score' --dir_prompts $dir_prompts

# SD-Score 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'sd_score' --dir_prompts $dir_prompts

# BLIP-BLUE 
## TODO: need to update [transformers] to latest using [pip install -U transformers], also comment out the [cached_download] from 'huggingface_hub'
## need to find a way to not do this removal of [cached_download] in the future
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'blip_bleu' --dir_prompts $dir_prompts

# CLIP-Temp 
cd $EC_path
cd ./metrics/Scores_with_CLIP 
python3 Scores_with_CLIP.py --dir_videos $dir_videos --metric 'clip_temp_score' --dir_prompts $dir_prompts

# # # Action Score
## TODO: this could be zero (because the action may not be recognized from {Kinetics 400 action classes})
# cd $EC_path
# cd ./metrics/mmaction2/demo
# python3 action_score.py --dir_videos $dir_videos --metric 'action_score' --dir_prompts $dir_prompts


# Flow-Score
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score' --dir_prompts $dir_prompts

# Motion AC-Score
## TODO: this could be zero (I tihnk this is based on the motion detected in action_score, which could be zero; so here this could also be zero because no motion detected)
# cd $EC_path
# cd ./metrics/RAFT
# python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'motion_ac_score' --dir_prompts $dir_prompts

# Warping Error
cd $EC_path
cd ./metrics/RAFT
python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'warping_error' --dir_prompts $dir_prompts


# Count-Score
## TODO: have problem running this, but we don't need this metric for now.
## error is[NameError: name '_C' is not defined]
# cd $EC_path
# cd ./metrics/Segment-and-Track-Anything
# python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'count_score' --dir_prompts $dir_prompts

# # Color-Score
## TODO: this could be zero
# cd $EC_path
# cd ./metrics/Segment-and-Track-Anything
# python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'color_score' --dir_prompts $dir_prompts

# Detection-Score
## TODO: this could be zero (because no objects could be detected)
# cd $EC_path
# cd ./metrics/Segment-and-Track-Anything
# python3 object_attributes_eval.py --dir_videos $dir_videos --metric 'detection_score' --dir_prompts $dir_prompts


# # Final results
cd $EC_path
python eval_from_metrics.py 



