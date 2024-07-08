export GLOG_minloglevel=3
export MAGNUM_LOG=quiet
export OMP_NUM_THREADS=12

#### Config
CKPT_RELEASED_R2RIE_CE='data/results/ckpt_released/ckpt.iter16200_tm.pth' 
NUM_GPU=2
#####

RUN_NAME_EVAL="released_val_unseen_error_v1_direction" 

EVAL_FLAGS="--exp_name $RUN_NAME_EVAL
      --run-type eval
      --exp-config run_R2RIE-CE/iter_train.yaml
      SIMULATOR_GPU_IDS [$(seq -s, 0 $((NUM_GPU-1)))]
      TORCH_GPU_IDS [$(seq -s, 0 $((NUM_GPU-1)))]
      GPU_NUMBERS $NUM_GPU
      NUM_ENVIRONMENTS 9

      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR $CKPT_RELEASED_R2RIE_CE
      IL.back_algo control
      MODEL.DEPTH_ENCODER.ddppo_checkpoint /media/data_1/ftaioli/embodied_ai_ckpt/gibson-2plus-resnet50.pth
      
      MODEL.VERBOSE.IS_VERBOSE False
      EVAL.EPISODE_COUNT -1
      
      EVAL.SPLIT val_unseen_error_v1_direction
      VIDEO_DIR data/logs/video_debug
      "

echo "#######################"
echo "###### EVAL mode ######"
echo "#######################"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $1 run.py $EVAL_FLAGS
echo "Eval finished for:" $RUN_NAME_EVAL for ckpt $CKPT_RELEASED_R2RIE_CE
