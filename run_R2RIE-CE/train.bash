export GLOG_minloglevel=3
export MAGNUM_LOG=quiet
export OMP_NUM_THREADS=12

#### Config
WANDB_RUN_NAME='train_my_model_my_infos' # set wandb to true to use this
WANDB_NOTES='train_my_model_my_infos' # cannot contains spaces
TRAIN_SPLIT_FILE='train_error_v1_all'
NUM_GPU=2
#####

TRAIN_FLAGS="--exp_name $WANDB_RUN_NAME
      --run-type train
      --exp-config run_R2RIE-CE/iter_train.yaml
      SIMULATOR_GPU_IDS [$(seq -s, 0 $((NUM_GPU-1)))]
      TORCH_GPU_IDS [$(seq -s, 0 $((NUM_GPU-1)))]
      GPU_NUMBERS $NUM_GPU
      NUM_ENVIRONMENTS 8
      IL.iters 18000
      IL.lr 1.5e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000

      
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

      MODEL.DEPTH_ENCODER.ddppo_checkpoint /media/data_1/ftaioli/embodied_ai_ckpt/gibson-2plus-resnet50.pth
      MODEL.pretrained_path ckpt/model_step_50000.pt


      MODEL.WANDB.use False    
      MODEL.WANDB.run_name $WANDB_RUN_NAME
      MODEL.WANDB.notes $WANDB_NOTES
      IL.load_from_ckpt True

      MODEL.TRAIN_TRAJECTORY_MATCHING.train True
      TASK_CONFIG.DATASET.SPLIT $TRAIN_SPLIT_FILE
      "

echo "###########################"
echo "###### Training mode ######"
echo "###########################"
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port $1 run.py $TRAIN_FLAGS

echo "training finished!"
