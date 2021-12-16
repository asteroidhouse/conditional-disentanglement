for SEED in 1 2 3; do 
  for OBJECTIVE in Base BaseMI BaseCMI; do
    if [ $OBJECTIVE == Base ];
    then
     MI_TYPE=none
     SAVE_DIR=saves/celeba_weakly/celeba_cls
    fi
    
    if [ $OBJECTIVE == BaseMI ];
    then
     MI_TYPE=unconditional
     SAVE_DIR=saves/celeba_weakly/celeba_uncond
    fi
    
    if [ $OBJECTIVE == BaseCMI ];
    then
     MI_TYPE=conditional
     SAVE_DIR=saves/celeba_weakly/celeba_cond
    fi
    
    for WEAK_SUPERVISION_PERCENTAGE in 0 50 75 90 95 ; do
      python train_correlated.py \
           --model=mlp \
           --epochs=200 \
           --dataset_type=correlated1 \
           --target_variable1=Male \
           --target_variable2=Smiling \
           --plot_covariance \
           --train_corr=0.8 \
           --test_corr=-0.8 \
           --D_lr=1e-4 \
           --lr=1e-5 \
           --cls_lr=1e-4 \
           --num_cls_steps=1 \
           --z_dim=10 \
           --disentangle_weight=10.0 \
           --mi_type=$MI_TYPE \
           --save_dir=$SAVE_DIR \
           --weak_supervision_percentage=$WEAK_SUPERVISION_PERCENTAGE \
           --seed=$SEED &
    done
  done
done
