# Disentanglement and Generalization Under Correlation Shifts

This repository contains code for the paper "Disentanglement and Generalization Under Correlation Shifts."


## Experiments

### Toy 2D Linear Regression
```
python toy_linear_regression_notebook.py
```

### Toy Linear Classification
```
for DIM in 2 4 10 ; do
    python toy_linear_classification.py $DIM 1
done
```

### Correlated Multi-Digit MNIST

**Train with only Classification Loss**
```
for NOISE in 0.0 0.2 0.4 0.6 0.8 ; do
    for TRAIN_CORRELATION in 0.0 0.6 0.9 ; do
        python train_correlated.py \
            --model=mlp \
            --epochs=400 \
            --dataset_type=multi-digit \
            --dataset=mnist \
            --train_corr=$TRAIN_CORRELATION \
            --test_corr=-$TRAIN_CORRELATION \
            --lr=1e-4 \
            --cls_lr=1e-3 \
            --z_dim=10 \
            --noise=$NOISE \
            --mi_type=none \
            --save_dir=saves/multi_mnist_cls &
    done
done
```

**Train with Unconditional Disentanglement**
```
for NOISE in 0.0 0.2 0.4 0.6 0.8 ; do
    for TRAIN_CORRELATION in 0.0 0.6 0.9 ; do
        python train_correlated.py \
            --model=mlp \
            --epochs=400 \
            --dataset_type=multi-digit \
            --dataset=mnist \
            --train_corr=$TRAIN_CORRELATION \
            --test_corr=-$TRAIN_CORRELATION \
            --D_lr=1e-4 \
            --lr=1e-5 \
            --cls_lr=1e-4 \
            --z_dim=10 \
            --noise=$NOISE \
            --mi_type=unconditional \
            --save_dir=saves/multi_mnist_uncond &
    done
done
```

**Train with Conditional Disentanglement**
```
for NOISE in 0.0 0.2 0.4 0.6 0.8 ; do
    for TRAIN_CORRELATION in 0.0 0.6 0.9 ; do
        python train_correlated.py \
            --model=mlp \
            --epochs=400 \
            --dataset_type=multi-digit \
            --dataset=mnist \
            --train_corr=$TRAIN_CORRELATION \
            --test_corr=-$TRAIN_CORRELATION \
            --D_lr=1e-4 \
            --lr=1e-5 \
            --cls_lr=1e-4 \
            --z_dim=10 \
            --noise=$NOISE \
            --mi_type=conditional \
            --save_dir=saves/multi_mnist_cond &
    done
done
```

**Plot Results**
```
python plot_multi_mnist.py
```


### Correlated CelebA

**Train with only Classification Loss**
```
for TRAIN_CORRELATION in 0.0 0.2 0.4 0.6 0.8 ; do
	python train_correlated.py \
        --model=mlp \
        --epochs=200 \
        --dataset_type=correlated1 \
        --target_variable1=Male \
        --target_variable2=Smiling \
        --plot_covariance \
        --train_corr=$TRAIN_CORRELATION \
        --test_corr=-$TRAIN_CORRELATION \
        --D_lr=1e-4 \
        --lr=1e-5 \
        --cls_lr=1e-4 \
        --num_cls_steps=1 \
        --z_dim=10 \
        --disentangle_weight=10.0 \
        --mi_type=none \
        --save_dir=saves/celeba_cls &
done
```

**Train with Unconditional Disentanglement**
```
for TRAIN_CORRELATION in 0.0 0.2 0.4 0.6 0.8 ; do
    python train_correlated.py \
        --model=mlp \
        --epochs=200 \
        --dataset_type=correlated1 \
        --target_variable1=Male \
        --target_variable2=Smiling \
        --plot_covariance \
        --train_corr=$TRAIN_CORRELATION \
        --test_corr=-$TRAIN_CORRELATION \
        --D_lr=1e-4 \
        --lr=1e-5 \
        --cls_lr=1e-4 \
        --num_cls_steps=1 \
        --z_dim=10 \
        --disentangle_weight=10.0 \
        --mi_type=unconditional \
        --save_dir=saves/celeba_uncond &
done
```

**Train with Conditional Disentanglement**
```
for TRAIN_CORRELATION in 0.0 0.2 0.4 0.6 0.8 ; do
    python train_correlated.py \
        --model=mlp \
        --epochs=200 \
        --dataset_type=correlated1 \
        --target_variable1=Male \
        --target_variable2=Smiling \
        --plot_covariance \
        --train_corr=$TRAIN_CORRELATION \
        --test_corr=-$TRAIN_CORRELATION \
        --D_lr=1e-4 \
        --lr=1e-5 \
        --cls_lr=1e-4 \
        --num_cls_steps=1 \
        --z_dim=10 \
        --disentangle_weight=10.0 \
        --mi_type=conditional \
        --save_dir=saves/celeba_cond &
done
```

**Plot Results**
```
python plot_celeba.py
```


### Weakly-Supervised CelebA

**Train on CelebA with Weak Supervision**
```
TODO
```

**Plot Results**
```
TODO
```
