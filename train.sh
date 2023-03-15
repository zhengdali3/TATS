python scripts/train_transformer.py --num_workers 32 --val_check_interval 0.5 --progress_bar_refresh_rate 500 \
                        --gpus 1 --sync_batchnorm --batch_size 3 --unconditional \
                        --vqvae /userhome/42/msd21003/ckpt/vqgan_ucf.ckpt --data_path /userhome/42/msd21003/ucf101 --default_root_dir /userhome/42/msd21003/train_ckpt \
                        --vocab_size 16384 --block_size 1024 --n_layer 24 --n_head 16 --n_embd 1024  \
                        --resolution 128 --sequence_length 16 --max_steps 2000000
