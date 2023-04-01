
python scripts/sample_vqgan_transformer_short_videos.py \
	    --gpt_ckpt /userhome/42/msd21003/train_ckpt/ori_YoYo/version_0/checkpoints/best_checkpoint.ckpt --vqgan_ckpt /userhome/42/msd21003/ckpt/vqgan_ucf.ckpt\
	        --save /userhome/42/msd21003/train_ckpt/ori_YoYo/result --data_path /userhome/42/msd21003/ucf101 --batch_size 4 --n_sample 12 --resolution 128 \
		    --top_k 2048 --top_p 0.8 --dataset ucf101 --compute_fvd --save_videos
