The code for focused attention is using TATS as the baseline model, and is constructed based on the open-sourced repository at (https://songweige.github.io/projects/tats)

Modified:
gpt.py, sample_vqgan_transformer_short_videos.py, train_transformer.py was modified for the inference and training of focused attention embedded transformer.
compute_fvd.py data.py was modified for customized dataset processing.

Created:
focus_testing.ipynb, FVD testing.ipynb was created for testing focused attention module and FVD score generation.
focused attention module is stored as a inherited class of torch.cuda.autograd.function with overwritten forward and backward function in focus.py
train.sh and infer.sh are scripts to train transformer and infer results.
