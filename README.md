# SpectMatch
This Repository contains the implementation of the SpectMatch adaptation of FixMatch, first introduced in [1] and further proposed as a viable system in [2]. Notebooks Present in this repository contain model implementations, preprocessing notebooks and metrics.

# Requirements
1. Python 3.6 +
2. Librosa 0.9.0 +
3. torchaudio
4. timm 0.4.5 (For AST), timm 0.6.5+ (For ViT)
5. numpy
6. tqdm
7. wget
8. scikit-learn
9. seaborn
10. torchvision
11. Matplotlib

# How to run repo
Running AST for classroom observation for 1650 examples labeled
````
python train.py --dataset classaudio_AST --num-labeled 1650 --arch ast --amp --batch-size 2 --lr 0.001 --eval-step 1 --total-steps 1 --expand-labels --use-ema --seed 5 --threshold 0.95 --num-workers 4 --out  results/classaudio_AST@1650.5/log

````

# References
[1] A. Chanchal and I. Zualkernan, “Exploring semi-supervised learning for audio-based automated classroom
observations,” 19th International Conference on Cognition and Exploratory Learning in Digital Age (CELDA),
Nov. 2022. [Online]. Available: https://www.celda-conf.org/wp-content/uploads/2022/11/3_CELDA2022_F_184.pdf.

[2]

# Referenced Repositories
https://github.com/kekmodel/FixMatch-pytorch

https://github.com/YuanGongND/ast



