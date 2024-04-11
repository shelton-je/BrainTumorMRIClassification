# BrainTumorMRIClassification
With this project we hope to increase the sensitivity and accuracy in our model to correctly identify the presence of a tumor in a MR image by using a combination of a GAN and transfer learning over normal approaches, such as only using a CNN with no extra data or other basic classification methods.

# Methods
### Generating fake images for data augmentation 
- Clone Nvidias's offical pytorch repo https://github.com/NVlabs/stylegan3 
- Follow there readme.md for setting up enviroment and dependinces 
- Use their file gen_images.py with our trained networks to generate images 
#### Example lines for generating images:
- python gen_images.py --network \tumor_weights.pkl --noise-mode random --outdir \output-dir --seeds 10-110
- python gen_images.py --network \normal_weights.pkl --noise-mode random --outdir \output-dir --seeds 210-310

# Image Source
https://www.kaggle.com/datasets/mhantor/mri-based-brain-tumor-images

# Trained Stylgan3 networks for generating fake tumor/normal images
https://drive.google.com/drive/folders/1UBhrHxwNu4fwmg0dEso-pECfL2W83ZSg?usp=sharing
### Training parameters
- discriminator-gamma = .2
- batch = 16 (can do 32 if you have something like a a100 or better)
- cdg = stylegan3-t
- mirror = True
- cbase = 16384
- snap = 10
- training time = 23 hours for both networks
- Tensorflow Version 2.15
