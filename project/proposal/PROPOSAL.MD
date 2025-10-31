# Neural Approximation of Generative Art for Real-Time Rendering


Student Name: Fernando Nogueira  
Student ID: 300 195 829  
Date: October 31, 2025

## Broad Project Area

Computer Vision / Graphics  

## Research Question 

Can MLPs accurately approximate the output of computationaly expensive generative art shader as a function of normalized spatiotemporal coordinates $(x, y, t)$, achieving faster inference while preserving visual fidelity across dynamic scenes?

## Dataset

## Dataset

A synthetic spatiotemporal dataset is generated from a simplified variant of the jellyfish animation shader (adapted from https://x.com/yuruyurau/status/1966891530729541865 and a simplified Shadertoy implementation https://www.shadertoy.com/view/tXlBW8). This produces grayscale intensity maps of oscillating, wireframe jellyfish structures over a 12.5 s cycle, with reduced computational complexity compared to the original.

To manage rendering demands, 750 evenly spaced frames are subsampled across a temporal range of 12.5 seconds. Per frame, 32,768 stratified points are sampled within a bounding box of the active canvas region, using a 256×256 grid to balance light/dark distributions and capture fine details like tendrils. This results in ~6.5 million (x, y, t, I) tuples, with I $\in$ [0, 1] as ground-truth intensity.

Spatial coordinates (x, y) are centered and scaled to [-1, 1]; t is normalized to [0, 1]. Stratification ensures coverage of sparse, high-contrast features.

I will generate a synthetic spatiotemporal dataset 
(inspired by https://x.com/yuruyurau/status/1966891530729541865), rendered 

from a variant of the jellyfish animation shade
A self-generated synthetic dataset from a variant of aspecific Processing shader (jellyfish animation from https://x.com/yuruyurau/status/1966891530729541865).


I'll sample from $N_f$ frames of $t$ over a 12.5 second period, computing $N_p$ particle positions per $t$. 

For each frame, I'll use stratified sampling from a bbox grid of the relevant area of the canvas to balance light and dark pixels, yielding $(x, y, t, intensity)$ tuples where intensity $\in [0,1]$. 

Coordinates will be shifted to the center and scaled to $[-1, 1]$; $t$ normalized to $[0,1]$ over the 12.5s period. 



Shadertoy page: https://www.shadertoy.com/view/tXlBW8

NOTE: This page is very computationally expensive to render.

## Algorithm(s)

I'll implement a multi-layer perceptron (MLP), treating the problem as regression from (x, y, t) to grayscale intensity. As an extension, I'll try training a small convolutional neural network (CNN) for spatial awareness.

In addition, I'll try pruning the model to reduce the number of parameters and the inference time.

## Model Evaluation  
   
Models will be trained on 80% of the data (random split), with 10% for final validation and 10% test. Using PSNR as the evaluation metric.

# Prework

![Training Results](.cache/frames/frames_comparison.png)
![Train Curves](.cache/frames/training_curves.png)

I have initially attempted to train it using a 5 layer MLP with 1024 neurons per layer.


80/20/10 random split
PSNR 9 to 14db on validationn dataset 
after 1000 epochs with batch size of 32k (trained for 2 hours on an RTX 3090)