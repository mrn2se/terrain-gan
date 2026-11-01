# Endless Seamless Terrain from a Cascade of Small GANs

*Project report, November 2026. Written with Claude Opus 5.5. Not peer reviewed.*

## Abstract

We describe a system that generates unbounded elevation maps from a seed and a
position. Three small convolutional GANs are trained on the ETOPO 2022 global
relief model: an unconditional generator at 14.8 km per pixel and two conditional
refiners reaching 0.93 km per pixel. All randomness is a hash of seed and position,
and the generators use no padding, so tiles rendered independently agree to about
4×10⁻⁷ in normalised height without blending. The whole cascade was trained on a
4 GB consumer GPU in about 8 hours of training time. A single-scale model produced
homogeneous terrain, whereas the cascade matches real Earth's tile-to-tile spread in
land fraction (0.36 vs. 0.36) and height variation (686 vs. 665 m). Generated terrain
is plausible, but lacks long rivers and continuous ridges, and our evaluation is
limited: there is no FID score, no user study and no controlled ablation.

## 1. Introduction

Procedural terrain usually comes from hand-tuned noise or erosion simulation.
Learned models can capture the look of real terrain, but a standard GAN maps one
vector to one finite image, whose edges do not match its neighbours. Applications
such as games and maps need the opposite: any window of an endless world, on
demand, always the same for a given position.

Terrain also spans many scales. A viewer zooming out expects oceans and continents,
and zooming in expects ridges and valleys. One network covering the whole range needs
very large crops, and so more memory than the 4 GB available here.

Our contributions are modest and practical:

1. A rendering scheme giving exact tile agreement across a multi-level GAN cascade.
2. A cascade small enough for a GTX 1650, with a browsable map that can compare
   against the ground truth at each zoom.
3. A record of what failed, including one failure that a single metric hid.

## 2. Related work

Spatial GAN (Jetchev et al., 2016) and PSGAN (Bergmann et al., 2017) showed that
fully convolutional generators fed with spatial noise can make textures of arbitrary
size. ALIS (Skorokhodov et al., 2021) and InfinityGAN (Lin et al., 2022) extend the
idea to unbounded natural scenes. Our level-0 generator follows this family, and is
closest in spirit to a position-keyed spatial GAN.

GANs have been applied to terrain before, for example by Beckham and Pal (2017), and
in conditional sketch-to-terrain authoring by Guérin et al. (2017). Our refiners are
conditional GANs in the sense of pix2pix (Isola et al., 2017) and act like the
super-resolution GANs of Ledig et al. (2017). The cascade structure resembles
progressive growing (Karras et al., 2018), though our levels are trained separately
and remain separate at inference, and cascaded diffusion models (Ho et al., 2022).

## 3. Data

We use the "bedrock" variant of ETOPO 2022 (NOAA NCEI, 2022) at 30 arc-seconds:
43,200 × 21,600 samples of land and sea-floor height. Ice sheets are absent from this
variant. Note that this is a compiled global model, not raw SRTM data.

**Levels.** Block means with factors 16, 4 and 1 give about 14.8, 3.7 and 0.93 km
per pixel at the equator. A 128 px crop covers about 1,900, 475 and 120 km.

**Height transform.** Heights span about −11,000 to +8,800 m but most of the signal
is near sea level, so training uses

    x = asinh(h / 300 m) / asinh(9000 / 300)

This keeps sea level at zero and gives lowlands more of the range.

**Sampling.** Crop latitude is drawn uniformly in area within ±72°. The map projection
narrows ground distance per pixel towards the poles, so each crop reads about
1/cos(lat) columns and resamples them to square, giving about 1:1 ground scale at
the crop's centre. Random rotations and flips follow. Level 0 accepts every crop to
learn Earth's real land and ocean mix. Refiners skip 90% of crops below 30% land,
concentrating training on land.

**Small top level.** At 14.8 km per pixel, Earth has only a few hundred non-overlapping
128 px crops, so memorisation is a real risk there. See section 6.3.

## 4. Method

### 4.1 Position-keyed randomness

Latent vectors and noise come from a splitmix64-style hash of seed, layer tag, channel
and lattice position, converted to Gaussians with the Box-Muller transform. Training
uses the same latent function for level 0 and ordinary random noise for the rest,
which has the same distribution.

### 4.2 Exact tile agreement

Three rules together give tile independence:

- Every filter is unpadded, so each output depends only on its local inputs.
- No normalisation uses batch or image statistics.
- Output is cut to world coordinates through a known lattice offset.

Each stage is nearest-neighbour 2× upsampling, a fixed [1 2 1] blur and two 3×3
convolutions, so it maps n samples to 2n − 6. The world position of the first output
follows from the shifts: −S/4 from upsampling and 3·S/2 from unpadded filters at
input stride S. For the level-0 generator with S = 16, 8, 4, 2:

    output size  = 16n − 90
    first pixel  = 16·i₀ + 37

where i₀ is the first latent cell. Any requested window expands to enclosing cells,
renders and crops. Large windows are rendered as 512 px chunks, which agree with each
other in the same way.

### 4.3 Level 0: world generator

Input is a latent grid with one cell per 16 px: 48 independent channels per cell, plus
16 channels of a smooth field interpolated from a lattice 8 cells apart, with smoothstep
weights rescaled to keep unit variance. Two 1×1 layers map this to 192 channels. Four
stages then reduce to 128, 64 and 32 channels (first stage 192), with noise injected and
pixel normalisation applied after each convolution. Total: 1.22 M parameters. Measured
through gradients, one output pixel depends on 6 latent cells across (about 96 px or
1,400 km), so nothing directly ties together terrain further apart than that, apart
from the shared smooth field.

### 4.4 Levels 1 and 2: refiners

A refiner takes a coarse height map and 16 noise channels per coarse pixel. Two unpadded
convolutions and two upsampling stages produce a residual added to the bilinearly
upsampled input, with the residual layer initialised to zero. This gives 4n − 34 output
pixels, whose first pixel is 4·i₀ + 17, where coarse pixel i covers output pixels 4i to
4i + 3. We check this alignment directly in a test: an untrained refiner must keep a
linear ramp's value at each pixel's ground position. Each refiner has 0.42 M parameters
and depends on about 10 coarse pixels across, and has no pixel normalisation (section 5).

### 4.5 Discriminator and losses

The discriminator is a residual convolutional network (4.76 M parameters) with
minibatch standard deviation. Its input is height plus forward differences in x and y
multiplied by 10, because slopes carry much of terrain's appearance. Refiner
discriminators also see the upsampled condition and (output − condition) × 10.

We use the non-saturating logistic GAN loss with a lazy R1 penalty (Mescheder et al.,
2018), applied every 8 steps. Refiners add a consistency term: the L1 distance, weight
10, between the output's block means and the coarse input, computed in metres and then
transformed, exactly as the coarse data is made. Gaussian noise (σ = 0.01) on the coarse
input during training aims to tolerate imperfect generated input.

### 4.6 Training

Adam with learning rate 2×10⁻⁴, β = (0, 0.99), batch size 16, 128 px crops, and an
exponential moving average of generator weights (0.999). Each level ran 20,000 steps
at 1.9 to 2.2 steps/s on a GTX 1650 (4 GB). Levels were trained independently from real
data, so a level can be retrained without touching the others.

## 5. Experiments that failed

**Divergence.** The first single-scale model used learning rate 10⁻³, R1 weight 1 every
16 steps and no normalisation. Discriminator outputs reached ±40,000 and heights about
10¹² m. Adding pixel normalisation, lowering the learning rate to 2×10⁻⁴ and using R1
weight 10 every 8 steps restored stability. These were changed together.

**Homogeneous terrain.** That stable model, trained at 0.93 km per pixel on land-biased
120 km crops, was stopped after 7,300 steps. Its output had blobs of the same size
everywhere, and it was this that motivated the cascade. Training at that one scale was
never carried further, so the comparison is against an early single-scale result and
not a converged one.

**Worm artefacts hidden by a metric.** The first refiner used pixel normalisation and
R1 weight 10. At 6,000 steps, its roughness had reached 0.86× real and looked healthy,
but the images showed grid-aligned, worm-like contours everywhere, including on the
sea floor (Figure 1). The score was counting artefacts as detail. Discriminator loss
was also close to the chance level of 1.39, though the difference was obvious by eye.

We removed pixel normalisation from the refiner and lowered R1 to 1. By 2,000 steps the
artefacts were gone. Pixel normalisation can inflate weak activations, which would
produce such isolated structures, but because two changes were made at once this cause
is an unproven hypothesis. The level-0 generator keeps both original settings and shows
faint ring shapes that may be related.

![Worm-like artefacts in the pixel-normalised refiner at step 6,000](docs/figures/pixelnorm_artefact.png)

*Figure 1. Worm-like artefacts from the first refiner run, about 3,800 km wide.*

## 6. Results

![Level 0 at step 20,000](docs/figures/world_level.png)

*Figure 2. Level 0 only, seed 0, about 15,200 × 7,600 km.*

![Level 1 on generated input](docs/figures/regional_level.png)

*Figure 3. Levels 0 and 1, seed 0, about 3,800 × 1,900 km, showing the generated
world refined.*

![Regional refiner samples](docs/figures/regional_samples.png)

*Figure 4. Level 1 on real coarse input. Rows: upsampled input, refined output, real.*

![Local refiner samples](docs/figures/local_samples.png)

*Figure 5. Level 2 on real coarse input. Rows: upsampled input, refined output, real.*

![All levels](docs/figures/island_all_levels.png)

*Figure 6. All three levels, about 950 km wide, `--lon 20.7 --lat -14`, seed 0.*

### 6.1 Seamlessness

Comparing a large render with separately rendered parts gave these largest differences
in normalised height, on trained models:

| Model | Maximum difference |
|---|---|
| Level 1 cascade | 1.3 × 10⁻⁶ |
| All three levels | 3.6 × 10⁻⁷ |

This is floating-point noise. The test suite also checks untrained models at world
positions near 10⁹ pixels, to a tolerance of 10⁻⁴.

### 6.2 Variety at level 0

Over 256 tiles at step 20,000 (spread is the standard deviation across tiles):

| Statistic | Generated | Real | Distance between distributions |
|---|---|---|---|
| Land fraction spread | 0.36 | 0.36 | 0.077 |
| Height variation spread (m) | 686 | 665 | 219 |
| Roughness spread (m/km) | 2.36 | 3.79 | 1.26 |

Distance is the 1-Wasserstein distance over tiles. Land and height match well. Roughness
spread is too low, staying between 2.4 and 2.8 from step 8,000 on, so more training at
this level may not fix it. The readings vary between checkpoints, and land distance was
lowest at step 16,000 (0.044), while we use the final step.

### 6.3 Memorisation at level 0

For 24 varied generated tiles, we found the best normalised correlation anywhere on
real Earth, across eight orientations and six latitude bands. For the baseline, we did
the same with 24 real crops, excluding their own neighbourhoods.

| | Median best match | Highest | Above 0.95 |
|---|---|---|---|
| Generated | 0.805 | 0.977 | 12% |
| Real baseline | 0.807 | 0.952 | 4% |

Medians agree. The upper tail is slightly higher for generated tiles, but 12% and 4% are
about three tiles and one. The six closest pairs, inspected visually, were different
terrain sharing a similar coast orientation, not copies. This check compares overall shape
on small samples. It was not run on refiners, where data is much more plentiful.

### 6.4 Refiner detail

Roughness ratios, generated over real, on one batch of real coarse input per checkpoint:

| Step | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | 18k | 20k |
|---|---|---|---|---|---|---|---|---|---|---|
| Level 1 | 1.08 | 0.87 | 0.66 | 0.98 | 0.90 | 1.10 | 0.94 | 0.98 | 1.06 | 0.97 |
| Level 2 | 0.85 | 1.01 | 0.99 | 1.12 | 0.96 | 1.13 | 0.94 | 0.93 | 0.94 | 1.02 |

These are single-batch readings, so swings such as 0.66 are mostly noise, and section 5
shows the measure can be fooled. Final block-mean mismatch with the input was about 0.005.

### 6.5 Qualitative assessment

Visually, generated results include rugged massifs, basins, coastal plains and offshore
islands, and refined outputs on real input are often hard to tell apart from real at a
glance. Weaknesses are consistent:

- Long rivers and continuous ridge lines rarely form, while real terrain shows them clearly.
- Long coastlines are too clean, and sea floors blur at finer levels, as refiners saw
  mostly land.
- Level-0 highlands are bumpy, some landmasses too round, and real trenches are mostly absent.
- Refined output on generated input is weaker than on real input.

### 6.6 Rendering speed

On the GTX 1650, most 256 px map tiles took about 0.05 s, and the most zoomed-out took about
1.4 s. These were single measurements and exclude browser time.

## 7. Discussion

**Why the cascade helps.** Variety at continental scale comes from level 0, which sees
1,900 km crops, whereas the single-scale model never saw more than 120 km. This fits with
the visual results, but no converged single-scale baseline was trained.

**Why rivers are missing.** Rivers and ridges are organised over long distances. Each
refiner sees only about 10 coarse pixels, and level 0 has too little resolution, so no level
is well placed to create them. Explaining this properly would need a dedicated experiment.
Longer receptive fields or a learned or simulated flow step are natural next steps.

**Conditioning gap.** Refiners only ever saw real coarse maps during training. Noise
augmentation is a weak substitute for generated input. Training later levels on the output
of earlier levels might narrow this gap.

## 8. Limitations of this evaluation

- No FID or other standard score, and no human perceptual study.
- No controlled ablation. Several changes were made together, and causes are hypotheses.
- One training run per configuration, and all visual checks by the authors.
- Variety statistics describe tile-level summaries, not terrain structure, and the
  memorisation check uses just 24 tiles and only covers level 0.
- Training stopped at 20,000 steps per level. Longer runs were not explored.
- Figures come from a few seeds and positions, chosen to include land.

## 9. Conclusion

A cascade of three small unpadded GANs with position-keyed randomness can generate an
endless, exactly seamless world spanning continents to about 1 km per pixel, within 4 GB
of GPU memory and about 8 hours of training time. The cascade addressed the homogeneity of a
single-scale model. Producing organised long-distance features such as river networks is
the main open problem. Images alone caught a failure that our roughness metric missed, so
metrics and visual checks should be used together.

## References

These citations were written from memory and should be checked before any formal use.

- Beckham, C., Pal, C. (2017). A step towards procedural terrain generation with GANs. arXiv:1707.03383.
- Bergmann, U., Jetchev, N., Vollgraf, R. (2017). Learning texture manifolds with the Periodic Spatial GAN. ICML.
- Guérin, É., et al. (2017). Interactive example-based terrain authoring with conditional generative adversarial networks. ACM Transactions on Graphics.
- Ho, J., et al. (2022). Cascaded diffusion models for high fidelity image generation. JMLR.
- Isola, P., et al. (2017). Image-to-image translation with conditional adversarial networks. CVPR.
- Jetchev, N., Bergmann, U., Vollgraf, R. (2016). Texture synthesis with spatial generative adversarial networks. arXiv:1611.08207.
- Karras, T., et al. (2018). Progressive growing of GANs for improved quality, stability, and variation. ICLR.
- Ledig, C., et al. (2017). Photo-realistic single image super-resolution using a generative adversarial network. CVPR.
- Lin, C. H., et al. (2022). InfinityGAN: Towards infinite-pixel image synthesis. ICLR.
- Mescheder, L., Geiger, A., Nowozin, S. (2018). Which training methods for GANs do actually converge? ICML.
- NOAA National Centers for Environmental Information (2022). ETOPO 2022 15 Arc-Second Global Relief Model. doi:10.25921/fd45-gt74.
- Skorokhodov, I., Sotnikov, G., Elhoseiny, M. (2021). Aligning latent and image spaces to connect the unconnectable. ICCV.
