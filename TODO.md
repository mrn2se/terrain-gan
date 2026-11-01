# What would be interesting to check

Open questions from [PAPER.md](PAPER.md) and its references, grouped by theme. Effort is a
rough estimate for the GTX 1650, and "training" means a run of about 3 hours per level.
Items are ordered within each group, with the best value first. Start with "First five".

## First five

These are cheap and could change what to do next.

1. Does the refiner use its noise? (A1)
2. Do the refiners beat plain bicubic upscaling? (A2)
3. How do slope and elevation distributions compare with real terrain? (A3)
4. How much worse is a refiner on generated input than on real input? (C1)
5. Do generated rivers drain properly? (A5)

## A. Measure what we have

No new training is needed.

- [ ] **A1. Is the noise ignored?** Render the same coarse input with different noise. Measure
  how much the output changes, per level. Isola et al. saw their generator ignore Gaussian
  noise. If ours does too, then the refiners are deterministic sharpeners. *Effort: an hour.*
- [ ] **A2. Bicubic and bilinear baseline.** Compare each refiner with plain upscaling on elevation
  error, and on slope and roughness. Demiray et al. use this baseline for 4× elevation GANs. A GAN
  should lose on pixel error but win on realism, and Ledig et al. explain why. *Effort: an hour.*
- [ ] **A3. Terrain statistics against real.** Compare hypsometric curves, slope histograms and
  radial power spectra at each level over many tiles. This replaces the noisy single-batch
  roughness. Include generated-input results. *Effort: half a day.*
- [ ] **A4. Multi-scale sliced Wasserstein distance.** Implement the Laplacian-pyramid SWD of
  Karras et al. for heightmaps, per level and through all checkpoints. It would also show
  whether it detects the worm artefacts in `runs/regional_pixelnorm`, which roughness missed.
  Could add ∞-FID or ScaleInv FID for comparison with ALIS and InfinityGAN, although
  Inception features may suit heightmaps poorly. *Effort: 1 day.*
- [ ] **A5. Hydrology check.** Fill sinks, compute flow accumulation and compare drainage
  density, internal basin fraction and longest river length with real terrain. Would make the
  "rivers are missing" claim a measurement. Needs a library such as `pysheds` or `richdem`.
  *Effort: 1 day.*
- [ ] **A6. Beyond 1,400 km.** Measure how elevation correlation falls with distance for real
  Earth and generated worlds. Theory predicts generated worlds lose all correlation past level
  0's reach. Compare latitude-free statistics only, as real Earth is finite. *Effort: half a day.*
- [ ] **A7. Zoom consistency.** Render every level over the same area, measure mismatch and look
  for visible jumps when zooming on the map. *Effort: half a day.*
- [ ] **A8. Repetition across worlds.** Do many seeds produce the same few continent forms?
  Compare tiles across seeds, like MS-SSIM in Karras et al. *Effort: half a day.*
- [ ] **A9. Copying in refiners.** Extend `evaluate.py` beyond level 0, and rerun level 0 with
  many more than 24 tiles. *Effort: half a day.*
- [ ] **A10. Human test.** Can people pick real tiles from generated ones? InfinityGAN uses a user
  study. The map's seed `-1` mode could serve as a base. *Effort: 1 day plus participants.*

## B. Understand training

Needs training runs. Use a shorter run, around 6,000 steps, where this is enough.

- [ ] **B1. Separate the pixel normalisation and R1 changes.** Train a refiner with each change
  on its own. Karras et al. found pixel normalisation mostly harmless, so R1 being the real cause
  is plausible. *Effort: 2 short runs.*
- [ ] **B1b. Look inside the artefacts.** Use the unit ablation of Bau et al. (GAN Dissection) on
  `runs/regional_pixelnorm` to find units causing worms, and on level 0 for its rings. This needs
  no training and may explain the cause where B1 only separates it. *Effort: 1 day.*
- [ ] **B2. Level 0 without pixel normalisation, with lower R1.** May remove its rings and bumpy
  texture. Refiners would not need retraining. *Effort: 1 training.*
- [ ] **B3. Slope gain against R1.** The ×10 slope channels probably strengthen R1. Test different
  gains and R1 weights, and record the discriminator's real and fake separation. *Effort: a few short runs.*
- [ ] **B3b. R3GAN training recipe.** Try the relativistic loss with both R1 and R2 of Huang et al.,
  on one refiner first. They report it removing the need for many GAN tricks, and it may
  help the mode coverage at level 0. Keep generators padding-free. *Effort: 1 or 2 trainings.*
- [ ] **B4. Converged single-scale baseline.** The cascade was only compared with a model stopped at
  7,300 steps. Train one to 20,000 or more at 0.93 km per pixel and use the A3 and A4 metrics. This
  directly tests the paper's central claim. *Effort: 1 training.*
- [ ] **B5. Longer training.** Do metrics and rivers improve by 100,000 steps? Theory says rivers
  should not, which makes this a useful test. *Effort: about 14 hours per level.*

## C. Close the gap between real and generated input

- [ ] **C1. Measure the gap.** Use A3 and A4 for refiners on real and on generated input. Sets the
  baseline for the rest of this group. *Effort: half a day.*
- [ ] **C2. Conditioning augmentation.** Try stronger noise at level 1 and Gaussian blur at level 2, as
  in Ho et al. Could train across a range of strengths and choose afterwards, which they also do.
  *Effort: 2 or 3 trainings.*
- [ ] **C3. Train on generated input.** Feed level 0 output to level 1 with a discriminator judging
  output only. Beckham and Pal name joint training as the next step for their similar problem.
  *Effort: 1 or 2 trainings, new code.*

## D. Add long-range structure

- [ ] **D1. River and ridge inputs.** Follow Guérin et al.: extract rivers and ridges automatically from
  real terrain, train a level to generate them and condition the refiner on them. Most direct
  attack on the main weakness. *Effort: several days.*
  Baier et al. found spatially adaptive normalisation better than concatenating such map inputs.
- [ ] **D2. Learned erosion as a post-process.** Guérin et al. train a conditional GAN to imitate erosion
  simulation. Keeping tiles seamless needs that network unpadded too. *Effort: several days.*
- [ ] **D3. Larger reach.** Try dilated convolutions, more stages or a slower latent field. Measure the
  receptive field, and whether A5 improves. *Effort: 1 or 2 trainings each.*
- [ ] **D4. Slower level-0 fields.** Add fields beyond the 128 px lattice, as a simple version of ALIS's
  anchors, and check with A6. *Effort: 1 training.*
- [ ] **D5. Style modulation.** Compare concatenated latents with spatial style modulation, as in ALIS and
  InfinityGAN. Must stay local to stay seamless. *Effort: a few days.*
- [ ] **D6. Terrain types.** Condition on a generated biome or roughness map, in the spirit of the
  segmentation idea of Beckham and Pal. *Effort: several days.*

## E. Data

- [ ] **E1. Real SRTM level.** Add a level towards 30 m using SRTM, the data Guérin et al. use and the
  original idea of this project. Handle voids and storage. *Effort: several days plus data.*
- [ ] **E2. Ocean-aware refiners.** Refiners saw mostly land, and sea floors blur. Try a separate model,
  a higher ocean share, or masking. *Effort: 1 training.*
- [ ] **E3. Ice surface.** Compare the "bed" files against ETOPO "surface" files, and whether high-latitude
  results change. Crops past ±72° are unused for now. *Effort: 1 training plus data.*
- [ ] **E4. Top-level data.** Try 64 px crops at level 0, as Earth only has a few hundred independent
  128 px crops. *Effort: 1 training.*
- [ ] **E5. Generalisation.** Hold out a region from training, and compare with A3 and A4 there.
  *Effort: 1 training.*

## F. Compare with other approaches

- [ ] **F1. Classic baselines.** Compare with Perlin or diamond-square noise, as named by Beckham and Pal,
  using A3 and A5. Cheap, and shows whether the GAN beats simple methods at all. *Effort: half a day.*
  Nunes et al. also compare GAN variants on Earth-like game maps, and Naik et al. offer a VAE and
  cGAN approach, both useful for choosing further baselines.
- [ ] **F2. Diffusion refiner.** Replace one refiner with a diffusion model as in Ho et al. Slower, and
  seamless sampling is an open design question here. *Effort: 1 week.*
- [ ] **F3. Prior work.** Run ALIS or InfinityGAN on this data if their code fits in 4 GB, and check the
  later literature (StyleGAN2 and its droplet artefacts, in particular, relate to B1). *Effort: days.*

## G. Use the result

- [ ] **G1. 3D view.** Render in 3D, as Beckham and Pal do in Unity. Flaws hidden by shaded relief may show.
- [ ] **G2. Texture.** Generate surface colour from height, like their pix2pix stage.
- [x] **G3. Browser version.** Done: `web/` runs all three levels and matches PyTorch in its tests.
- [ ] **G4. Publish the browser version** on GitHub Pages, and try it on phones and with WebGPU.
- [ ] **G5. Real-Earth mode in the browser.** The server's seed `-1` needs the 2 GB data, so this
  would need a small coarse copy or pre-made tiles.
