# What would be interesting to check

Open questions from [PAPER.md](PAPER.md) and its references. "Training" means a run of
about 3 hours per level on the GTX 1650. The easy items come first; larger projects are
listed under [Open ideas](#open-ideas). Item names (A1, B2, ...) are kept from earlier versions
of this list, as the paper and commits refer to them.

## Done

`measure.py` covers A1, A2, A3, A5, A6, A7, A8, C1 and F1; section 6.7 of the paper has the numbers.

- [x] **A1. Is the noise ignored?** No. Two renders of the same real input share only about 45% of
  their added detail. The latent field gives most of the variation, the per-layer noise little.
- [x] **A2. Bicubic and bilinear baseline.** Refiners have larger height error than bicubic
  (57 against 44 m at level 2), as expected, but much more realistic slopes and fine-scale spectra.
- [x] **A3. Terrain statistics against real.** Hypsometric curves, slope histograms and power spectra
  per level. Level 0 lacks flat plains and has a soft shelf edge. Refiners on real input add too
  little detail, at about 77% of real slope.
- [x] **A5. Hydrology check.** Generated land at level 2 has twice the real share in closed
  depressions and drainage areas less concentrated into large rivers. Fractal noise is worse still.
- [x] **A6. Beyond 1,400 km.** Generated worlds differ too much at 1,000 to 3,000 km and reach their
  maximum difference sooner than real Earth.
- [x] **A7. Zoom consistency.** Block means of generated levels stay within 16 to 35 m RMS of the
  level above.
- [x] **A8. Repetition across worlds.** Random generated tile pairs are slightly more alike than
  real ones (median correlation 0.46 against 0.40), not enough to suggest collapse.
- [x] **B2. Level 0 without pixel normalisation.** Less stable early on, and less bumpy, but rings and
  round landmasses remain. Kept on branch `world-no-pixelnorm`; `models/world.pt` stays.
- [x] **B3. Stronger R1 at level 0.** R1 40 instead of 10: level 0 slope ratio 1.20 → 1.13, and
  every level comes closer to real. Visually more plains, but speckled coastal flats. Kept as
  `models/experiments/world_r1_40.pt`.
- [x] **B4. Converged single-scale baseline.** Trained to 20,000 steps at 0.93 km per pixel: tiles far more
  alike than real, land and sea alternating every 100 km, grainy look. The cascade is clearly better.
- [x] **C1. Gap between real and generated input.** On generated input, slopes come out near the real
  level (ratio 1.0), against 0.77 on real input: two errors cancel, as level 0 is too rough.
- [x] **C2. Blurred coarse input at level 2.** Slopes on real input rise from 77% to 92% of real, but
  the refiner adds more detail to every input, so the cascade becomes too rough (127%) with
  today's level 0. Kept as `models/experiments/local_blur.pt`; worth retrying once level 0 is smoother.
- [x] **D4. Slower level-0 fields.** Mixed. 16 extra channels on a 512 px lattice bring the distance
  curve close to real and reduce repetition, but add speckled shallow flats and sharp-edged
  plateaus. Kept as `models/experiments/world_slow.pt`; `models/world.pt` stays the default.
- [x] **F1. Classic baseline.** Fractal noise with the real spectrum: slopes and drainage much less
  realistic than the GAN at every level.
- [x] **G3. Browser version.** `web/` runs all three levels and matches PyTorch in its tests.
- [x] **G4. Publish the browser version.** On GitHub Pages under `preview/`. iPhones crashed after a
  tile or two; with 256 px chunks there, both CPU and WebGPU work on an iPhone.

## Next, easy

Each is one short training run or less, with no new methods.

- [ ] **C2b. Stronger input noise at level 1**, as in Ho et al. Level 2 with blur is done (C2 above).
- [ ] **B1. Separate the pixel normalisation and R1 changes** in a refiner, with two short runs.
- [ ] **B3c. More R1 weights** at level 0 (20, 80), and R1 against slope gain in a refiner.
- [ ] **E2. Ocean-aware refiners.** Retrain one refiner with a higher ocean share, and compare sea floors.
- [ ] **E4. 64 px crops at level 0.** Earth has only a few hundred independent 128 px crops.
- [ ] **A9. Copying in refiners.** Extend `evaluate.py` beyond level 0, with more than 24 tiles.
- [ ] **A4. Multi-scale sliced Wasserstein distance** (Karras et al.) as one summary score per level.

## Open ideas

Larger projects: several days, new methods or more data.

- **D1. River and ridge inputs.** Extract rivers and ridges from real terrain, generate them at a coarse
  level and condition the refiners on them, as Guérin et al. do. The most direct attack on the
  drainage gap shown by A5. Baier et al. found spatially adaptive normalisation better than
  concatenating such map inputs.
- **D2. Learned erosion as a post-process.** Guérin et al. train a conditional GAN to imitate erosion
  simulation. To stay seamless the network must be unpadded.
- **D3. Larger reach.** Dilated convolutions or more stages, checked with A5 and A6.
- **D5. Style modulation** in place of concatenated latents, as in ALIS and InfinityGAN, kept local
  to stay seamless.
- **D6. Terrain types.** Condition on a generated biome or roughness map.
- **B3b. R3GAN training recipe.** The relativistic loss with R1 and R2 of Huang et al., on one
  refiner first; may help mode coverage at level 0.
- **B1b. Look inside the artefacts** with the unit ablation of Bau et al. (GAN Dissection), for the
  worms in `runs/regional_pixelnorm` and the rings at level 0.
- **B5. Longer training**, about 14 hours per level. Theory predicts rivers will not appear.
- **C3. Train on generated input**, with a discriminator judging output only (Beckham and Pal).
- **E1. Real SRTM level** towards 30 m per pixel; needs voids handled and much more storage.
- **E3. Ice surface.** ETOPO "surface" files instead of "bed", and crops past ±72°.
- **E5. Generalisation.** Hold out a region from training and compare there.
- **F2. Diffusion refiner** as in Ho et al.; seamless sampling is an open design question.
- **F3. Prior work.** Run ALIS or InfinityGAN on this data if they fit in 4 GB.
- **A10. Human test.** Can people pick real tiles from generated ones? The seed `-1` mode could serve.
- **G1. 3D view**, where flaws hidden by shaded relief may show.
- **G2. Texture.** Surface colour from height, like the pix2pix stage of Beckham and Pal.
- **G5. Real-Earth mode in the browser**, from a small coarse copy of the data or pre-made tiles.
