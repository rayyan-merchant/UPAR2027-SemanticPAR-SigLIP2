# Third party models and data

The project uses general-purpose pretrained Google SigLIP2 and Meta DINOv3 via the maintainer-supported timm implementation. DINOv3 license, timm Apache 2.0 and relevant notices are retained in `dino/licenses/`; the older DINOv2 notice remains because fallback code/history is preserved. This repository does not change upstream licenses or grant additional permission to redistribute weights.

Official UPAR annotations and source images from PA100K, PETA and Market1501 retain their original licenses, research/noncommercial limitations and citation obligations. Obtain them from authorized sources. No training images, feature caches, sample-level private predictions or model weights are included in the assignment source ZIP. Private same-owner cloud archives were not published or relicensed. Cite the UPAR challenge, HydraPlus-Net/PA100K, PETA and Market1501 along with the foundations when reporting the work.

Upstream references: https://github.com/speckean/UPAR-Challenge-2027 ; https://huggingface.co/google/siglip2-base-patch16-naflex ; https://github.com/facebookresearch/dinov3 ; https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m ; https://github.com/facebookresearch/dinov3/issues/375 .

No blanket repository license is newly assigned by this cleanup. Existing third-party terms and notices remain authoritative.
