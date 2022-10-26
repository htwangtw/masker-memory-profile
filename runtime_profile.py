import time
import warnings

warnings.filterwarnings(action='ignore')

from pathlib import Path
import numpy as np

from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from nilearn.datasets import (fetch_atlas_difumo, fetch_atlas_schaefer_2018,
                              load_mni152_gm_mask, fetch_adhd)
from nilearn.image import load_img, mean_img, binarize_img, resample_img

from time import time


if __name__ == '__main__':
    results = Path(__file__).parents[1] / 'results'
    data_dir = Path(__file__).parents[1] / 'data'
    dataset = fetch_adhd(1, data_dir=data_dir)
    func = load_img(dataset.func[0])

    # fetch difumo
    atlas_difumo = fetch_atlas_difumo(dimension=256, resolution_mm=3,
                                      data_dir=data_dir, resume=True)

    # fetch schaefer 800
    atlas_schaefer_800_1mm = fetch_atlas_schaefer_2018(n_rois=800,
                                                       yeo_networks=7,
                                                       resolution_mm=1,
                                                       data_dir=data_dir)
    atlas_schaefer_800_2mm = fetch_atlas_schaefer_2018(n_rois=800,
                                                       yeo_networks=7,
                                                       resolution_mm=2,
                                                       data_dir=data_dir)
    # Grey matter
    gm_3mm = load_mni152_gm_mask(resolution=3)
    gm_1mm = load_mni152_gm_mask(resolution=1)

    # provide a mask or not
    epi_mask = binarize_img(mean_img(func))

    # NifitMasker
    # Mask for the data. If not given, a mask is computed in the fit step.
    # Optional parameters (mask_args and mask_strategy) can be set to
    # fine tune the mask extraction.
    # If the mask and the images have different resolutions, the images
    # are resampled to the mask resolution.
    # If target_shape and/or target_affine are provided, the mask is
    # resampled first. After this, the images are resampled to the
    # resampled mask.

    niftimasker_options = {
        'epi_pre': {
            'mask_img': epi_mask,
            'description': "Precompute EPI mask precomputed"
        },
        'epi_masker': {
            'mask_img': None,
            'description': "Masker compute EPI mask"
        },
        'gm_3mm': {
            'mask_img': gm_3mm,
            'description': "GM mask 3mm"
        },
        'gm_1mm': {
            'mask_img': gm_1mm,
            'description': "GM mask 1mm"
        }
    }

    for option in niftimasker_options.values():
        t1 = time()
        preprocessor = NiftiMasker(mask_img=option['mask_img'],
                                   memory=str(Path(__file__).parents[1] / 'nilearn_cache'),
                                   memory_level=1, verbose=0)
        data = preprocessor.fit_transform(func)
        t2 = time()
        print('\t' + option['description'] + f' :{(t2-t1):.4f}s')
        # time.sleep(10)

    # Extract atlas: difumo
    masker = NiftiMapsMasker(
        atlas_difumo.maps,
        mask_img=epi_mask).fit()

    t1 = time()
    timeseries = masker.transform(func)
    t2 = time()
    print(f'\tNiftiMapsMasker transform data with resample:{(t2-t1):.4f}s')

    # Extract atlas: difumo
    t1 = time()
    atlas_difumo_resampled = resample_img(
        atlas_difumo.maps,
        target_affine=epi_mask.affine,
        target_shape=epi_mask.shape)
    t2 = time()
    print(f'\tResample probseg before masker:{(t2-t1):.4f}s')
    masker = NiftiMapsMasker(
        atlas_difumo_resampled,
        mask_img=epi_mask).fit()

    t1 = time()
    timeseries = masker.transform(func)
    t2 = time()
    print(f'\tNiftiMapsMasker transform data no resample in masker object:{(t2-t1):.4f}s')

    # Extract atlas: schaefer 800 2 mm
    masker = NiftiLabelsMasker(
        atlas_schaefer_800_2mm.maps,
        mask_img=epi_mask).fit()

    t1 = time()
    timeseries = masker.transform(func)
    t2 = time()
    print(f'\tNiftiLabelsMasker transform data with resampling low res:{(t2-t1):.4f}s')

    # Extract atlas: schaefer 800 1 mm
    masker = NiftiLabelsMasker(
        atlas_schaefer_800_1mm.maps,
        mask_img=epi_mask).fit()

    t1 = time()
    timeseries = masker.transform(func)
    t2 = time()
    print(f'\tNiftiLabelsMasker transform data with resampling high res:{(t2-t1):.4f}s')

    # Extract atlas: schaefer 800 user resampled
    atlas_schaefer_800_3mm_resampled = resample_img(
        atlas_schaefer_800_2mm.maps,
        target_affine=epi_mask.affine,
        target_shape=epi_mask.shape,
        interpolation="nearest")

    masker = NiftiLabelsMasker(
        atlas_schaefer_800_3mm_resampled,
        mask_img=epi_mask).fit()

    t1 = time()
    timeseries = masker.transform(func)
    t2 = time()
    print(f'\tNiftiLabelsMasker transform data no resample in masker object:{(t2-t1):.4f}s')

	# Precompute EPI mask precomputed :0.8314s
	# Masker compute EPI mask :1.4853s
	# GM mask 3mm :0.7599s
	# GM mask 1mm :1.6995s
	# NiftiMapsMasker transform data with resample:98.9525s
	# Resample probseg before masker:96.2000s
	# NiftiMapsMasker transform data no resample in masker object:6.9591s
	# NiftiLabelsMasker transform data with resampling low res:5.2431s
	# NiftiLabelsMasker transform data with resampling high res:6.1257s
	# NiftiLabelsMasker transform data no resample in masker object:4.7030s
