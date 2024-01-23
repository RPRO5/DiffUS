import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.improved_denoising_diffusion_pytorch import (Improved_Diffusion,
                                                      Trainer, Unet,
                                                      set_seed)

# Code will be upload

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    sys.stdout.flush()
    set_seed(10)
    debug = False

    # train
    trainer.train()

    # test
    if not trainer.accelerator.is_local_main_process:
        pass
    else:
        trainer.load(trainer.train_num_steps // save_and_sample_every)
        trainer.set_results_folder('./results/test_timestep_' + str(sampling_timesteps))
        trainer.test(last=True)
