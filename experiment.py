import os

import numpy as np
import time
from argparse import Namespace

import torch

from summarytracker import SummaryTracker
from monolab.data_loader.data_loader import prepare_dataloader
from monolab.loss import MonodepthLoss
from test import run_test
from utils import get_model, setup_logging, notify_mail, to_device, time_delta_now
import logging


logger = logging.getLogger(__name__)


class Experiment:
    """ A class for training and testing a model that contains the actual network as self.model
    The arguments are defined in the arg_parse()-function above.
    Important:
        - args.data_dir is the root directory
        - args.filenames_file should be different depending on whether you want to train or test the model
        - args.val_filenames_file is only used during training
    """

    def __init__(self, args: Namespace, base_dir: str):
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        self.args = args
        self.base_dir = base_dir

        self.loss_names = dict(
            full="monodepth-loss",
            images="image-loss",
            disp_gp="disparity-gradient-loss",
            lr_consistency="lr-consistency-loss",
        )

        # Setup summary tracker
        self.summary = SummaryTracker(
            metric_names=list(self.loss_names.values()), args=args, base_dir=base_dir
        )

        # Get the model
        self.model = self._get_model(args)

        # Setup loss, optimizer and validation set
        self.loss_function = MonodepthLoss(
            device=self.device,
            SSIM_w=args.weight_ssim,
            disp_gradient_w=args.weight_disp_gradient,
            lr_w=args.weight_lr_consistency,
        ).to(self.device)
        logger.debug(f"Using loss function: {self.loss_function}")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate
        )
        logger.debug(f"Using optimizer: {self.optimizer}")

        self.dataset_val = args.dataset_name_val

        # the validation loader is a train loader but without data augmentation!
        self.val_n_img, self.val_loader = prepare_dataloader(
            root_dir=args.data_dir,
            filenames_file=args.val_filenames_file,
            mode="val",
            augment_parameters=None,
            do_augmentation=False,
            shuffle=False,
            shuffle_before=True,
            batch_size=args.batch_size,
            size=(args.input_height, args.input_width),
            num_workers=args.num_workers,
            dataset=self.dataset_val,
            pin_memory=args.pin_memory,
        )
        logging.info(f"Using a validation set with {self.val_n_img} images")

        # Load data
        self.output_dir = args.output_dir
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.dataset_train = args.dataset_name_train

        self.n_img, self.loader = prepare_dataloader(
            root_dir=args.data_dir,
            filenames_file=args.filenames_file,
            mode="train",
            augment_parameters=args.augment_parameters,
            do_augmentation=args.do_augmentation,
            shuffle=not args.overfit,
            shuffle_before=False,
            batch_size=args.batch_size,
            size=(args.input_height, args.input_width),
            num_workers=args.num_workers,
            dataset=self.dataset_train,
            pin_memory=args.pin_memory,
        )
        logging.info(
            f"Using a training data set from {self.dataset_train} with {self.n_img} images"
        )

        if "cuda" in self.device:
            torch.cuda.synchronize()

    def _get_model(self, args) -> torch.nn.Module:
        """
        Get the model. Automatically moves model to specified device(s).
        Args:
            args: Experiment args
        Returns:
            Model
        """
        # Determine device
        if args.cuda_device_ids[0] == -2:
            self.device = "cpu"
            logger.info("Running experiment on the CPU ...")
        else:
            self.device = f"cuda:{args.cuda_device_ids[0]}"

        # Get model
        self.model = get_model(
            model=args.model,
            n_input_channels=args.input_channels,
            pretrained=args.imagenet_pretrained,
            args=self.args,
        )
        logger.info(
            "Training a {} model with {} parameters".format(
                args.model, sum(p.numel() for p in self.model.parameters())
            )
        )

        # Load pretrained model
        if args.checkpoint:
            logging.info(f"Loading pretrained model from {args.checkpoint}")
            self.load(args.checkpoint)

        self.multi_gpu = len(args.cuda_device_ids) > 1 or args.cuda_device_ids[0] == -1

        # Check if multiple cuda devices are selected
        if self.multi_gpu:
            num_cuda_devices = torch.cuda.device_count()

            if args.cuda_device_ids[0] == -1:
                # Select all devices
                cuda_device_ids = list(range(num_cuda_devices))
            else:
                cuda_device_ids = args.cuda_device_ids

            # Check if multiple cuda devices are available
            if num_cuda_devices > 1:
                logger.info(
                    f"Running experiment on the following GPUs: {cuda_device_ids}"
                )

                # Transform model into data parallel model on all selected cuda deviecs
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=cuda_device_ids
                )
            else:
                logger.warning(
                    f"Attempted to run the experiment on multiple GPUs while only {num_cuda_devices} GPU was available"
                )

        logger.debug(f"Sending model to device: {self.device}")
        return self.model.to(self.device)

    def train(self) -> None:
        """ Train the model for self.args.epochs epochs
        Returns:
            None
        """
        train_start_time = time.time()

        # Store the best validation loss
        best_val_loss = float("Inf")

        # Start training
        logger.info(
            f"Starting training for {self.args.epochs} epochs on {self.n_img} images"
        )
        for epoch in range(1, self.args.epochs + 1):
            # Adjust learning rate if flag is set
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch, self.args.learning_rate)

            epoch_time = time.time()

            # Init training loss
            running_loss = 0.0
            running_image_loss = 0.0
            running_disp_gradient_loss = 0.0
            running_lr_loss = 0.0

            # Init validation loss
            running_val_loss = 0.0
            running_val_image_loss = 0.0
            running_val_disp_gradient_loss = 0.0
            running_val_lr_loss = 0.0

            self.model.train()

            #################
            # Training loop #
            #################
            for iteration, data in enumerate(self.loader):
                # Load data
                data = to_device(data, self.device)
                left = data["left_image"]
                right = data["right_image"]

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss, image_loss, disp_gradient_loss, lr_loss = self.loss_function(
                    disps, [left, right]
                )
                loss.backward()
                self.optimizer.step()

                # Collect training loss
                running_loss += loss.item()
                running_image_loss += image_loss.item()
                running_disp_gradient_loss += disp_gradient_loss.item()
                running_lr_loss += lr_loss.item()

                # Stop after 10 batches if overfitting is enabled
                if self.args.overfit and iteration >= 5:
                    break

            # Training finished #
            logger.info(
                f"Epoch [{epoch}/{self.args.epochs}] time: {time_delta_now(epoch_time)} s"
            )

            ###################
            # Validation loop #
            ###################
            self.model.eval()
            val_time = time.time()
            with torch.no_grad():
                for iteration, data in enumerate(self.val_loader):
                    data = to_device(data, self.device)
                    left = data["left_image"]
                    right = data["right_image"]
                    disps = self.model(left)
                    loss, image_loss, disp_gradient_loss, lr_loss = self.loss_function(
                        disps, [left, right]
                    )

                    # Collect validation loss
                    running_val_loss += loss.item()
                    running_val_image_loss += image_loss.item()
                    running_val_disp_gradient_loss += disp_gradient_loss.item()
                    running_val_lr_loss += lr_loss.item()

                    # Stop after 10 batches if overfitting is enabled
                    if self.args.overfit and iteration >= 5:
                        break
            logger.info(f"Validation took {time_delta_now(val_time)}s")

            #################
            # Track results #
            #################
            running_val_loss /= len(self.val_loader)
            running_val_image_loss /= len(self.val_loader)
            running_val_disp_gradient_loss /= len(self.val_loader)
            running_val_lr_loss /= len(self.val_loader)

            # Generate 10 random disparity map predictions
            self.gen_val_disp_maps(epoch)

            # Estimate loss per image
            running_loss /= len(self.loader)
            running_image_loss /= len(self.loader)
            running_disp_gradient_loss /= len(self.loader)
            running_lr_loss /= len(self.loader)

            # Update best loss
            if running_val_loss < best_val_loss:
                best_val_loss = running_val_loss

            self.summary.add_epoch_metric(
                epoch=epoch,
                train_metric=running_loss,
                val_metric=running_val_loss,
                metric_name=self.loss_names["full"],
            )
            self.summary.add_epoch_metric(
                epoch=epoch,
                train_metric=running_image_loss,
                val_metric=running_val_image_loss,
                metric_name=self.loss_names["images"],
            )
            self.summary.add_epoch_metric(
                epoch=epoch,
                train_metric=running_disp_gradient_loss,
                val_metric=running_val_disp_gradient_loss,
                metric_name=self.loss_names["disp_gp"],
            )
            self.summary.add_epoch_metric(
                epoch=epoch,
                train_metric=running_lr_loss,
                val_metric=running_val_lr_loss,
                metric_name=self.loss_names["lr_consistency"],
            )

            self.summary.add_checkpoint(
                model=self.model, val_loss=running_val_loss, multi_gpu=self.multi_gpu
            )

        logging.info(f"Finished Training. Best loss: {best_val_loss}")
        self.summary.save()

        # Store best validation loss
        self.best_val_loss = best_val_loss
        self.training_time = time_delta_now(train_start_time)

    def gen_val_disp_maps(self, epoch: int):
        """
        Generate n validation disparity maps
        Args:
            epoch (int): Current epoch
        """
        gen_count = 0
        n_gen_images = 25

        with torch.no_grad():
            for (i, data) in enumerate(self.val_loader):
                # Stop after n_gen_images
                if gen_count >= n_gen_images:
                    break

                # Get the inputs
                data = to_device(data, self.device)
                left = data["left_image"]
                # Do a forward pass
                disps = self.model(left)

                while gen_count < n_gen_images and gen_count < self.args.batch_size:
                    batch_idx = gen_count % self.args.batch_size
                    largest_disp_map = disps[0]
                    image_in_batch = largest_disp_map[batch_idx]
                    left_disp = image_in_batch[0]
                    d = left_disp.cpu().numpy()
                    self.summary.add_disparity_map(
                        epoch=epoch,
                        disp=torch.Tensor(d),
                        idx=gen_count,
                        input_img=left[batch_idx],
                    )
                    gen_count += 1

    def save(self, path: str) -> None:
        """ Save a .pth state dict from self.model
        Args:
            path: path to .pth state dict file
        Returns:
            None
        """
        if self.multi_gpu:
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """ Load a .pth state dict into self.model
        Args:
            path: path to .pth state dict file
        Returns:
            None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def test(self):
        # Run test
        self.test_result, self.test_result_pp = run_test(
            model=self.model,
            args=self.args,
            device=self.device,
            result_dir=os.path.join(self.base_dir, "test"),
        )


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, learning_rate: float
):
    """ Sets the learning rate to the initial LR\
        decayed by factor 0.5 every 10 epochs after 30 epochs
    Args:
        optimizer: torch.optim type optimizer
        epoch: current epoch
        learning_rate: current learning rate
    """

    if epoch >= 30:
        lr = learning_rate * (0.5 ** (epoch // 10))
    else:
        lr = learning_rate

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
