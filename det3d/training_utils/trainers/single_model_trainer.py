import torch
from tqdm import tqdm
from datetime import date, datetime
import os
from pathlib import Path

from det3d.training_utils.trackers import Tracker
from det3d.losses.loss_configurator import LossConfigurator

from det3d.validation.eval_utils import save_detection_results_to_txt
from det3d.validation.evaluators import KittiEvaluator

from tools.batch_transforms import example_convert_to_torch


from det3d.builders.optimization_builder import build_optimizer, build_scheduler

class SingleModelTrainer():

    def __init__(self, 
                 model, 
                 #optimizer,
                 #scheduler, 
                 cfg, 
                 train_dataloader, 
                 eval_dataloader=None):

        # Building based on configuration file
        training_cfg = cfg.training 

        # we need to track the current epoch
        # in case we store the model and we need to 
        # resume the training process
        self.current_epoch = -1 # training hasn't started yet
        self.num_iters = 0 # number of network activations

        # total number of epochs
        self.num_epochs = training_cfg.num_epochs
        # result visualization at the end of the training
        self.visualize_training_curves = training_cfg.visualize_training_curves
        # model device #NOTE: currently supports single device
        self.device = torch.device(training_cfg.device)

        # Checkpoint Configuration
        self.save_to_checkpoint_during_training = training_cfg.checkpoints.save_to_checkpoint_during_training
        self.save_after_k_epochs = training_cfg.checkpoints.save_after_k_epochs
        self.save_checkpoint_after_training = training_cfg.checkpoints.save_checkpoint_after_training
        self.save_path = Path(training_cfg.checkpoints.save_path)  # making it a Path
        self.use_relative_path = training_cfg.checkpoints.use_relative_path
        self.relative_path_name = training_cfg.checkpoints.relative_path_name
        self.replace_older_checkpoints = training_cfg.checkpoints.replace_older_checkpoints
        self.checkpoint_save_file_name = training_cfg.checkpoints.checkpoint_save_file_name
        self.save_tracker = training_cfg.checkpoints.save_tracker

        # Evaluation configurations
        self.eval_during_training = training_cfg.evaluation.eval_during_training
        self.eval_after_k_epochs = training_cfg.evaluation.eval_after_k_epochs
        self.eval_after_training = training_cfg.evaluation.eval_after_training

        # Setting Model/ Optimizers/ Schedulers/ Dataloaders
        self.model = model.to(self.device)
        #self.optimizer = optimizer
        #self.scheduler = scheduler
        self.train_loader = train_dataloader
        self.eval_loader = eval_dataloader
        # save path for evaluation
        # NOTE: we need to save the results to run the evaluation code
        self.detection_save_path = training_cfg.evaluation.save_path    

        # Creating optimizer and scheduler        
        self.optimizer = build_optimizer(self.model, cfg.optimization)
        
        self.scheduler, self.scheduler_step_in_epoch = build_scheduler(
            self.optimizer,
            total_iters_each_epoch = len(self.train_loader),
            total_epochs = self.num_epochs,      
            last_epoch = self.current_epoch,
            optim_cfg = cfg.optimization)


        #print(self.optimizer.state_dict)
        #print(self.scheduler)

        # Initialize Loss Configurator  
        self.loss_configurator = LossConfigurator(cfg.loss)

        # Initialize Tracker 
        self.tracker = Tracker(self.loss_configurator.keys)

        # After everything is initialized
        # Checking if the training is implemented from scratch
        # or if we are loading from a checkpoint
        if training_cfg.load_from_checkpoint:
            self.load_checkpoint(training_cfg.check_point_path)


    def train(self):
        start = self.current_epoch+1
        print("Resuming Training from epoch ", start)

        for epoch in range(start+1, self.num_epochs+1):
            
            print("Epoch : ", epoch)
            # incrementing current epoch by 1
            self.current_epoch += 1

            self.train_epoch()

            if self.save_to_checkpoint_during_training:
                if epoch % self.save_after_k_epochs == 0:
                    self.save_checkpoint()
            #self.tracker.print_losses()
            self.tracker.print_latest_losses()

            if self.eval_during_training:
                # NOTE: the weird epoch counting ensures that no checkpoint
                #       will be saved after the first epoch (unless specified)
                if epoch % self.eval_after_k_epochs == 0:
                    # evaluating the network
                    self.eval()
                    # printing evaluation results
                    #self.tracker.print_latest_metrics() #prints too much info
                    self.tracker.print_latest_3d_prec()
            
        if self.eval_after_training:
            self.eval(return_metrics=False) # dont store eval results on tracker
            #self.tracker.print_latest_metrics() #prints too much info
            self.tracker.print_latest_3d_prec()
            #print(self.tracker.metrics)
            #print(self.tracker.losses)

        if self.save_checkpoint_after_training:
            self.save_checkpoint()

        if self.visualize_training_curves:
            self.visualize_results()


    def train_epoch(self):
        # telling the tracker to start new epoch
        self.tracker.start_new_epoch()
        # setting the network to training mode
        self.model.train()
        # for every batch
        for batch in tqdm(self.train_loader):
            self.num_iters += 1
            self.training_step(batch)
            

        # calculate average epoch stats
        self.tracker.regularize_epoch_loss()

        # scheduler step
        if not self.scheduler_step_in_epoch:
            self.scheduler.step()    


    def training_step(self, batch):
        # reseting the optimizer
        self.optimizer.zero_grad()

        # sending the example to the gpu
        example = example_convert_to_torch(batch, device=self.device)
        # activating the network
        prediction = self.model(example)
        
        # computing the loss using the loss configurator
        loss, loss_tracker = self.loss_configurator.compute_losses(
                                prediction, example)

        # Caclulate Gradients
        loss.backward()
        if self.current_epoch < 1:
            print(loss)

        # updating weights
        self.optimizer.step()
        
        # scheduling step - if needed
        if self.scheduler_step_in_epoch:
            self.scheduler.step(self.num_iters)

        # updating the tracker
        self.tracker.update_losses(loss_tracker)


    def eval(self, dataset=None, path=None, return_metrics=True):

        if dataset is None:
            dataset = self.eval_loader.dataset 
        
        if path is None:
            path = self.detection_save_path

        # Epochs the model was trained for
        start = self.current_epoch+1
        print(f"Model has been train for {start} epochs " )
        
        # setting model to evaluation mode
        self.model.eval()
        
        # Initializing evaluator
        evaluator = KittiEvaluator(dataset._infos, 0)

        # Creating an empty list to store detection results
        # detections = []
        evaluator.track_detections()
        with torch.no_grad():
            
            for batch in tqdm(self.eval_loader):
                # sending the example to the GPU
                example = example_convert_to_torch(batch, device=self.device)
                # activating the network
                model_out = self.model(example)

                # NOTE: Permutation due to target assigner missmatch
                model_out['cls_preds'] = model_out['cls_preds'].permute(0, 3, 1, 2)
                pred_boxes_shape = model_out['box_preds'].shape
                model_out['box_preds'] = model_out['box_preds'].reshape(*list(pred_boxes_shape)[:-1], 2, 7)
                model_out['box_preds'] = model_out['box_preds'].permute(0, 3, 1, 2, 4)

                # TODO: ADD METADATA TO OUTPUT!
                # detections += model_out 
                # use predict function to predict the final bounding boxes
                preds_dict = self.model.predict(example, model_out)

                evaluator.add_detections(preds_dict)
                #raise ValueError
                #save_detection_results_to_txt("/home/ioannis/Desktop/Programming/Python-Thesis/detection3d/data/kitti/detection_results", 
                #                                pred_dicts = preds_dict)


        # Save Results using the evaluator
        evaluator.save_detections_for_evaluation(path)
        

        det_path = "/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/detection_results/data"
        gt_path = "/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/training/label_2"
        gt_split_file = "/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/kittiSplits/val.txt"
        if return_metrics:
            new_metrics = evaluator.evaluate(det_path, gt_path, gt_split_file, return_metrics=return_metrics)
            self.tracker.add_metrics(new_metrics)
        else:
            evaluator.evaluate(det_path, gt_path, gt_split_file, return_metrics=return_metrics)

    
    def save_checkpoint(self):

        if self.use_relative_path:
            if self.relative_path_name == "USE_DATE":
                subfolder_name = date.today().strftime("%d_%m_%y")
            else:
                subfolder_name = self.relative_path_name

            # creating full path using relative path
            full_path = self.save_path / subfolder_name

            # if the directory doesn't exist
            if not os.path.isdir(full_path):
                # create the directory
                os.mkdir(full_path)
        else:
            full_path = self.save_path

        # constructing the checkpoint name
        if self.replace_older_checkpoints:
            # to replace older checkpoints, insted of using timespamp
            # we will use the date as the checkpoint name
            if self.checkpoint_save_file_name == "USE_TIMESTAMP":
                checkpoint_name = date.today().strftime("%d_%m_%y")
            else:  # custom name is provided
                checkpoint_name = self.checkpoint_save_file_name
        # do not replace older checkpoints
        else:
            timestamp = datetime.now().strftime("%H_%M_%S")
            if self.checkpoint_save_file_name == "USE_TIMESTAMP":
                checkpoint_name = timestamp
            else:
                # if custom name is provided, we will add the timestamp
                # to separete the different checkpoints
                checkpoint_name = self.checkpoint_save_file_name + "_" + timestamp

        # adding suffix to the filename
        checkpoint_name = checkpoint_name + ".pt"

        # calculating the complete path
        PATH = full_path / checkpoint_name

        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_tracker': self.tracker.losses,
            'metrics_tracker': self.tracker.metrics
        }, PATH)


    def load_checkpoint(self, PATH):
        # loading checkpoint
        checkpoint = torch.load(PATH)
        # updating the state of training components
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.tracker.losses = checkpoint['loss_tracker']
        self.tracker.metrics = checkpoint['metrics_tracker']


    def visualize_results(self):
        self.tracker.visualize_losses()

        if self.eval_during_training:
            self.tracker.visualize_metrics(self.num_epochs)
