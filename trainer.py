import os
import torch
import time
from tqdm import tqdm
import numpy as np

from utils.utils import get_cifar10
from utils.logger import Logger
from utils.callbacks import EarlyStopping
from torch.utils.tensorboard import SummaryWriter



class Trainer:
    def __init__(
        self,
        model, 
        dataset_params,
        train_params,
        callback_params,
        optimizer_params,
        transform_train=None,
        transform_test=None
    ):
        self.output_dir = train_params["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.max_epochs = train_params["max_epoch"]
        self.device = train_params["device"]

        # init logger
        self.logger = Logger(self.output_dir)

        # init model
        self.model = model
        # print layer summary
        prev_layer_name = ""
        total_params = 0
        for name, param in self.model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name != prev_layer_name:
                prev_layer_name = layer_name
                self.logger.log_block("{:<70} {:<30} {:<30} {:<30}".format('Name','Weight Shape','Total Parameters', 'Trainable'))
            self.logger.log_message("{:<70} {:<30} {:<30} {:<30}".format(name, str(param.data.shape), param.data.numel(), param.requires_grad))
            total_params += np.prod(param.data.shape)
        self.logger.log_block(f"Total Number of Paramters: {total_params}")
        self.logger.log_line()
        self.writer = SummaryWriter()

        # init dataloaders
        self.train_dataloader, self.dev_dataloader, self.test_dataloader = get_cifar10(
            train_bs = dataset_params["train_bs"],
            test_bs = dataset_params["test_bs"],
            transform_train = transform_train,
            transform_test = transform_test
        )
        self.total_train_batch = len(self.train_dataloader)
        self.total_test_batch = len(self.test_dataloader)
        self.total_dev_batch = len(self.dev_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10 # use to log step loss
        self.logger.log_block(f"Training Dataset Size: {len(self.train_dataloader.dataset)}")
        self.logger.log_message(f"Training Dataset Total Batch#: {self.total_train_batch}")
        self.logger.log_block(f"Dev Dataset Size: {len(self.dev_dataloader.dataset)}")
        self.logger.log_message(f"Dev Dataset Total Batch#: {self.total_dev_batch}")
        self.logger.log_message(f"Test Dataset Size: {len(self.test_dataloader.dataset)}")
        self.logger.log_message(f"Test Dataset Total Batch#: {self.total_test_batch}")

        # init callback [early stopping]
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callback_params)

        # init optimizer and loss
        param_dict = [{
            "params": self.model.parameters(), 
            "lr": optimizer_params["lr"], 
        }]
        self.optimizer = getattr(torch.optim, optimizer_params["type"])(param_dict, **optimizer_params["kwargs"])
        self.criterion = torch.nn.CrossEntropyLoss()
    

        # put model to device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # log all configs
        self._log_configs(train_params)
        
    def _log_configs(
        self,
        train_params
    ):
        # log trainer kwargs
        self.logger.log_line()
        self.logger.log_message("Trainer Kwargs:")
        self.logger.log_new_line()
        for k, v in train_params.items():
            self.logger.log_message("{:<30} {}".format(k, v))

        # log optimizer kwargs
        self.logger.log_line()
        self.logger.log_message(f"Optimizer: {self.optimizer.__class__.__name__}")

        # log Callbacks kwargs
        self.logger.log_line()
        self.logger.log_message(f"Callbacks: {self.callbacks.__class__.__name__}")
        self.logger.log_new_line()
        self.logger.log_message("{:<30} {}".format('save_final_model', self.callbacks.save_final_model))
        self.logger.log_message("{:<30} {}".format('patience', self.callbacks.patience))
        self.logger.log_message("{:<30} {}".format('threshold', self.callbacks.threshold))
        self.logger.log_message("{:<30} {}".format('mode', self.callbacks.mode))

    def train(self):
        for epoch in range(self.max_epochs):
            # train one epoch
            self.cur_epoch = epoch
            self.logger.log_line()
            self.train_one_epoch()
            
            # eval one epoch
            self.logger.log_line()
            self.eval_one_epoch()

        self.logger.log_block(f"Max epoch reached. Best f1-score: {self.callbacks.best_score:.4f}")
        self.eval_best_model_on_testdataset()
        self.writer.close()
        exit(1)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        ten_percent_batch_loss = 0 # use to accumlate training loss every 10% of an epoch
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            loss, preds = self.train_one_step(inputs, targets)
            epoch_loss += loss
            ten_percent_batch_loss += loss
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            if (batch_idx+1)%self.ten_percent_train_batch == 0:
                ten_percent_avg_loss = ten_percent_batch_loss/self.ten_percent_train_batch
                message = f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - 10% Avg loss {ten_percent_avg_loss:.5f}"
                self.logger.log_message(message=message)
                ten_percent_batch_loss = 0
        
        end_time = time.time()
        epoch_acc = correct/total*100
        avg_loss = epoch_loss/self.total_train_batch
        self.writer.add_scalars("Loss", {"train":avg_loss},self.cur_epoch)
        self.writer.add_scalars("Accuracy", {"train":epoch_acc}, self.cur_epoch)
        epoch_time = (end_time-start_time)/60
        self.logger.log_message(f"Epoch #{self.cur_epoch}: Epoch Average Loss {avg_loss:.5f} - Epoch Acc: {epoch_acc:.5f} - Epoch Training Time: {epoch_time:.2} min(s)")

    def train_one_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        _, preds = outputs.max(1)

        return loss.item(), preds

    def eval_one_epoch(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.dev_dataloader, desc="Evaluate on Devtset"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.eval_one_step(inputs, targets)
                test_loss += loss
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
        
        end_time = time.time()
        test_acc = correct/total*100
        avg_loss = test_loss/self.total_test_batch
        epoch_time = (end_time-start_time)/60
        self.writer.add_scalars("Loss", {"val":avg_loss}, self.cur_epoch)
        self.writer.add_scalars("Accuracy", {"val":test_acc}, self.cur_epoch)
        self.logger.log_message(f"Eval Devset: Epoch #{self.cur_epoch}: Average Loss {avg_loss:.5f} - Epoch Acc: {test_acc:.5f} - Epoch Testing Time: {epoch_time:.2} min(s)")

        # saving best model and early stopping
        if not self.callbacks(self.model, test_acc):
            self.eval_best_model_on_testdataset()
            exit(1)

    def eval_one_step(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, preds = outputs.max(1)

        return loss.item(), preds

    def eval_best_model_on_testdataset(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best-model.pt")))
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_dataloader, desc="Evaluate on Testset"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.eval_one_step(inputs, targets)
                test_loss += loss
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
        
        end_time = time.time()
        test_acc = correct/total*100
        avg_loss = test_loss/self.total_test_batch
        epoch_time = (end_time-start_time)/60
        self.logger.log_message(f"Test Devset: Epoch #{self.cur_epoch}: Average Loss {avg_loss:.5f} - Epoch Acc: {test_acc:.5f} - Epoch Testing Time: {epoch_time:.2} min(s)")