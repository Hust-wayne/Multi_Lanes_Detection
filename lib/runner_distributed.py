import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from lib.distributed_utils import *
import sys


class Runner_Distributed:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False,
                 syncBN=False, distributed=False, device_id=0, main_pod=0):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.SyncBN = syncBN
        self.device_id = device_id
        self.main_pod = main_pod
        self.distributed=distributed
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler, self.device)
            starting_epoch = last_epoch + 1

        if self.SyncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)

        # 转为DDP模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id])

        max_epochs = self.cfg['epochs']
        train_loader, train_sampler = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            train_sampler.set_epoch(epoch)
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = train_loader

            if is_main_process():
                pbar = tqdm(train_loader)

            #mean_loss = torch.zeros(1).to(self.device)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())


                """
                DDP出现如AttributeError: 'DDP' object has no attribute 'xxx'的错误。

                原因：在使用net = torch.nn.DDP(net)之后，原来的net会被封装为新的net的module属性里。

                解决方案：所有在net = DDP(net)后调用了不是初始化与forward的属性，需要将net替换为net.module.XXX  XXX为模型中自定义得函数
                """
                ######multy lane
                #loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)
                loss, loss_dict_i = model.module.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                loss = reduce_value(loss, average=True)
                #mean_loss = (mean_loss * i + loss.detach()) / (i + 1)  # update mean losses
                optimizer.step()
                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                if self.main_pod == 0:
                    postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                    postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                    self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                    postfix_dict['loss'] = loss.item()
                    pbar.set_postfix(ordered_dict=postfix_dict)

            # 等待所有进程计算完毕
            if self.device != torch.device("cpu"):
                torch.cuda.synchronize(self.device)

            if self.main_pod == 0:
                self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

        self.exp.train_end_callback()
        
        
    def eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                
                for bb in range(len(images)):
                    img = (images[bb].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[bb])
                    cv2.imwrite('/home/LaneATT-main/XXX/'+'pred_'+ str(idx)+ "_"+str(bb)+".jpg", img)
                
                
                #if self.view:
                 #   img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                  #  img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                   # if self.view == 'mistakes' and fp == 0 and fn == 0:
                    #    continue
                    #cv2.imshow('pred', img)
                    #cv2.waitKey(0)

        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        ######DDP
        # 给每个rank对应的进程分配训练的样本索引
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # 将样本索引每batch_size个元素组成一个list
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, self.cfg['batch_size'], drop_last=True)

        nw = min([os.cpu_count(), self.cfg['batch_size'] if self.cfg['batch_size'] > 1 else 0, 8])  # number of workers

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_sampler=train_batch_sampler,
                                                   num_workers=nw,
                                                   pin_memory=True,
                                                   worker_init_fn=self._worker_init_fn_)

        return train_loader, train_sampler

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
