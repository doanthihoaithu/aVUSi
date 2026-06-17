import datetime
import os
from time import time

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from detectors.Detector import Detector
from utils import transform_to_dimension_contribution, color
from detectors.tran_ad.model import *


class TranADDetector(Detector):

    def __init__(self, customParameters, config):
        super().__init__(customParameters=customParameters, config=config)
        # self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def name(self) -> str:
        return self.customParameters.model

    def _init_model(self):
        (self.model,
         self.optimizer,
         self.scheduler,
         self.epoch, self.accuracy_list) = self.load_model()

    def train(self, data, labels=None)-> float:
        modelname = self.customParameters.model
        print('Model name', modelname)
        args = self.customParameters
        # model_class = getattr(detectors.model, modelname)
        if 'tran_ad' in modelname:
            self.model = TranAD(args['n_feats'], args['n_window'], device=self.device).to(device=self.device, dtype=torch.float32)
            data = self.convert_to_windows(data, customParameters=self.customParameters).to(device=self.device)
        elif 'omni_anomaly' in modelname:
            # data = self.convert_to_windows(data, customParameters=self.customParameters).to(device=self.device)
            # data = torch.tensor(data, dtype=torch.float32).to(device=self.device)
            data = self.convert_to_windows(data, customParameters=self.customParameters).to(device=self.device)
            self.model = OmniAnomaly(args['n_feats'], args['n_window'], device=self.device).to(device=self.device, dtype=torch.float32)
        # elif 'gdn' in modelname:
        #     data = self.convert_to_windows(data, customParameters=self.customParameters).to(device=self.device)
        #     self.model = GDN(args['n_feats'], device=self.device).to(device=self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        fname = f'{self.absolute_modelOutputDir}/checkpoints/{args.model}_{args.dataset}/model.ckpt'
        if os.path.exists(fname) and (not args.retrain):
            print(f"{color.GREEN}Loading pre-trained model: {self.model.name}{color.ENDC}")
            checkpoint = torch.load(fname, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.accuracy_list = checkpoint['accuracy_list']
            return checkpoint['training_time']
        else:
            print(f"{color.GREEN}Creating new model: {self.model.name}{color.ENDC}")
            self.accuracy_list = []
            start_process_time = time()
            epoch = 0
            num_epochs = self.customParameters['epochs']
            # trainD = torch.tensor(data.clone().detach()).to(device=self.device, dtype=torch.float32)
            # trainO = torch.tensor(data.clone().detach()).to(device=self.device, dtype=torch.float32)
            trainD = data.clone()
            trainO = data.clone()
            # trainD = data[:,:-1, :].clone()
            # trainO = data[:,-1, :].clone()
            for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
                lossT, lr = self.backprop(e, self.model, trainD, trainO, self.optimizer, self.scheduler)
                self.accuracy_list.append((lossT, lr))
            end_process_time = time()
            training_time = end_process_time - start_process_time
            print(color.BOLD + 'Training time: ' + "{:10.4f}".format(
                training_time) + ' s' + color.ENDC)
            self.save_model(self.model, self.optimizer, self.scheduler, e, self.accuracy_list, training_time)
            # plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
            # self.model.fit(data, self.absolute_modelOutputFile)
            # self.model.save(self.absolute_modelOutputFile)
            # end_process_time = datetime.datetime.now()
            total_time = end_process_time - start_process_time
            pd.DataFrame([total_time], columns=['train_time']).to_csv(
                os.path.join(self.absolute_modelOutputDir, 'docker-algorithm-train-time.csv'), index=False)
            return total_time

    def predict(self, test_dict):
        self.model = self.model if self.model is not None else self.load_model()[0]

        (test_filenames,
         test_data_list,
         test_labels_list,
         test_multivariate_labels_list,
         test_contamination_list) = test_dict['test_filenames'], test_dict['test_data_list'], test_dict[
            'test_labels_list'], test_dict['test_multivariate_labels_list'], test_dict['test_contamination_list']

        result_dict = {test_filename: {} for test_filename in test_filenames}
        torch.zero_grad = True
        self.model.eval()
        print(f'{color.HEADER}Testing {self.customParameters.model} on {self.customParameters.dataset}{color.ENDC}')
        for test_filename, data, labels, multivariate_labels, contamination in tqdm(zip(test_filenames,
                                                                                        test_data_list,
                                                                                        test_labels_list,
                                                                                        test_multivariate_labels_list,
                                                                                        test_contamination_list),
                                                                                    total=len(test_filenames),
                                                                                    desc="Executing test files"):
            multivariate_label_df = pd.DataFrame(multivariate_labels)
            if multivariate_label_df is not None:
                assert multivariate_label_df.shape[0] == data.shape[0]
                test_filepath = os.path.join(self.absolute_dataOutputDir, test_filename)
                os.makedirs(test_filepath, exist_ok=True)
                multivariate_label_df.to_csv(os.path.join(test_filepath, 'docker-algorithm-multivariate-labels.csv'))

            start_process_time = datetime.datetime.now()

            testD = data
            testO = data

            if self.customParameters.model in ['tran_ad', 'gdn', 'mtad_gat', 'omni_anomaly']:
                testD = self.convert_to_windows(testD, self.customParameters).to(device=self.device)
                testO = self.convert_to_windows(testO, self.customParameters).to(device=self.device)

            else:
                testD = torch.tensor(testD, dtype=torch.float32).to(device=self.device)
                testO = torch.tensor(testO, dtype=torch.float32).to(device=self.device)

            loss, preds = self.backprop(0, self.model, testD, testO, self.optimizer, self.scheduler, training=False)
            # preds_per_var = np.abs(preds - data[:,-1,:].numpy())
            # preds = np.sum(np.abs(preds - data[:,-1,:].numpy()), axis=1, keepdims=True)
            loss_per_var = loss.copy()
            scores = np.sum(loss, axis=1, keepdims=True)
            # preds = np.sum(np.abs(preds - data[:, -1, :].numpy()), axis=1, keepdims=True)

            end_process_time = datetime.datetime.now()
            total_time = (end_process_time - start_process_time).total_seconds()
            result_dict[test_filename]['execute_main_time'] = total_time
            result_dict[test_filename]['scores'] = scores
            # pd.DataFrame([total_time], columns=['execute_time']).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-execute-time.csv'), index=False)

            # np.savetxt(os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores.csv'), preds,
            #            delimiter=",")

            # scores_per_var = clf.decision_scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(config.anomalyScorePerVarOutput, index=False, header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(config.anomalyRankingOutput, index=False, header=None)

            scores_per_var = loss_per_var
            result_dict[test_filename]['scores_per_var'] = scores_per_var
            # pd.DataFrame(scores_per_var).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, f'docker-algorithm-scores-per-var.csv'),
            #     index=False,
            #     header=None)
            # scores_per_var_ranking = np.argsort(-scores_per_var, axis=1)
            # pd.DataFrame(scores_per_var_ranking).to_csv(
            #     os.path.join(absolute_dataOutput, test_filename, f'docker-algorithm-scores-per-var-ranking.csv'),
            #     index=False, header=None)
            dimension_contribution = transform_to_dimension_contribution(loss_per_var)
            result_dict[test_filename]['dimension_contribution'] = dimension_contribution
            # pd.DataFrame(dimension_contribution).to_csv(
            #     os.path.join(self.absolute_dataOutputDir, test_filename, 'docker-algorithm-dimension-contribution.csv'),
            #     index=False,
            #     header=None)

        return result_dict

    def save_model(self, model, optimizer, scheduler, epoch, accuracy_list, training_time):
        args = self.customParameters
        folder = f'{self.absolute_modelOutputDir}/checkpoints/{args.model}_{args.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy_list': accuracy_list,
            'training_time': training_time

        }, file_path),

    def load_model(self):
        modelname = self.customParameters.model
        args = self.customParameters
        # model_class = getattr(detectors.model, modelname)
        if 'tran_ad' in modelname:
            model = TranAD(args['n_feats'], args['n_window'], device=self.device).to(device=self.device, dtype=torch.float32)
        elif 'omni_anomaly' in modelname:
            model = OmniAnomaly(args['n_feats'], args['n_window'], device=self.device).to(device=self.device, dtype=torch.float32)
        # elif 'gdn' in modelname:
        #     model = GDN(args['n_feats'], device=self.device).to(device=self.device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        fname = f'{self.absolute_modelOutputDir}/checkpoints/{args.model}_{args.dataset}/model.ckpt'
        if os.path.exists(fname) and (not args.retrain):
            print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
            checkpoint = torch.load(fname, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            accuracy_list = checkpoint['accuracy_list']
        else:
            print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
            epoch = 0
            accuracy_list = []
        return model, optimizer, scheduler, epoch, accuracy_list

    def backprop(self, epoch, model, data, dataO, optimizer, scheduler, training=True):
        l = nn.MSELoss(reduction='mean' if training else 'none')
        feats = dataO.shape[-1]

        if 'omni_anomaly' in model.name:
            # l = nn.MSELoss(reduction='none')
            # data_x = torch.DoubleTensor(data)
            # dataset = TensorDataset(data_x, data_x)
            # bs = model.batch if training else len(data)
            # dataloader = DataLoader(dataset, batch_size=bs)

            # dataset = TensorDataset(data, data)
            # bs = model.batch if training else len(data)
            # dataloader = DataLoader(dataset, batch_size=bs)

            if training:
                mses, klds = [], []
                # for i, d in enumerate(data):
                #
                # i=-1
                data_x = data[:, :-1, :]
                data_y = data[:, -1, :]
                dataset = TensorDataset(data_x, data_y)
                # bs = model.batch if training else len(data)
                bs = model.batch
                dataloader = DataLoader(dataset, batch_size=bs)
                for i, (d,y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Omni Anomaly training {epoch}",
                                 leave=True,
                                 position=0):
                    # i+=1
                    d = d.to(device=self.device, dtype=torch.float32)
                    # y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                    y_pred, mu, logvar, hidden = model(d)
                    MSE = l(y_pred, y)
                    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                    loss = MSE + model.beta * KLD
                    mses.append(torch.mean(MSE).item())
                    klds.append(model.beta * torch.mean(KLD).item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
                scheduler.step()
                return loss.item(), optimizer.param_groups[0]['lr']
            else:
                data_x = data[:, :-1, :]
                data_y = data[:, -1, :]
                dataset = TensorDataset(data_x, data_y)
                # bs = model.batch if training else len(data)
                bs = model.batch
                dataloader = DataLoader(dataset, batch_size=bs)
                y_preds = []
                for i, (d, d_y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Omni Anomaly testing...",
                                 leave=True,
                                 position=0):
                    d = d.to(device=self.device, dtype=torch.float32)
                    y_pred, _, _, hidden = model(d)
                    y_preds.append(y_pred)
                y_pred = torch.concatenate(y_preds, dim=0)
                MSE = l(y_pred, data_y)
                return MSE.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
        elif 'mtad_gat' in model.name or 'gdn' in model.name:
            print('training model.name')
            l = nn.MSELoss(reduction='none')
            n = epoch + 1
            w_size = model.n_window
            l1s = []
            if training:
                for i, d in enumerate(data):
                    if 'mtad_gat' in model.name:
                        x, h = model(d, h if i else None)
                    else:
                        x = model(d)
                    loss = torch.mean(l(x, d))
                    l1s.append(torch.mean(loss).item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
                return np.mean(l1s), optimizer.param_groups[0]['lr']
            else:
                xs = []
                for d in data:
                    if 'mtad_gat' in model.name:
                        x, h = model(d, None)
                    else:
                        x = model(d)
                    xs.append(x)
                xs = torch.stack(xs)
                y_pred = xs[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                loss = l(xs, data)
                loss = loss[:, data.shape[1] - feats:data.shape[1]].view(-1, feats)
                return loss.detach().numpy()[-1, :], y_pred.detach().numpy()[-1, :]
        elif 'tran_ad' in model.name:
            # data = self.convert_to_windows(data, self.customParameters)
            # dataO = self.convert_to_windows(data, self.customParameters)
            l = nn.MSELoss(reduction='none')
            # data_x = torch.DoubleTensor(data).to(device=self.device, dtype=torch.float32)
            data_x = data[:,:-1, :]
            data_y = data[:,-1, :]
            dataset = TensorDataset(data_x, data_y)
            # bs = model.batch if training else len(data)
            bs = model.batch
            dataloader = DataLoader(dataset, batch_size=bs)
            n = epoch
            w_size = model.n_window
            l1s = []
            l2s = []
            if training:
                for d, d_y in dataloader:
                    local_bs = d.shape[0]
                    window = d.permute(1, 0, 2)
                    elem = d_y.view(1, local_bs, feats)
                    z = model(window, elem)
                    l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1],
                                                                                                               elem)
                    if isinstance(z, tuple): z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                scheduler.step()
                tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
                return np.mean(l1s), optimizer.param_groups[0]['lr']
            else:
                losses = []
                predict_errors = []
                for d, d_y in dataloader:
                    window = d.permute(1, 0, 2)
                    new_bs = d.shape[0]
                    # elem = window[-1, :, :].view(1, bs, feats)
                    elem = d_y.view(1, new_bs, feats)
                    z = model(window, elem)
                    if isinstance(z, tuple): z = z[1]
                    loss = l(z, elem)[0]
                    losses.append(loss.detach().cpu().numpy())
                    predict_errors.append(z.detach().cpu().numpy()[0])
                return np.concatenate(losses, axis=0), np.concatenate(predict_errors, axis=0)
                # return loss.detach().numpy(), z.detach().numpy()[0]
        else:
            y_pred = model(data)
            loss = l(y_pred, data)
            if training:
                tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                return loss.item(), optimizer.param_groups[0]['lr']
            else:
                return loss.detach().numpy(), y_pred.detach().numpy()

    def convert_to_windows(self, data, customParameters):
        data = torch.tensor(data, dtype=torch.float32)

        windows = []
        w_size = customParameters.n_window + 1
        for i, g in enumerate(data):
            if i >= w_size:
                w = data[i - w_size:i, :]
            else:
                # w = torch.cat([data[0,:].repeat(w_size - i, 1), data[0:i, :]])
                w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
            windows.append(w if customParameters.model in ['tran_ad','gdn','mtad_gat', 'omni_anomaly'] else w.view(-1))
        return torch.stack(windows, dim=0)