import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger("zebel-scanner")


def dice_loss(y_true, y_pred, epsilon=1e-6):
    y_true_f = y_true.squeeze().reshape(-1).float()
    y_pred_f = y_pred.squeeze().reshape(-1).float()
    intersection = torch.dot(y_true_f, y_pred_f)
    return 1 - (2.0 * intersection + epsilon) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) + epsilon
    )


class EvaluatorFactory:
    """
    This class is used to get different versions of evaluators
    """

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="rmse", cuda=True):
        if testType == "rmse":
            return DocumentMseEvaluator(cuda)


class DocumentMseEvaluator:
    """
    Evaluator class for softmax classification
    """

    def __init__(self, cuda, tb_writer=None):
        self.cuda = cuda
        self.tb_writer = tb_writer

    def evaluate(self, model, iterator):
        model.eval()
        lossAvg = None
        with torch.no_grad():
            for img, target in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))
                # print (response[0])
                # print (target[0])
                loss = F.mse_loss(response, Variable(target.float()))
                loss = torch.sqrt(loss)
                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))

        lossAvg /= len(iterator)
        logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))


class GenericTrainer:
    """
    Base class for trainer; to implement a new training routine, inherit from this.
    """

    def __init__(self):
        pass


class Trainer:
    def __init__(
        self,
        train_iterator: DataLoader,
        valid_iterator: DataLoader,
        model,
        cuda,
        optimizer,
        tb_writer,
    ):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.model = model
        self.optimizer = optimizer
        self.criterion = dice_loss
        self.tb_writer = tb_writer

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group["lr"]
                    param_group["lr"] = self.current_lr * gammas[temp]
                    logger.debug(
                        "Changing learning rate from %0.9f to %0.9f",
                        self.current_lr,
                        self.current_lr * gammas[temp],
                    )
                    self.current_lr *= gammas[temp]

    def train(self, epoch):
        self.model.train(True)
        lossAvg = None
        counter = 0
        for img, target in tqdm(self.train_iterator, position=0):
            if self.cuda:
                img, target = img.cuda(), target.cuda()
            self.optimizer.zero_grad()
            # remove variable
            # print("img.shape", img.shape, img.dtype)

            y_pred = self.model(img)
            # print (response[0])
            # print (target[0])
            loss = self.criterion(target, y_pred)
            if lossAvg is None:
                lossAvg = loss
            else:
                lossAvg += loss
            # logger.debug("Cur loss %s", str(loss))
            loss.backward()
            self.optimizer.step()
            if counter % 100 == 0:
                loss_valid_avg = self.evaluate()
                self.tb_writer.add_scalars(
                    "train/loss",
                    {"Training": lossAvg / counter, "Validation": loss_valid_avg},
                    epoch * len(self.train_iterator) + counter,
                )
            counter += 1

        lossAvg /= len(self.train_iterator)
        logger.info("Avg Loss %s", str((lossAvg).cpu().data.numpy()))

    def evaluate(self):
        self.model.eval()
        lossAvg = None
        with torch.no_grad():
            for img, target in tqdm(self.valid_iterator, position=1):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                y_pred = self.model(img)
                loss = self.criterion(y_pred, target)
                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                logger.debug("Cur loss %s", str(loss))

        lossAvg /= len(self.valid_iterator)
        logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))
        return lossAvg
