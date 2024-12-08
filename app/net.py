import os.path

import torch
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from app.core.architecture import SiameseNetwork
from app.helpers import distance_function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NetworkDriver:
    def __init__(
        self, pattern_size: int, embedding_dim: int, epochs: int = 10, margin: float = 1.0, learning_rate: float = 0.001
    ):
        self.embedding_dim = embedding_dim
        self.pattern_size = pattern_size
        # self.epochs = epochs
        # self.margin = margin
        # self.learning_rate = learning_rate
        self.anchor_pattern, self.anchor_no_pattern = self.load_anchors()
        self.scaler = self.load_scaler()
        self.net = self.load_model()

    @staticmethod
    def load_anchors() -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pattern, anchor_no_pattern = None, None
        if os.path.exists('app/model/anchor_pattern.pt'):
            anchor_pattern = torch.load('app/model/anchor_pattern.pt')
        if os.path.exists('app/model/anchor_no_pattern.pt'):
            anchor_no_pattern = torch.load('app/model/anchor_no_pattern.pt')
        return anchor_pattern, anchor_no_pattern

    def load_model(self) -> SiameseNetwork:
        model = SiameseNetwork(embedding_dim=self.embedding_dim).to(device)
        if os.path.exists('app/model/model.pt'):
            model.load_state_dict(torch.load('app/model/model.pt'))
            model.eval()
        return model

    @staticmethod
    def load_scaler() -> StandardScaler:
        if os.path.exists('app/model/scaler.save'):
            return joblib.load('app/model/scaler.save')

        return StandardScaler()

    # def train(self, train_dataloader, anchors: Tuple[object, object], val: Tuple[object, object]) -> dict:
    #     """
    #     Method for gradient calculating
    #
    #     Returns:
    #         None
    #     """
    #     # torch.use_deterministic_algorithms(True)
    #     # torch.set_num_threads(1)
    #
    #     import os, random
    #     os.environ['PYTHONHASHSEED'] = str(1300)
    #     random.seed(1300)
    #     np.random.seed(1300)
    #     torch.cuda.manual_seed_all(1300)
    #
    #     # print(torch.cuda.is_available())
    #     # print(self.learning_rate, list(self.net.parameters()))
    #     # torch.set_num_threads(1)
    #     # print(self.learning_rate, self.margin, torch.cuda.is_available())
    #     optimizer = optim.Adam(self.net.parameters(), self.learning_rate)
    #     triplet_loss = TripletMarginWithDistanceLoss(distance_function=distance_function, margin=self.margin)
    #
    #     # return None
    #
    #     loss_history = []
    #     accuracy_history = []
    #     precision_history = []
    #     f_score_history = []
    #
    #     # init classes for metrics
    #     precision = Precision(task="binary", num_classes=2).to(device)
    #     accuracy = Accuracy(task="binary", num_classes=2).to(device)
    #     f_score = F1Score(task="binary", num_classes=2).to(device)
    #
    #     for epoch in tqdm(range(self.epochs)):
    #         for i, (anchor, positive, negative) in enumerate(train_dataloader, 0):
    #             # put the model in a training state
    #             # (https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
    #             self.net.train()
    #
    #             # print('anchor', anchor, positive, negative)
    #             # break
    #             # get embdeing for triplet
    #             output1, output2, output3 = self.net(anchor.to(device), positive.to(device), negative.to(device))
    #             # calculate loss
    #             output = triplet_loss(output1, output2, output3)
    #             output.backward()
    #
    #             # do one optimisation step and clear gradients
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #             # will calculate the metrics for every 10th batch
    #             # to speed up the process
    #             if i % 10 == 0:
    #                 loss_history.append(output.item())
    #
    #                 # put the model in a val state
    #                 # (https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch)
    #                 self.net.eval()
    #
    #                 # get the model's inferences on a dataset
    #                 # it did not see in training and count the metrics
    #                 outputs = get_validation_outputs(self.net, anchors[0], anchors[1], val[0])
    #                 accuracy_value = accuracy(outputs, val[1])
    #                 accuracy_history.append(accuracy_value.item())
    #
    #                 precision_value = precision(outputs, val[1])
    #                 precision_history.append(precision_value.item())
    #
    #                 f_score_value = f_score(outputs, val[1])
    #                 f_score_history.append(f_score_value.item())
    #
    #     self.anchor_pattern = self.net.forward_once(anchors[0])
    #     self.anchor_no_pattern = self.net.forward_once(anchors[1])
    #
    #     metrics = {
    #         'loss': loss_history,
    #         'accuracy': accuracy_history,
    #         'precision': precision_history,
    #         'f-score': f_score_history,
    #     }
    #     return metrics

    def predict(self, input_data: np.array) -> int:
        """
        Goes through the val dataset and generates a list of preds
        to compare with the targets

        Args:
          input_data: Torch val dataset

        Returns:
          Tensor of preds ([0, 1 ... 1, 1])
        """
        if self.anchor_pattern is None or self.anchor_no_pattern is None:
            raise ValueError('Train model before predict!')

        # get embedding for input data
        output = self.net.forward_once(input_data)

        # calculate distances between inputs and anchors
        pattern_dist = distance_function(self.anchor_pattern, output).item()
        no_pattern_dist = distance_function(self.anchor_no_pattern, output).item()

        # an object is a pattern if its embedding is close
        # to the embedding of the pattern anchor and farther
        # away from the non-pattern anchor
        if pattern_dist > no_pattern_dist:
            return 0
        else:
            return 1


net_driver = NetworkDriver(pattern_size=30, embedding_dim=30 * 10)
