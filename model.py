import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from op.body import Body


class Dnn(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)

        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.drop5 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.drop1 = nn.Dropout(0.1)

        #self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        ########################################################################################
        self.out = nn.Linear(hidden_size4, output_size)
        ############################################################################################

    def forward(self, x):
        output = self.fc1(x)
        output = self.drop3(output)
        output = F.relu(output)

        output = self.fc2(output)
        output = self.drop3(output)
        output = F.relu(output)

        output = self.fc3(output)
        output = F.relu(output)
        output = self.drop1(output)

        output = self.fc4(output)
        output = F.relu(output)

        output = self.out(output)
        #output = self.out(x)
        #output = F.softmax(output)
        return output


class Model:

    def __init__(self, body_pose_model_path: str, sit_or_stand_model_path: str):
        self.body_estimation = Body(body_pose_model_path)
        self.sit_or_stand = Dnn(36, 20, 10, 8, 5, 2)
        self.sit_or_stand.load_state_dict(torch.load(sit_or_stand_model_path))
        self.sit_or_stand.eval()

    def classify_sit_or_stand(self, data: torch.FloatTensor):
        return self.sit_or_stand(data)

    def estimate_body(self, img: np.ndarray):
        return self.body_estimation(img)
