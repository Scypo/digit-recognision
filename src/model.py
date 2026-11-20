import torch
import torch.nn as nn
ACTVATIONS = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid
}
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = None
        self.model = nn.Sequential()
    def load_config(self, config):
        self.config = config
    def init_network(self):
        assert self.config is not None
        activation = ACTVATIONS[self.config["activation"]]

        self.model.append(nn.Linear(self.config["input_size"],self.config["hidden_size"]))
        self.model.append(activation())

        for _ in range(self.config["num_layers"] - 1):
            self.model.append(nn.Linear(self.config["hidden_size"],self.config["hidden_size"]))
            self.model.append(activation())
            
        self.model.append(nn.Linear(self.config["hidden_size"], self.config["output_size"]))
    def forward(self, x):
        return self.model(x)

def create_network(input_size, hidden_size, output_size, num_layers, activation):
    layers = []

    layers.append(nn.Linear(input_size,hidden_size))
    layers.append(activation())

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size,hidden_size))
        layers.append(activation())
        
    layers.append(nn.Linear(hidden_size, output_size))
        
    return nn.Sequential(*layers)
def create_model(config)->Model:
    model = Model()
    model.load_config(config)
    model.init_network()
    model.eval()
    return model
def save_model(model: Model, path: str = "models/saved_model.pth")->None:
    torch.save({
        "config": model.config,
        "model_state": model.state_dict()
        }, path)
def load_model(path: str = "models/saved_model.pth")->Model:
    model = Model()
    state = torch.load(path, weights_only = True, map_location="cpu",)
    model.load_config(state["config"])
    model.init_network()
    model.load_state_dict(state["model_state"])
    model.eval()
    return model