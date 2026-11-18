import torch
import torch.nn as nn


def _cbr(cin, cout, k, s=1, act="leaky", pad=0):
    a = {"leaky": nn.LeakyReLU(0.01), "tanh": nn.Tanh()}[act]
    return [nn.Conv2d(cin, cout, k, stride=s, padding=pad), a, nn.BatchNorm2d(cout)]


class DeepLOBFeatures(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(*_cbr(1, 32, (1, 2), (1, 2)), *_cbr(32, 32, (4, 1)), *_cbr(32, 32, (4, 1)))
        self.conv2 = nn.Sequential(*_cbr(32, 32, (1, 2), (1, 2), "tanh"), *_cbr(32, 32, (4, 1), act="tanh"),
                                   *_cbr(32, 32, (4, 1), act="tanh"))
        self.conv3 = nn.Sequential(*_cbr(32, 32, (1, 10)), *_cbr(32, 32, (4, 1)), *_cbr(32, 32, (4, 1)))
        self.inp1 = nn.Sequential(*_cbr(32, 64, (1, 1)), *_cbr(64, 64, (3, 1), pad=(1, 0)))
        self.inp2 = nn.Sequential(*_cbr(32, 64, (1, 1)), *_cbr(64, 64, (5, 1), pad=(2, 0)))
        self.inp3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)), *_cbr(32, 64, (1, 1)))

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)


class DeepLOB(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = DeepLOBFeatures()
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


class DeepLOBNoLSTM(nn.Module):

    def __init__(self, n_classes=3):
        super().__init__()
        self.features = DeepLOBFeatures()
        self.temporal = nn.Sequential(
            *_cbr(192, 64, (5, 1), (2, 1)), *_cbr(64, 64, (5, 1), (2, 1)), *_cbr(64, 64, (5, 1), (3, 1)))
        self.fc = nn.Linear(64 * 5, n_classes)

    def forward(self, x):
        x = self.temporal(self.features(x))
        return self.fc(torch.flatten(x, 1))


def _layers(bits, first=False, last=False, abits=None):
    if bits is None:
        return nn.Conv2d, nn.Linear, (lambda: nn.ReLU()), None
    import brevitas.nn as qnn
    ab = abits or bits
    wb = 8 if (first or last) else bits
    conv = lambda *a, **k: qnn.QuantConv2d(*a, weight_bit_width=wb, **k)
    lin = lambda *a, **k: qnn.QuantLinear(*a, weight_bit_width=wb, **k)
    relu = lambda: qnn.QuantReLU(bit_width=ab)
    return conv, lin, relu, qnn.QuantIdentity(bit_width=8)


class LightLOB(nn.Module):

    def __init__(self, n_classes=3, c=16, bits=None, abits=None):
        super().__init__()
        self.bits = bits
        conv0, _, relu, inq = _layers(bits, first=True, abits=abits)
        conv, _, _, _ = _layers(bits, abits=abits)
        _, lin, _, _ = _layers(bits, last=True, abits=abits)

        def cbr(cv, cin, cout, k, s=1):
            return [cv(cin, cout, k, stride=s, bias=False), nn.BatchNorm2d(cout), relu()]

        self.inq = inq if inq is not None else nn.Identity()
        self.body = nn.Sequential(
            *cbr(conv0, 1, c, (1, 2), (1, 2)), *cbr(conv, c, c, (4, 1)),
            *cbr(conv, c, c, (1, 2), (1, 2)), *cbr(conv, c, c, (4, 1)),
            *cbr(conv, c, c, (1, 10)), *cbr(conv, c, c, (4, 1)),
            *cbr(conv, c, 2 * c, (5, 1), (2, 1)),
            *cbr(conv, 2 * c, 2 * c, (5, 1), (2, 1)),
            *cbr(conv, 2 * c, 2 * c, (5, 1), (3, 1)),
        )
        self.fc = lin(2 * c * 6, n_classes, bias=True)

    def forward(self, x):
        x = self.body(self.inq(x))
        return self.fc(torch.flatten(x, 1))


class LightLOB2(nn.Module):

    def __init__(self, n_classes=3, c=32, bits=None, abits=None):
        super().__init__()
        self.bits = bits
        conv0, _, relu, inq = _layers(bits, first=True, abits=abits)
        conv, _, _, _ = _layers(bits, abits=abits)
        _, lin, _, _ = _layers(bits, last=True, abits=abits)

        def cbr(cv, cin, cout, k, s=1):
            return [cv(cin, cout, k, stride=s, bias=False), nn.BatchNorm2d(cout), relu()]

        self.inq = inq if inq is not None else nn.Identity()
        self.body = nn.Sequential(
            *cbr(conv0, 40, c, (1, 1)),
            *cbr(conv, c, c, (4, 1)), *cbr(conv, c, c, (4, 1)), *cbr(conv, c, c, (4, 1)),
            *cbr(conv, c, c, (5, 1), (2, 1)),
            *cbr(conv, c, c, (5, 1), (2, 1)),
            *cbr(conv, c, c, (5, 1), (3, 1)),
        )
        self.fc = lin(c * 6, n_classes, bias=True)

    def forward(self, x):
        if x.shape[1] == 1 and x.shape[-1] == 40:
            x = x.permute(0, 3, 2, 1)
        x = self.body(self.inq(x))
        return self.fc(torch.flatten(x, 1))


EXPORT_SHAPE = {"light": (1, 1, 100, 40), "light2": (1, 40, 100, 1)}


def build(name, bits=None, abits=None):
    if name == "deeplob":
        return DeepLOB()
    if name == "deeplob_nolstm":
        return DeepLOBNoLSTM()
    if name == "light":
        return LightLOB(bits=bits, abits=abits)
    if name == "light2":
        return LightLOB2(bits=bits, abits=abits)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
