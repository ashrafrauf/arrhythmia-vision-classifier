import torch
import torch.nn as nn

# -- References ---
# This was an initial attempt to recreate the 1D-CNN model used in Yilidirim et al (2020).
# However, it wasn't used in the final project.

# Source:
# Yildirim, O. et al. (2020) “Accurate deep neural network model to detect cardiac arrhythmia on more than 10,000 individual subject ECG records,”
# Computer Methods and Programs in Biomedicine, 197, p. 105740. Available at: https://doi.org/10.1016/j.cmpb.2020.105740.



class YildirimDNNModel(nn.Module):
    def __init__(
            self,
            n_classes = 4,
            in_channels = 1,
            input_length = 5000
    ):
        super().__init__()
        
        ## --- Representation Learning -- ##
        # Conv1: filters=64, kernel_size=21, stride=11
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=21, stride=11),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )

        # Conv2: filters=64, kernel_size=7, stride=1
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=1),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64)
        )

        # Conv3: filters=128, kernel_size=5, stride=1
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.MaxPool1d(kernel_size=2)
        )

        # Conv4: filters=256, kernel_size=13, stride=1
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=13, stride=1)
        )

        # Conv5: filters=512, kernel_size=7, stride=1
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=7, stride=1),
            nn.Dropout(0.3)
        )

        # Conv6: filters=256, kernel_size=9, stride=1
        self.conv6 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=9, stride=1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        ## -- Sequence Learning -- ##
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)

        # Infer LSTM output.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            out = self._forward_conv(dummy)
            out = out.permute(0, 2, 1).contiguous()
            out, _ = self.lstm(out)
            flatten_dim = out.numel()

        ## -- Classification -- ##
        self.seq_len_post_conv = self._compute_seq_len(input_length)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def forward(self, x):
        # -- Representation Learning -- 
        x = self._forward_conv(x)

        # -- Sequence Learning --
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)

        # -- Classification
        x = self.classifier(x)

        return x