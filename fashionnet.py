import torch
import torch.nn as nn

class FashionNetSmall(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()

        # Feature extractor: stem + 3 stages
        self.features = nn.Sequential(
            # ---- Stem ----
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # ---- Stage 1 (local textures) ----
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # ---- Stage 2 (bigger patterns) ----
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # ---- Stage 3 (silhouette / global) ----
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)   # [B, 256]
        return x

    def forward(self, x):
        x = self.features(x)          # [B, 256, H', W']
        x = self.pool(x)              # [B, 256, 1, 1]
        x = torch.flatten(x, 1)       # [B, 256]
        #x = self.dropout(x)
        x = self.classifier(x)        # [B, num_classes]
        return x
    


class FashionNetMedium(nn.Module):
    def __init__(self, num_classes: int = 50):
        super().__init__()

        # Feature extractor: stem + 3 stages (wider + slightly deeper than Small)
        self.features = nn.Sequential(
            # ---- Stem ----
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 224 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # ---- Stage 1 (local textures, widened to 128ch) ----
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # ---- Stage 2 (bigger patterns, widened/deepened) ----
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # ---- Stage 3 (silhouette / global, widened/deepened) ----
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # optional, currently not used in forward

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)   # [B, 384]
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)   # [B, 384]
        # x = self.dropout(x)     # enable if you see overfitting
        x = self.classifier(x)    # [B, num_classes]
        return x
