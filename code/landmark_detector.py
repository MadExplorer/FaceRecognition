class FaceLandmarksDataset(Dataset):
    def __init__(self, img_dir, landmarks_file, transform=None, img_size=256, sigma=2):
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.sigma = sigma
        df = pd.read_csv(landmarks_file)
        self.image_names = df['image_id'].values.tolist()
        coords = df.drop(columns=['image_id']).values.astype(np.float32)
        if coords.shape[1] % 2 != 0:
            raise ValueError("Landmarks file must have even coordinate columns.")
        self.num_landmarks = coords.shape[1] // 2
        self.landmarks = coords.reshape(-1, self.num_landmarks, 2)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        landmarks = self.landmarks[idx]
        img_path = os.path.join(self.img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img = cv2.resize(img, (self.img_size, self.img_size))
        scale_x, scale_y = self.img_size / w, self.img_size / h
        landmarks = landmarks * [scale_x, scale_y]
   
        heatmaps = np.zeros((self.num_landmarks, self.img_size, self.img_size), dtype=np.float32)
        for i, (x, y) in enumerate(landmarks):
            xx, yy = int(x), int(y)
            ul, br = [xx - 3*self.sigma, yy - 3*self.sigma], [xx + 3*self.sigma + 1, yy + 3*self.sigma + 1]
            if ul[0] >= self.img_size or ul[1] >= self.img_size or br[0] < 0 or br[1] < 0:
                continue
            size = 6 * self.sigma + 1
            x_range = np.arange(size)
            y_range = x_range[:, None]
            gaussian = np.exp(-((x_range - 3*self.sigma)**2 + (y_range - 3*self.sigma)**2) / (2*self.sigma**2))
            g_x = max(0, -ul[0]), min(br[0], self.img_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.img_size) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.img_size)
            img_y = max(0, ul[1]), min(br[1], self.img_size)
            heatmaps[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                heatmaps[i, img_y[0]:img_y[1], img_x[0]:img_x[1]],
                gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            )
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img), torch.tensor(heatmaps)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        r = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + r)

class HourglassModule(nn.Module):
    def __init__(self, depth, num_features, block):
        super().__init__()
        self.depth = depth
        self.block = block
        self.num_features = num_features
        self._generate_network(depth)
    def _generate_network(self, level):
        setattr(self, f"b1_{level}", self.block(self.num_features, self.num_features))
        setattr(self, f"pool_{level}", nn.MaxPool2d(2,2))
        setattr(self, f"b2_{level}", self.block(self.num_features, self.num_features))
        if level>1: self._generate_network(level-1)
        else: self.b2_plus = self.block(self.num_features, self.num_features)
        setattr(self, f"b3_{level}", self.block(self.num_features, self.num_features))
        setattr(self, f"upsample_{level}", nn.Upsample(scale_factor=2, mode='nearest'))
    def _forward(self, level, x):
        up1 = getattr(self, f"b1_{level}")(x)
        low1 = getattr(self, f"pool_{level}")(x)
        low1 = getattr(self, f"b2_{level}")(low1)
        low2 = self._forward(level-1, low1) if level>1 else self.b2_plus(low1)
        low3 = getattr(self, f"b3_{level}")(low2)
        up2 = getattr(self, f"upsample_{level}")(low3)
        return up1 + up2
    def forward(self, x): return self._forward(self.depth, x)

class StackedHourglass(nn.Module):
    def __init__(self, num_stacks=2, num_features=256, num_landmarks=68):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64,128), nn.MaxPool2d(2,2), ResidualBlock(128,128), ResidualBlock(128,num_features)
        )
        self.hgs = nn.ModuleList([HourglassModule(4,num_features,ResidualBlock) for _ in range(num_stacks)])
        self.features = nn.ModuleList([nn.Sequential(
            ResidualBlock(num_features,num_features), nn.Conv2d(num_features,num_features,1),
            nn.BatchNorm2d(num_features), nn.ReLU(inplace=True)
        ) for _ in range(num_stacks)])
        self.out_convs = nn.ModuleList([nn.Conv2d(num_features,num_landmarks,1) for _ in range(num_stacks)])
        self.merge_features = nn.ModuleList([nn.Conv2d(num_features,num_features,1) for _ in range(num_stacks-1)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(num_landmarks,num_features,1) for _ in range(num_stacks-1)])
    def forward(self, x):
        x = self.pre(x)
        outputs = []
        for i in range(len(self.hgs)):
            hg = self.hgs[i](x)
            feat = self.features[i](hg)
            preds = self.out_convs[i](feat)
            outputs.append(preds)
            if i < len(self.hgs)-1:
                x = x + self.merge_features[i](feat) + self.merge_preds[i](preds)
        return outputs

def train(model, dataloader, device, epochs=10, lr=2e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for imgs, heatmaps in dataloader:
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            gt_resized = [F.interpolate(heatmaps, size=p.size()[2:], mode='bilinear', align_corners=False) for p in preds]
            loss = sum(criterion(p, gt) for p, gt in zip(preds, gt_resized))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.4f}", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = FaceLandmarksDataset(img_dir, landmarks_file)
dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
num_landmarks = dataset.dataset.num_landmarks
model = StackedHourglass(num_stacks=2, num_features=256, num_landmarks=num_landmarks)
train(model, loader, device)
