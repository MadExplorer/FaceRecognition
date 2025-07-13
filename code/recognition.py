class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
        self.easy_margin = easy_margin

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class FaceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self._validate_data()

    def _validate_data(self):
        valid = []
        for idx, row in self.df.iterrows():
            path = os.path.join(self.img_dir, row['image_id'])
            if os.path.exists(path): valid.append(idx)
        self.df = self.df.loc[valid].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row['image_id']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label = int(row['label'])
        return img, label

class FaceRecognitionSystem:
    def __init__(self, config):
        self.config = {
            'identity_file': './identity_CelebA.txt',
            'img_dir': './detected_faces',
            'min_samples': 5,
            'max_classes': 100,
            'test_size': 0.2,
            'batch_size': 32,
            'lr': 1e-3,
            'epochs': 10,
            'margin_s': 30.0,
            'margin_m': 0.50,
            'easy_margin': False
        }
        self.config.update(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        df = pd.read_csv(self.config['identity_file'], sep=' ', names=['image_id','identity_id'])
        imgs = set(os.listdir(self.config['img_dir']))
        df = df[df['image_id'].isin(imgs)]
        counts = df['identity_id'].value_counts()
        valid = counts[counts>=self.config['min_samples']].index
        valid = valid[:self.config['max_classes']]
        df = df[df['identity_id'].isin(valid)]
        self.le = LabelEncoder()
        df['label'] = self.le.fit_transform(df['identity_id'])
        self.num_classes = len(self.le.classes_)
        self.train_df, self.val_df = train_test_split(df, test_size=self.config['test_size'],
                                                     stratify=df['label'], random_state=42)

    def _init_model(self):
        self.backbone = models.resnet18(pretrained=True)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.ce_head = nn.Linear(feat_dim, self.num_classes)
        self.arc_head = ArcMarginProduct(feat_dim, self.num_classes,
                                         s=self.config['margin_s'], m=self.config['margin_m'],
                                         easy_margin=self.config['easy_margin'])
        self.backbone = self.backbone.to(self.device)
        self.ce_head = self.ce_head.to(self.device)
        self.arc_head = self.arc_head.to(self.device)

    def train_ce(self):
        train_ds = FaceDataset(self.train_df, self.config['img_dir'])
        loader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True)
        opt = torch.optim.Adam(list(self.backbone.parameters())+list(self.ce_head.parameters()), lr=self.config['lr'])
        crit = nn.CrossEntropyLoss()

        for ep in range(self.config['epochs']):
            self.backbone.train(); self.ce_head.train()
            total, acc = 0, 0
            for X,y in loader:
                X,y = X.to(self.device), y.to(self.device)
                feats = self.backbone(X)
                logits = self.ce_head(feats)
                loss = crit(logits,y)
                opt.zero_grad(); loss.backward(); opt.step()
                preds = logits.argmax(1)
                total += y.size(0); acc += (preds==y).sum().item()
            print(f"[CE] Ep {ep+1}/{self.config['epochs']} Acc: {acc/total:.4f}")

    def train_arcface(self):
        train_ds = FaceDataset(self.train_df, self.config['img_dir'])
        loader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True)
        opt = torch.optim.Adam(list(self.backbone.parameters())+list(self.arc_head.parameters()), lr=self.config['lr'])
        crit = nn.CrossEntropyLoss()

        for ep in range(self.config['epochs']):
            self.backbone.train(); self.arc_head.train()
            total, acc = 0, 0
            for X,y in loader:
                X,y = X.to(self.device), y.to(self.device)
                feats = self.backbone(X)
                logits = self.arc_head(feats, y)
                loss = crit(logits,y)
                opt.zero_grad(); loss.backward(); opt.step()
                preds = logits.argmax(1)
                total += y.size(0); acc += (preds==y).sum().item()
            print(f"[ArcFace] Ep {ep+1}/{self.config['epochs']} Acc: {acc/total:.4f}")

    def extract_embeddings(self, images_batch):
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(images_batch.to(self.device))
            return F.normalize(feats)

if __name__ == '__main__':
    cfg = {
        'identity_file': '/kaggle/input/celeba-meta/identity_CelebA.txt',
        'img_dir': '/kaggle/working/detected_faces',
        'epochs': 5
    }
    sys = FaceRecognitionSystem(cfg)
    sys.train_ce()
    sys.train_arcface()
