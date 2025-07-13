def predict_and_plot(model, dataset, device, index=2):
    model.eval()
    ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    img_dir = ds.img_dir
    name = ds.image_names[index]
    img_path = os.path.join(img_dir, name)
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h0, w0, _ = orig.shape

    img_resized, _ = dataset[index]
    input_img = img_resized.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_img)
    heat = preds[-1][0].cpu()
    H, W = heat.shape[1], heat.shape[2]

    landmarks_pred = []
    for hm in heat:
        idx = torch.argmax(hm)
        y, x = divmod(idx.item(), W)
        x0 = x * (w0 / W)
        y0 = y * (h0 / H)
        landmarks_pred.append((int(x0), int(y0)))

    plt.figure(figsize=(6,6))
    plt.imshow(orig)
    for x0, y0 in landmarks_pred:
        plt.plot(x0, y0, 'ro', markersize=2)
    plt.title(f"Predicted Landmarks on Original (idx {index})")
    plt.axis('off')
    plt.show()

predict_and_plot(model, dataset, device, index=2)
