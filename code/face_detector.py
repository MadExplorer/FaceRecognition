face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
os.makedirs(output_dir, exist_ok=True)

all_images = sorted([fn for fn in os.listdir(input_dir) if fn.endswith('.jpg')])
img_files = all_images[:5000]

for idx, fn in enumerate(img_files):
    img_path = os.path.join(input_dir, fn)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        continue
    
    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (160, 160))
    
    out_path = os.path.join(output_dir, fn)
    cv2.imwrite(out_path, face_crop)
    
    if idx == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.add_patch(plt.Rectangle(
            (x, y), w, h,
            fill=False, edgecolor='red', linewidth=2
        ))
        ax1.set_title('Оригинал + бокс')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        ax2.set_title('Вырезанное лицо (160×160)')
        ax2.axis('off')
        
        plt.show()
