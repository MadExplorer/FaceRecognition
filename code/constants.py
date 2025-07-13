input_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
output_dir = '/kaggle/working/detected_faces'
img_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
landmarks_file = '/kaggle/input/celeba-dataset/list_landmarks_align_celeba.csv'
batch_size = 8
subset_size = 5000
TEMPLATE = np.float32([
    [0.35, 0.35],
    [0.65, 0.35],
    [0.5,  0.55],
])
