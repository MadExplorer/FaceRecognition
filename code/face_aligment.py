def align_face(orig_img, landmarks, output_size=(256,256)):

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pts_src = np.float32([
        landmarks[0],
        landmarks[1],
        landmarks[2],
    ])
    w, h = output_size
    pts_dst = np.float32([pt * [w, h] for pt in TEMPLATE])

    M = cv2.getAffineTransform(pts_src, pts_dst)
    aligned = cv2.warpAffine(orig_img, M, output_size, flags=cv2.INTER_LINEAR)
    return aligned

    w, h = output_size
    pts_dst = np.float32([pt * [w, h] for pt in TEMPLATE])
    M = cv2.getAffineTransform(pts_src, pts_dst)
    aligned = cv2.warpAffine(orig_img, M, output_size, flags=cv2.INTER_LINEAR)
    return aligned
