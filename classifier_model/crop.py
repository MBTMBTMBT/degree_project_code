import cv2
import os

# prepare lists for images and annotations
img_path_list = list()
img_dir = r'E:\my_files\programmes\python\dp_dataset\full_dataset\imgs'
for each_img in os.listdir(img_dir):
    if each_img.split('.')[-1] == 'jpg' \
            or each_img.split('.')[-1] == 'JPG' \
            or each_img.split('.')[-1] == 'png':  # or each_img.split('.')[-1] == 'webp':
        img_path = os.path.join(img_dir, each_img)
        img_path_list.append(img_path)
    else:
        print("Exception: ", each_img)

for img_path in img_path_list:
    img = cv2.imread(img_path)
    cv2.imshow('original', img)
    img = cv2.resize(img, (800, 600))

    roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    print(roi)

    if roi != (0, 0, 0, 0):
        crop = img[y:y+h, x:x+w]
        # cv2.imshow('crop', crop)
        # print(os.path.join(r'C:\Users\13769\desktop', os.path.basename(img_path)) + 'crop.png')
        cv2.imwrite(os.path.join(r'C:\Users\13769\desktop', os.path.basename(img_path)) + 'head0.png', crop)
        print('Saved!')
cv2.destroyAllWindows()
