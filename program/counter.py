import numpy as np
import cv2
import imutils

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

image = cv2.imread("example.jpg")
image = increase_brightness(image, value=10)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image = cv2.filter2D(image, -1, kernel)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg", gray_image)
gray_image = cv2.GaussianBlur(gray_image,(3,3),5)
ret, threshold_image = cv2.threshold(gray_image, 110, 255, 0)
cv2.imwrite("threshold_image.jpg", threshold_image)

#контурирование
edged = cv2.Canny(threshold_image, 0, 250, apertureSize = 7, L2gradient=True)
cv2.imwrite("edged.jpg", edged)

#достраивание контуров
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed.jpg", closed)

#удаление белых точек по размеру
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(edged)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = 80
im_result = np.zeros((edged.shape))
for blob in range(nb_blobs):
    if 500 >= sizes[blob] >= min_size:
        im_result[im_with_separated_blobs == blob + 1] = 255
cv2.imwrite("final.jpg", im_result)

#достраивание контуров на изображении без белых точек
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed_final = cv2.morphologyEx(im_result, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed_final.jpg", closed_final)

#контурирование на картинке
image1 = image.copy()
im_result = im_result.astype(np.uint8)
contours, hierarchy = cv2.findContours(im_result.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# отображаем контуры поверх изображения
cv2.drawContours(image1, contours, -1, (0,0,0), 3, cv2.LINE_AA)
cv2.imwrite('contours_old.jpg', image1)

#контурирование на картинке с условием на площадь и периметр, чтобы убрать артефакты
closed_final = closed_final.astype(np.uint8)
contours, hierarchy = cv2.findContours(closed_final.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
new_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area/perimeter > 2.8:
        new_contours.append(cnt)
# отображаем контуры поверх изображения
cv2.drawContours(image, new_contours, -1, (0,0,0), 3, cv2.LINE_AA)
cv2.imwrite('contours_new.jpg', image)

#подсчет кол-ва нейронов. Нужно доработать, тк считает артефакты
closed_final = closed_final.astype(np.uint8)
cnts = cv2.findContours(closed_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
total = 0
for c in cnts:
    total += 1
print(f'На изображении {total} нейрон(ов)')
