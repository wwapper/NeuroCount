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
image = cv2.copyMakeBorder(src=image, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT,value=(255, 255, 255))
image = increase_brightness(image, value=10)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image = cv2.filter2D(image, -1, kernel)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg", gray_image)
gray_image_blur = cv2.medianBlur(gray_image, 5)
cv2.imwrite("image_2.jpg", gray_image_blur)


ret, threshold_image = cv2.threshold(gray_image_blur, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite("threshold_image.jpg", threshold_image)

threshold_image_blur = cv2.GaussianBlur(threshold_image, (3, 3), 5)
cv2.imwrite("threshold_image_blur.jpg", threshold_image_blur)

ret, threshold_image_new = cv2.threshold(threshold_image_blur, 120, 255, cv2.THRESH_BINARY)
cv2.imwrite("threshold_image_new.jpg", threshold_image_new)


#контурирование
edged = cv2.Canny(threshold_image_new, 0, 300, apertureSize = 7, L2gradient=True)
cv2.imwrite("edged.jpg", edged)

#удаление белых точек по размеру
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(edged)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = 80
max_size = 1000
im_result = np.zeros((edged.shape))
for blob in range(nb_blobs):
    if max_size >= sizes[blob] >= min_size:
        im_result[im_with_separated_blobs == blob + 1] = 255
cv2.imwrite("edhed_corrected.jpg", im_result)

#достраивание контуров на изображении без белых точек
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed_final = cv2.morphologyEx(im_result, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed_edged_corrected.jpg", closed_final)
closed_final = closed_final.astype(np.uint8)

#контурирование на картинке
image1 = image.copy()
im_result = im_result.astype(np.uint8)
contours, hierarchy = cv2.findContours(im_result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# отображаем контуры поверх изображения
cv2.drawContours(image1, contours, -1, (0,0,0), 2, cv2.LINE_AA)
cv2.imwrite('contours.jpg', image1)

#контурирование на картинке с условием на площадь и периметр, чтобы убрать артефакты
closed_final = closed_final.astype(np.uint8)
contours, hierarchy = cv2.findContours(closed_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area/perimeter > 2.7:
        new_contours.append(cnt)
# отображаем контуры поверх изображения
cv2.drawContours(image, new_contours, -1, (0,0,0), 2, cv2.LINE_AA)
cv2.imwrite('contours_new.jpg', image)


coord = []
for c in new_contours:
  M = cv2.moments(c)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  cv2.drawContours(image, [c], -1, (0, 0, 0), 2)
  cv2.circle(image, (cX, cY), 7, (0, 0, 0), -1)
  cv2.putText(image, "center", (cX - 20, cY - 20),
  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
  coord.append((cX,cY))
cv2.imwrite('contours_count.jpg', image)
print(len(coord))


