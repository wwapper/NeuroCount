import numpxy as np
import cv2
import imutils

image = cv2.imread("example.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image,(3,3),5)
ret, threshold_image = cv2.threshold(gray_image, 125, 255, 0)
cv2.imwrite("threshold_image.jpg", threshold_image)

#контурирование
edged = cv2.Canny(threshold_image, 0, 250, apertureSize = 5, L2gradient=True)
cv2.imwrite("edged.jpg", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed.jpg", closed)

#удаление белых точек
nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(edged)
sizes = stats[:, -1]
sizes = sizes[1:]
nb_blobs -= 1
min_size = 50
im_result = np.zeros((edged.shape))
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        im_result[im_with_separated_blobs == blob + 1] = 255
cv2.imwrite("final.jpg", im_result)
