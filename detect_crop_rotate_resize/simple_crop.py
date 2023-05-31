import cv2

x = 100  # top-left x-coordinate
y = 100  # top-left y-coordinate
w = 200  # width of the region
h = 200  # height of the region

img = cv2.imread("Com_gmapping.pgm")
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)