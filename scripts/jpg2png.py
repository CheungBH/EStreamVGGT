import imageio.v2 as iio

img = iio.imread("in.jpg")
iio.imwrite("out.png", img)