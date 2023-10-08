import cv2
import math
import os
from util.util import assure_path_exists

def resize_image(img, x, y):
    dsize = (x, y)
    img_resized = cv2.resize(img, dsize)
    return img_resized


if __name__ == '__main__':
    # width and height for image resizing
    new_size_x = 512
    new_size_y = 512
    # tile and offset size, if offset size is same as tile size, the tiles are non-overlapping, if smaller, they are overlapping, if bigger, some pixels are skipped
    tile = 192
    offs = 256 #int(tile - tile / 2)
    tile_size = (tile, tile)
    offset = (offs, offs)
    # build path to input
    task = "test"
    res = "x40"
    domain = "H"
    #nums = ["03", "04", "05", "07", "10", "11", "12", "14", "15", "17", "18"]
    nums = ["03", "04", "05"]
    #nums = ["06", "08", "09", "13", "16"]
    for num in nums:
        folder = domain + num
        input_dir = "/home/rhack/stainTransfer_CycleGAN_pytorch/FakultyData/extract"
        # path to output
        output_dir = "/home/rhack/stainTransfer_CycleGAN_pytorch/FakultyData/result/"
        print(output_dir)
        assure_path_exists(output_dir)

        # go through each image in directory
        for filename in os.listdir(input_dir):
            # read image
            img = cv2.imread(os.path.join(input_dir, filename))
            # resize image
            img_resized = resize_image(img, new_size_x, new_size_y)
            # extract tiles
            img_shape = img_resized.shape

            for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
                for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
                    cropped_img = img[offset[1] * i:min(offset[1] * i + tile_size[1], img_shape[0]),
                                  offset[0] * j:min(offset[0] * j + tile_size[0], img_shape[1])]
                    # Debugging the tiles
                    cv2.imwrite(output_dir + filename[:-5] + "_tile_" + str(i) + "_" + str(j) + ".tiff", cropped_img)