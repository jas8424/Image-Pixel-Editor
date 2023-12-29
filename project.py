import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3

# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # Raise exceptions here
        if not isinstance(pixels, list):
            raise TypeError()
        if len(pixels) < 1:
            raise TypeError()
        if not all([isinstance(i,list) for i in pixels]):
            raise TypeError()
        if not all([True if len(i) == len(pixels[0]) else False for i in \
            pixels]):
            raise TypeError()
        if not all( [isinstance(j,list) and len(j) == 3 for i in pixels for j \
            in i] ):
            raise TypeError()
        if not all( [k >= 0 and k <= 255 for i in pixels for j in i for k \
            in j] ):
            raise ValueError()

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[j[:] for j in i]for i in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)   
        True
        """
        return self.get_pixels()

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not (isinstance(row,int) and isinstance(col,int)):
            raise TypeError()
        if row >= self.num_rows or row < 0:
            raise ValueError()
        if col >= self.num_cols or col < 0:
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not (isinstance(row,int) and isinstance(col,int)):
            raise TypeError()
        if row >= self.num_rows or row < -self.num_rows:
            raise ValueError()
        if col >= self.num_cols or row < -self.num_cols:
            raise ValueError()
        if not (isinstance(new_color, tuple) and len(new_color) == 3 and \
            all([isinstance(i,int) for i in new_color])):
            raise TypeError()
        if not all([i <= 255 for i in new_color]):
            raise ValueError()
        for count,i in enumerate(new_color):
            if i < 0:
                continue
            self.pixels[row][col][count] = i

# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """

        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/gradient_16x16.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        new_img = RGBImage(image.get_pixels())
        new_img.pixels = [[[255 - k for k in j]for j in i] for i in \
        new_img.pixels]
        return new_img

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)

       
        """
        new_img = RGBImage(image.get_pixels())
        new_img.pixels = [[[sum(j)// 3 for k in j]for j in i] for i in \
        new_img.pixels]
        return new_img
    
    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)

        """
        new_img = RGBImage(image.get_pixels())
        new_img.pixels = [[[k for k in j] for j in i[::-1]] for i in \
        new_img.pixels[::-1]]
        return new_img
    
    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.get_average_brightness(img)
        133
        """

        avg_bri = sum([ sum([sum(j) // 3 for j in i]) for i in image.pixels]) \
         // (image.num_rows * image.num_cols)
        return avg_bri
    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/gradient_16x16_adjusted.png', img_adjust)

        """
        if not isinstance(intensity,int):
            raise TypeError()
        if intensity > 255 or intensity < -255:
            raise ValueError
        new_img = RGBImage(image.get_pixels())
        pixels_one = [[[k + intensity if k+intensity <= 255 else 255 for k in j]for j in i] for i in \
        new_img.pixels]
        new_img.pixels = [ [ [k if k >=0 else 0 for k in j] for j in i]for i in pixels_one]
        return new_img


    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_blur.png')
        >>> img_adjust = img_proc.blur(img)
        >>> img_adjust.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/gradient_16x16_blur.png', img_adjust)

        """
        new_img = RGBImage(image.get_pixels())
        for c1,rows in enumerate(image.pixels):
            for c2,cols in enumerate(rows):
                for c3,pixel in enumerate (cols):
                    if c1 == 0:
                        if c2 == 0:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 + 1][c2][c3] + image.pixels[c1 + 1][c2 + 1][c3] + image.pixels[c1][c2 + 1][c3])//4
                            continue
                        elif c1 == 0 and c2 == image.num_cols - 1:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 + 1][c2][c3] + image.pixels[c1 + 1][c2 - 1][c3] + image.pixels[c1][c2 - 1][c3])//4
                            continue
                        else:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 + 1][c2][c3] + image.pixels[c1 + 1][c2 - 1][c3] + image.pixels[c1 + 1][c2 + 1][c3] + image.pixels[c1][c2 - 1][c3] + image.pixels[c1][c2 + 1][c3])//6
                            continue
                    if c2 == 0:
                        if c1 == image.num_rows - 1:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 - 1][c2][c3] + image.pixels[c1 - 1][c2 + 1][c3] + image.pixels[c1][c2 + 1][c3])//4
                            continue
                        else:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 + 1][c2][c3] + image.pixels[c1 + 1][c2 + 1][c3] + image.pixels[c1 - 1][c2 + 1][c3] + image.pixels[c1 -1][c2][c3] + image.pixels[c1][c2 + 1][c3])//6
                            continue
                    if c1 == image.num_rows - 1:
                        if c2 == image.num_cols - 1:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 - 1][c2][c3] + image.pixels[c1 - 1][c2 - 1][c3] + image.pixels[c1][c2 - 1][c3])//4
                            continue
                        else:
                            new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 - 1][c2][c3] + image.pixels[c1 - 1][c2 + 1][c3] + image.pixels[c1 - 1][c2 - 1][c3] + image.pixels[c1][c2 - 1][c3] + image.pixels[c1][c2 + 1][c3])//6
                            continue
                    if c2 == image.num_cols - 1:
                        new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1][c2 - 1][c3] + image.pixels[c1 + 1][c2 - 1][c3] + image.pixels[c1 - 1][c2 - 1][c3] + image.pixels[c1 - 1][c2][c3] + image.pixels[c1 + 1][c2][c3])//6
                        continue
                    new_img.pixels[c1][c2][c3] = (pixel + image.pixels[c1 + 1][c2][c3] + image.pixels[c1 + 1][c2 - 1][c3] + image.pixels[c1 + 1][c2 + 1][c3] + image.pixels[c1 - 1][c2][c3] + image.pixels[c1 - 1][c2 - 1][c3] + image.pixels[c1 - 1][c2 + 1][c3] + image.pixels[c1][c2 + 1][c3] + image.pixels[c1][c2 - 1][c3] )//9
        return new_img



    # Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """
    
    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        super().__init__()
        self.cost = 0
        self.free_calls = 0
        self.calls = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        self.calls += 1
        if self.calls <= self.free_calls:
            self.cost = 0
        else:
            self.cost += 5 
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image
        """
        self.calls += 1
        if self.calls <= self.free_calls:
            self.cost = 0
        else:
            self.cost += 6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        self.calls += 1
        if self.calls <= self.free_calls:
            self.cost = 0
        else:
            self.cost += 10
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        self.calls += 1
        if self.calls <= self.free_calls:
            self.cost = 0
        else:
            self.cost += 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        # YOUR CODE GOES HERE #
        self.calls += 1
        if self.calls <= self.free_calls:
            self.cost = 0
        else:
            self.cost += 5
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        self.free_calls = self.calls + amount

# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given 
        color are replaced with the background image. 

        # Check output 
        >>> img_proc = PremiumImageProcessing() 
        >>> img_in = img_read_helper('img/square_16x16.png') 
        >>> img_in_back = img_read_helper('img/gradient_16x16.png') 
        >>> color = (255, 255, 255) 
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png') 
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color) 
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output 
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma) 
        """
        if not isinstance(chroma_image, RGBImage) or not isinstance(background_image, RGBImage):
            raise TypeError() 
        
        if chroma_image.size() != background_image.size(): 
            raise ValueError() 
        
        result_image = RGBImage(chroma_image.get_pixels())

        for i in range(chroma_image.size()[0]): 
            for j in range(chroma_image.size()[1]): 
                chroma_pixel = chroma_image.get_pixel(i, j) 
                if chroma_pixel == color: 
                    background_pixel = background_image.get_pixel(i, j) 
                    result_image.set_pixel(i, j, background_pixel) 
                else: 
                    result_image.set_pixel(i, j, chroma_pixel) 
                    
        return result_image 
        

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if not isinstance(sticker_image, RGBImage) or not isinstance(background_image, RGBImage):
            raise TypeError()
    
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        
        sticker_width, sticker_height = sticker_image.size()
        background_width, background_height = background_image.size()
        
        if sticker_width > background_width or sticker_height > background_height:
            raise ValueError()
        
        max_x_pos = background_width - sticker_width
        max_y_pos = background_height - sticker_height
        
        if x_pos > max_x_pos or y_pos > max_y_pos:
            raise ValueError()
        
        result_image = RGBImage(background_image.get_pixels())
        
        for i in range(sticker_width):
            for j in range(sticker_height):
                pixel = sticker_image.get_pixel(i, j)
                result_image.set_pixel(x_pos + i, y_pos + j, pixel)
        return result_image


    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/gradient_16x16.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/gradient_16x16_edge.png', img_edge)
        """
        new_image = RGBImage(image.get_pixels())
        x = [[sum(j)// 3 for j in i] for i in image.pixels]
        y = [[sum(j)// 3 for j in i] for i in image.pixels]

        for c1, rows in enumerate(x):
            for c2, cols in enumerate(rows):
                if c1 == 0:
                    if c2 == 0:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 + 1][c2] * -1 + x[c1 + 1][c2 + 1] * -1 + x[c1][c2 + 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                    elif c2 == len(x[c1]) - 1: 
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 + 1][c2] * -1 + x[c1 + 1][c2 - 1] * -1 + x[c1][c2 - 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                    else:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 + 1][c2] * -1 + x[c1 + 1][c2 + 1] * -1 + x[c1][c2 + 1] * -1 + x[c1][c2 - 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                elif c1 == len(x) - 1:
                    if c2 == 0:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 - 1][c2] * -1 + x[c1 - 1][c2 + 1] * -1 + x[c1][c2 + 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                    elif c2 == len(x[c1]) - 1:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 - 1][c2] * -1 + x[c1 - 1][c2 - 1] * -1 + x[c1][c2 - 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                    else:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 - 1][c2] * -1 + x[c1 - 1][c2 + 1] * -1 + x[c1][c2 + 1] * -1 + x[c1][c2 - 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                else:
                    if c2 == 0:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 + 1][c2] * -1 + x[c1 + 1][c2 + 1] * -1 + x[c1][c2 + 1] * -1 + x[c1 - 1][c2] * -1 + x[c1 - 1][c2 + 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                    elif c2 == len(x[c1]) - 1:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 + 1][c2] * -1 + x[c1 + 1][c2 - 1] * -1 + x[c1][c2 - 1] * -1 + x[c1 - 1][c2] * -1 + x[c1 - 1][c2 - 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
                    else:
                        y[c1][c2] = (x[c1][c2] * 8 + x[c1 + 1][c2] * -1 + x[c1 + 1][c2 + 1] * -1 + x[c1][c2 + 1] * -1 + x[c1 - 1][c2] * -1 + x[c1 - 1][c2 + 1] * -1 + x[c1 + 1][c2 - 1] * -1 + x[c1][c2 - 1] * -1 + x[c1 - 1][c2 - 1] * -1)
                        if y[c1][c2] < 0:
                            y[c1][c2] = 0
                        if y[c1][c2] > 255:
                            y[c1][c2] = 255
                        continue
        y = [[[j,j,j] for j in i] for i in y]
        new_image.pixels = y
        return new_image 

    
# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not (isinstance(image1,RGBImage) and isinstance(image2,RGBImage)):
            raise TypeError()
        if not (image1.num_rows == image2.num_rows and image1.num_cols == image2.num_cols):
            raise ValueError()
        return sum( [ (image1.pixels[i][j][k] - image2.pixels[i][j][k])**2 for i in range(image1.num_rows) for j in range(image1.num_cols) for k in range(3)]) ** 0.5

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        count = {}
        for i in candidates:
            if i in count:
                count[i] += 1
            else:
                count[i] = 1
        
        max_label = 0
        name_label = ''

        for key,value in count.items():
            if value > max_label:
                name_label = key
                max_label = value
        return name_label

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        try:
            dis_all = [(self.distance(image, i[0]), i[1]) for i in self.data]
            top_k_dis = sorted(dis_all, key=lambda x:x[0])[:self.k_neighbors]
            label = [i[1] for i in top_k_dis]
        except ValueError:
            raise ValueError()
        
        return self.vote(label)

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label