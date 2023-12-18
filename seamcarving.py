import numpy as np
import cv2
import os
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class SeamCarver:
    def __init__(self, filename, out_height, out_width, protect_mask='', object_mask=''):
        """
        Initializes the SeamCarving object.

        Parameters:
        - filename (str): The path to the input image file.
        - out_height (int): The desired height of the output image.
        - out_width (int): The desired width of the output image.
        - protect_mask (str, optional): The path to the protect mask image file. Defaults to an empty string.
        - object_mask (str, optional): The path to the object mask image file. Defaults to an empty string.

        Attributes:
        - filename (str): The path to the input image file.
        - out_height (int): The desired height of the output image.
        - out_width (int): The desired width of the output image.
        - in_image (ndarray): The input image stored as np.float64 format.
        - in_height (int): The height of the input image.
        - in_width (int): The width of the input image.
        - out_image (ndarray): The resulting image, initially a copy of the input image.
        - object (bool): Indicates whether object removal is enabled.
        - mask (ndarray): The mask image stored as np.float64 format in gray scale.
        - protect (bool): Indicates whether protect mask is enabled.
        - kernel_x (ndarray): The kernel for forward energy map calculation in the x-direction.
        - kernel_y_left (ndarray): The kernel for forward energy map calculation in the y-direction (left).
        - kernel_y_right (ndarray): The kernel for forward energy map calculation in the y-direction (right).
        - constant (int): The constant value for covered area by protect mask or object mask.

        """
        # initialize parameter
        self.filename = filename
        self.out_height = out_height
        self.out_width = out_width

        # read in image and store as np.float64 format
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        # object removal --> self.object = True
        self.object = (object_mask != '')
        if self.object:
            # read in object mask image file as np.float64 format in gray scale
            self.mask = cv2.imread(object_mask, 0).astype(np.float64)
            self.protect = False
        # image re-sizing with or without protect mask
        else:
            self.protect = (protect_mask != '')
            if self.protect:
                # if protect_mask filename is provided, read in protect mask image file as np.float64 format in gray scale
                self.mask = cv2.imread(protect_mask, 0).astype(np.float64)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # constant for covered area by protect mask or object mask
        self.constant = 1000

        # starting program
        self.start()


    def start(self):
        """
        :return:

        If object mask is provided --> object removal function will be executed
        else --> seam carving function (image retargeting) will be process
        """
        if self.object:
            self.object_removal()
        else:
            self.seams_carving()


    def seams_carving(self):
        """
        :return:

        We first process seam insertion or removal in vertical direction then followed by horizontal direction.

        If targeting height or width is greater than original ones --> seam insertion,
        else --> seam removal

        The algorithm is written for seam processing in vertical direction (column), so image is rotated 90 degree
        counter-clockwise for seam processing in horizontal direction (row)
        """

        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)


        """
        :return:

        Object covered by mask will be removed first and seam will be inserted to return to original image dimension
        """

    def object_removal(self):

        rotate = False
        object_height, object_width = self.get_object_dimension()
        if object_height < object_width:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.mask = self.rotate_mask(self.mask, 1)
            rotate = True

        while len(np.where(self.mask[:, :] > 0)[0]) > 0:
            energy_map = self.calc_energy_map()
            energy_map[np.where(self.mask[:, :] > 0)] *= -self.constant
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)
            self.delete_seam_on_mask(seam_idx)

        if not rotate:
            num_pixels = self.in_width - self.out_image.shape[1]
        else:
            num_pixels = self.in_height - self.out_image.shape[1]

        self.seams_insertion(num_pixels)
        if rotate:
            self.out_image = self.rotate_image(self.out_image, 0)


    def seams_removal(self, num_pixel):
        if self.protect:
            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask > 0)] *= self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)
        else:
            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)


    def seams_insertion(self, num_pixel):
        if self.protect:
            temp_image = np.copy(self.out_image)
            temp_mask = np.copy(self.mask)
            seams_record = []

            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask[:, :] > 0)] *= self.constant
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)

            self.out_image = np.copy(temp_image)
            self.mask = np.copy(temp_mask)
            n = len(seams_record)
            for dummy in range(n):
                seam = seams_record.pop(0)
                self.add_seam(seam)
                self.add_seam_on_mask(seam)
                seams_record = self.update_seams(seams_record, seam)
        else:
            temp_image = np.copy(self.out_image)
            seams_record = []

            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)

            self.out_image = np.copy(temp_image)
            n = len(seams_record)
            for dummy in range(n):
                seam = seams_record.pop(0)
                self.add_seam(seam)
                seams_record = self.update_seams(seams_record, seam)


    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        
        # b_energy_scharr = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        # g_energy_scharr = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        # r_energy_scharr = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        
        # b_energy_sobel = np.absolute(cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=5)) + np.absolute(cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=5))
        # g_energy_sobel = np.absolute(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)) + np.absolute(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5))
        # r_energy_sobel = np.absolute(cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=5)) + np.absolute(cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=5))
                
        # canny edge detection
        # b = cv2.convertScaleAbs(b)
        # g = cv2.convertScaleAbs(g)
        # r = cv2.convertScaleAbs(r)

        # b_energy_canny = cv2.Canny(b, 100, 200)
        # g_energy_canny = cv2.Canny(g, 100, 200)
        # r_energy_canny = cv2.Canny(r, 100, 200)
        
        # Prewitt Filter
        b_energy_prewitt = np.absolute(cv2.filter2D(b, -1, kernel=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))) + \
                            np.absolute(cv2.filter2D(b, -1, kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
        g_energy_prewitt = np.absolute(cv2.filter2D(g, -1, kernel=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))) + \
                            np.absolute(cv2.filter2D(g, -1, kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
        r_energy_prewitt = np.absolute(cv2.filter2D(r, -1, kernel=np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))) + \
                            np.absolute(cv2.filter2D(r, -1, kernel=np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])))
                            

    
        return b_energy_prewitt + g_energy_prewitt + r_energy_prewitt


    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                output[row, col] = \
                    energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return output


    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output


    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output
    


    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)


    def add_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.out_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)


    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output


    def rotate_image(self, image, ccw):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return output

    def get_object_dimension(self):
        rows, cols = np.where(self.mask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width


    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
        

def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)


if __name__ == '__main__':
    """
    Put image in Image folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """

    folder_in = 'Image'
    folder_out = 'Image'

    filename_input = 'tower.jpg'
    filename_output = 'tower_result_prewitt.jpg'
    filename_mask = ''

    input_image = os.path.join(folder_in, filename_input)
    input_mask = os.path.join(folder_in, filename_mask)
    output_image = os.path.join(folder_out, filename_output)
    
    k = (cv2.imread(input_image))
    print(k.shape)
    
    new_height = int(input())
    new_width = int(input())

    image_resize_without_mask(input_image, output_image, new_height, new_width)
