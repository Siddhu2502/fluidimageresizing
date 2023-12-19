// This function converts an RGB image to grayscale.
__global__ void bgr2gray(float *g_odata, float *g_idata, int width, int height) {
    // printf("%d, %d\\n", width, height);
    int des_x = blockIdx.x * blockDim.x + threadIdx.x;
    int des_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (des_x >= width || des_y >= height)
        return;
    int des_id = des_y * width + des_x;
    int src_r_id = des_id * 3;
    g_odata[des_id] = 0.299 * g_idata[src_r_id] + 0.587 * g_idata[src_r_id+1] + 0.114 * g_idata[src_r_id+2];
}


// This function calculates the gradient magnitude of each pixel in an image using the Sobel operator.
__global__ void sobel_abs(float *g_odata, float *g_idata, int width, int height) {
    int des_x = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the x-coordinate of the current thread
    int des_y = blockIdx.y * blockDim.y + threadIdx.y; // Calculate the y-coordinate of the current thread
    if (des_x >= width || des_y >= height) // Check if the current thread is outside the image boundaries
        return;

    int index = des_y * width + des_x; // Calculate the index of the current pixel in the image

    float value_x = 0; // Initialize the x-component of the gradient value
    float value_y = 0; // Initialize the y-component of the gradient value

    if (des_x == 0 || des_x == width - 1) {
        value_x = 0; // Set the x-component of the gradient value to 0 if the current pixel is at the left or right edge of the image
    }
    else {
        // Calculate the x-component of the gradient value using the Sobel operator
        value_x = -2 * g_idata[index - 1] + 2 * g_idata[index + 1];
        if (des_y != 0) {
            // g_idata represents the input image and then we operate on the index - width pixel to get the pixel above the current pixel
            value_x += -1 * g_idata[index - width - 1] + 1 * g_idata[index - width + 1];
        }
        if (des_y != height - 1) {
            value_x += -1 * g_idata[index + width - 1] + 1 * g_idata[index + width + 1];
        }
    }

    if (des_y == 0 || des_y == height - 1) {
        value_y = 0; // Set the y-component of the gradient value to 0 if the current pixel is at the top or bottom edge of the image
    }
    else {
        // Calculate the y-component of the gradient value using the Sobel operator
        value_y = -2 * g_idata[index - width] + 2 * g_idata[index + width];
        if (des_x != 0) {
            value_y += -1 * g_idata[index - width - 1] + 1 * g_idata[index + width - 1];
        }
        if (des_x != width - 1) {
            value_y += -1 * g_idata[index - width + 1] + 1 * g_idata[index + width + 1];
        }
    }

    // Calculate the absolute value of the gradient magnitude and store it in the output array
    g_odata[index] = fabsf(value_x) + fabsf(value_y);
}


// This function finds the index of the minimum value in an array.
__device__ int arg_min(float *arr, int size) {
    int min_offset = 0;
    float min_val = arr[0];
    for (int i = 1; i < size; i++) {
        // Check if the current element is smaller than the current minimum value
        if (arr[i] < min_val) {
            min_offset = i;
            min_val = arr[i];
        }
    }
    // Return the index of the minimum value
    return min_offset;
}

// This function calculates the index of an element in a 1D array given its column and row indices.
__device__ int get_array_index(int col, int row, int width) {
    return row * width + col;
}


// This function calculates the minimum energy value and its corresponding index in the energy matrix.
__global__ void min_energy_at_row(float *energy_m, int *backtrack_m, int width, int row) {
    // Calculate the column index for the current thread
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= width) {
        return;
    }

    int shift_col;
    // Determine the shift value for the column index
    if (col == 0) {
        shift_col = 0;
    } else if (col == width - 1) {
        shift_col = -2;
    } else {
        shift_col = -1;
    }

    // Calculate the index of the previous row's minimum energy value
    int head = get_array_index(col + shift_col, row - 1, width);

    // Find the offset of the minimum energy value in the previous row
    int min_offset = arg_min(energy_m + head, 3);

    // Calculate the column index of the minimum energy value in the previous row
    int min_col = col + shift_col + min_offset;

    // Calculate the indices for the current and minimum energy values
    int min_index = get_array_index(min_col, row - 1, width);
    int current_index = get_array_index(col, row, width);

    // Update the energy value at the current index by adding the minimum energy value from the previous row
    energy_m[current_index] += energy_m[min_index];

    // Store the column index of the minimum energy value in the backtrack matrix
    backtrack_m[current_index] = min_col;
}


/**
 * @file carve.cu
 * @brief CUDA kernel to calculate the minimum energy value and its corresponding index in the energy matrix.
 *
 * This file contains the CUDA kernel function that calculates the minimum energy value and its corresponding index in the energy matrix.
 * The energy matrix represents the energy values of each pixel in an image, which is used for image resizing.
 * The kernel function is designed to be executed on a CUDA-enabled GPU device.
 * The energy matrix is a 2D array represented by a 1D array in row-major order.
 *
 * @param energy_m Pointer to the energy matrix.
 * @param index Pointer to store the index of the minimum energy value.
 * @param width Width of the energy matrix.
 * @param height Height of the energy matrix.
 */
__global__ void get_min_index(float *energy_m, int *index, int width, int height) {
    int offset = width * (height - 1);
    *index = arg_min(energy_m + offset, width);
}


/**
 * @brief Applies a mask to the energy map by a given factor.
 * 
 * This CUDA kernel function adds a mask to the energy map by multiplying each element of the mask
 * with the corresponding element of the energy map and scaling it by a given factor.
 * 
 * @param energy_m Pointer to the energy map array.
 * @param mask Pointer to the mask array.
 * @param factor The scaling factor to apply to the mask.
 * @param width The width of the energy map.
 * @param height The height of the energy map.
 */

__global__ void add_mask_by_factor(float *energy_m, float *mask, float factor, int width, int height) {
    // Calculate the destination coordinates in the energy map
    int des_x = blockIdx.x * blockDim.x + threadIdx.x;
    int des_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if the destination coordinates are within the energy map boundaries
    if (des_x >= width || des_y >= height)
        return;
    
    // Calculate the destination index in the energy map
    int des_id = des_y * width + des_x;
    
    // Apply the mask to the energy map by multiplying with the factor
    energy_m[des_id] += mask[des_id] * factor;
}