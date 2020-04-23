// tips of debugging
// You can simply change out value to x or y or any variable you want to know.
// Then plot the return matrix in python and see how the value distributes.

// These parametes will be determined by utils.read_code
// #define FRAME_N ${FRAME_N} // #define H ${BLK_K}
// #define W ${W}
// #define H ${H}
// #define MAG ${MAG}
#define WEIGHT_N 4
#define W_ZOOMED int(W*MAG)
#define i_(a,b,c)  (a)*(H)*(W)+(b)*(W)+(c)                            //indexing for original matrix
#define i_Q_u(a,b,c)  (a)*(W_ZOOMED)*(WEIGHT_N)+(b)*(W_ZOOMED)+(c)    //indexing for Q and u
#define i_out(a,b,c)  (a)*(H)*(W_ZOOMED)+(b)*(H)+(c)                  //indexing for output matrix

extern "C" __global__
void cmlr_zoom_x_T(const float* image_mat, const int* u, const float* Q, const int Q_frame_n, float* out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.z * blockIdx.z + threadIdx.z;
    if (0 <= x && x < W_ZOOMED && 0 <= y && y < H)
    {
        float tmp = 0;

        # pragma unroll
        for (int k = 0; k < WEIGHT_N; ++k)
        {
            if (Q_frame_n == 1){
                int selected_index = u[i_Q_u(0, k, x)];
                float weight = Q[i_Q_u(0, k, x)];
                tmp += image_mat[i_(i, y, selected_index)] * weight;
            }
            else
            {
                int selected_index = u[i_Q_u(i, k, x)];
                float weight = Q[i_Q_u(i, k, x)];
                tmp += image_mat[i_(i, y, selected_index)] * weight;
            }
        }
        out[i_out(i, x, y)] = tmp;
    }
}

