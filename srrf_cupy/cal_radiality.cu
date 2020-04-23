
// #define W ${W}
// #define H ${H}
// #define SAMPLE_N ${SAMPLE_N}
#define i_(a,b,c)  (a)*(H)*(W)+(b)*(W)+(c)          //indexing for matrix
#define True 1
#define False 0

extern "C" __global__
void cal_radiality(const float* image_mat, const float* gradient_x, const float* gradient_y, float* Rt )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.z * blockIdx.z + threadIdx.z;
    const int sample_pts_rel_xy[] = SAMPLE_PTS_REL_XY;
    if (0 <= y && y < H && 0 <= x && x < W)
    {
        float Rt_val = 0.;
        float Gw = 0.;
        int j;
        # pragma unroll
        for (j = 0; j < SAMPLE_N; ++j)
        {
            int rel_x = sample_pts_rel_xy[j*2];
            int rel_y = sample_pts_rel_xy[j*2+1];
            if (0 <= y+rel_y && y+rel_y < H && 0 <= x+rel_x && x+rel_x < W)
            {
                float rel_gx = gradient_x[i_(i, y+rel_y, x+rel_x)];
                float rel_gy = gradient_y[i_(i, y+rel_y, x+rel_x)];
                float r_abs_gi = rhypotf(rel_gx, rel_gy);
                if (isinf(r_abs_gi)) continue; // divide 0 case, skiip this loop
                float r_R = rhypotf(rel_x, rel_y);
                float temp0 = rel_gx * rel_x + rel_gy * rel_y;
                float di = abs(rel_x * rel_gy - rel_y * rel_gx) * r_abs_gi;
                float temp1 = 1.0  - di * r_R;
                Rt_val += copysignf(temp1 * temp1, temp0);
                #if DO_GW
                    float abs_gc = hypotf(gradient_x[i_(i, y, x)], gradient_y[i_(i, y, x)]);
                    float temp2 = (1./r_abs_gi - abs_gc) / image_mat[i_(i, y, x)];
                    if (isinf(temp2)) continue; // divide-0 case, skip this loop
                    Gw += copysignf(temp2, temp0);
                #endif
            }
        }
        Rt_val /= SAMPLE_N * RADIUS;
        #if DO_GW
            Gw /= SAMPLE_N;
            Rt_val *= Gw;
        #endif
        #if DO_IW
            Rt_val *= image_mat[i_(i, y, x)];
        #endif
        Rt[i_(i, y, x)] = max(Rt_val, 0.0);
    }
}