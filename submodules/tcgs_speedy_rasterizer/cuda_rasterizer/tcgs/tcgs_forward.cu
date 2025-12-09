// Copyright (c) 2025 TCGS GROUP. MIT License. See LICENSE for details.
#include "tcgs.h"
#include "tcgs_utils.h"
#include <cuda.h>
#include <cooperative_groups.h>
#include "cuda_runtime.h"

namespace cg = cooperative_groups;
using namespace TCGS_UTIL;

__device__ void identify_pixel_properties(
    cg::thread_block &block, int thread_id, int warp_id,
    int width, int height,
    bool &inside, int &tile_id, int &pix_id,
    float2 &pixf_mid, float2 &pixf_local
)
{
    uint horizontal_blocks = (width + BLOCK_X_TCGS - 1) / BLOCK_X_TCGS;
    uint2 pix_min = make_uint2(block.group_index().x * BLOCK_X_TCGS,
                               block.group_index().y * BLOCK_Y_TCGS);
    uint2 pix_max = make_uint2(min(pix_min.x + BLOCK_X_TCGS, width),
                               min(pix_min.y + BLOCK_Y_TCGS, height));

    uint2 pix = make_uint2(
        pix_min.x + (((warp_id >> 2) << 3) | (thread_id & 7)),
        pix_min.y + ((thread_id & 127) >> 3));

    inside = (pix.x < width) && (pix.y < height);

    tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    pix_id  = width * pix.y + pix.x;

    float pix_min_x_f = __uint2float_rn(pix_min.x);
    float pix_min_y_f = __uint2float_rn(pix_min.y);
    pixf_mid = make_float2(pix_min_x_f + 7.5f, pix_min_y_f + 7.5f);
    pixf_local = make_float2(
        __uint2float_rn(pix.x) - pixf_mid.x,
        __uint2float_rn(pix.y) - pixf_mid.y
    );
}

__forceinline__ __device__ uint4 pix2vec(float2 pixf_local)
{
    uint4 pixel_vector;
    float x2 = pixf_local.x * pixf_local.x;
    float y2 = pixf_local.y * pixf_local.y;
    float xy = pixf_local.x * pixf_local.y;
    const float one_third = 1.0f / 3.0f;

    pixel_vector.x = float22reg(pixf_local.x, pixf_local.y);
    pixel_vector.y = float22reg(x2, y2);
    pixel_vector.z = float22reg(one_third, xy);
    pixel_vector.w = float22reg(one_third, one_third);
    return pixel_vector;
}

__forceinline__ __device__ uint4 gs2vec(
    float4 conic_opacity,
    float2 means,
    float2 pixf_mid
)
{
    means.x = pixf_mid.x - means.x;
    means.y = pixf_mid.y - means.y;

    float mx2 = means.x * means.x;
    float my2 = means.y * means.y;
    float mxy = means.x * means.y;
    float constant =
        conic_opacity.w +
        conic_opacity.x * mx2 +
        conic_opacity.y * mxy +
        conic_opacity.z * my2;

    uint4 gaussian_vector;
    gaussian_vector.x = float22reg(
        2.0f * conic_opacity.x * means.x + conic_opacity.y * means.y,
        2.0f * conic_opacity.z * means.y + conic_opacity.y * means.x
    );
    gaussian_vector.y = float22reg(conic_opacity.x, conic_opacity.z);
    gaussian_vector.z = float22reg(constant, conic_opacity.y);
    gaussian_vector.w = float22reg(constant, constant);
    return gaussian_vector;
}

__forceinline__ __device__ void vec2mat(
    uint4 vecs,
    uint* mat_smem,
    uint thread_id
)
{
    uint addr = (thread_id << 2);
    mat_smem[addr    ] = vecs.x;
    mat_smem[addr | 1] = vecs.y;
    mat_smem[addr | 2] = vecs.z;
    mat_smem[addr | 3] = vecs.w;
}

__forceinline__ __device__ void store_exponent_mat(
    uint reg0, uint reg1, uint reg2, uint reg3,
    uint* mat_smem
)
{
    mat_smem[0 ] = reg0;
    mat_smem[8 ] = reg1;
    mat_smem[16] = reg2;
    mat_smem[24] = reg3;
}

// culling + alpha blending with hoisted constants, no inner T break
__forceinline__ __device__ uint2 culling_and_blending(
    uint* exponent_matrix,     // shared memory
    uint2* channels_smem,
    half &T, int gs_index, int thread_id, uint2 RGBD
)
{
    const half threshold_low  = __float2half_rn(-7.995f);
    const half threshold_high = __float2half_rn(0.0f);
    const half alpha_max      = __float2half_rn(0.99f);

#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        half2 exponents = *reinterpret_cast<half2*>(
            exponent_matrix + ((k << 8) | thread_id)
        );

        bool valid_x = __hgt(exponents.x, threshold_low) &&
                       __hlt(exponents.x, threshold_high);
        bool valid_y = __hgt(exponents.y, threshold_low) &&
                       __hlt(exponents.y, threshold_high);

        if (valid_x)
        {
            half exp_val = fast_ex2_f16(exponents.x);
            half alpha   = __hmul(T, __hmin(alpha_max, exp_val));
            T = __hsub(T, alpha);

            uint2 channel = channels_smem[gs_index | (k << 1)];
            uint alpha2   = half22uint(make_half2(alpha, alpha));

            RGBD.x = fast_fma_rn_ftz_f16x2(channel.x, alpha2, RGBD.x);
            RGBD.y = fast_fma_rn_ftz_f16x2(channel.y, alpha2, RGBD.y);
        }

        if (valid_y)
        {
            half exp_val = fast_ex2_f16(exponents.y);
            half alpha   = __hmul(T, __hmin(alpha_max, exp_val));
            T = __hsub(T, alpha);

            uint2 channel = channels_smem[gs_index | (k << 1) | 1];
            uint alpha2   = half22uint(make_half2(alpha, alpha));

            RGBD.x = fast_fma_rn_ftz_f16x2(channel.x, alpha2, RGBD.x);
            RGBD.y = fast_fma_rn_ftz_f16x2(channel.y, alpha2, RGBD.y);
        }
    }
    return RGBD;
}

__global__ void renderCUDA_TCGS(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int width, int height,
    const float2* __restrict__ points_xy_image,
    const uint2* __restrict__ features,
    const float4* __restrict__ conic_opacity,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ invdepth
)
{
    auto block = cg::this_thread_block();
    int thread_id = block.thread_rank();
    int warp_id   = thread_id / WAPR_SIZE;

    bool inside;
    int pix_id;
    int tile_id;
    float2 pixf_mid;
    float2 pixf_local;

    identify_pixel_properties(
        block, thread_id, warp_id,
        width, height,
        inside, tile_id, pix_id, pixf_mid, pixf_local
    );

    __shared__ uint  multiuse_matrix[BLOCK_SIZE_TCGS * REDUCE_SIZE / 2];
    __shared__ uint  exponent_matrix[BLOCK_SIZE_TCGS * VECTOR_SIZE];
    __shared__ uint2 channels_smem[BLOCK_SIZE_TCGS];

    uint* exponent_matrix_addr =
        exponent_matrix +
        ((thread_id & 3) << 8) +
        (warp_id << 5) +
        ((thread_id & 31) >> 2);

    uint pixmat_reg[4];

    // pixel matrix
    uint4 pix_vec = pix2vec(pixf_local);
    vec2mat(pix_vec, multiuse_matrix, thread_id);
    __syncwarp();
    load_matrix_x4(
        pixmat_reg[0], pixmat_reg[1], pixmat_reg[2], pixmat_reg[3],
        multiuse_matrix + (thread_id << 2)
    );

    bool   done      = !inside;
    bool   warp_done = (__ballot_sync(~0, done) == (~0));
    uint2  range     = ranges[tile_id];
    int    toDo      = range.y - range.x;
    const int rounds = (toDo + BLOCK_SIZE_TCGS - 1) / BLOCK_SIZE_TCGS;

    half T = __float2half_rn(1.0f);
    uint  num_contrib = 0u;
    uint2 RGBD        = make_uint2(0u, 0u);

    for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE_TCGS)
    {
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE_TCGS)
            break;

        int   progress     = ((i << 8) | thread_id);
        int   idx_in_range = range.x + progress;
        uint4 gaussian_vec = make_uint4(0u, 0u, 0u, float22reg(1.0f, 1.0f));
        uint2 channel      = make_uint2(0u, 0u);

        if (idx_in_range < range.y)
        {
            int   coll_id = point_list[idx_in_range];
            float4 conics = conic_opacity[coll_id];
            float2 means  = points_xy_image[coll_id];
            gaussian_vec  = gs2vec(conics, means, pixf_mid);
            channel       = features[coll_id];
        }

        vec2mat(gaussian_vec, multiuse_matrix, thread_id);
        channels_smem[thread_id] = channel;

        block.sync();

        const int  gs_num             = min(BLOCK_SIZE_TCGS, toDo);
        const half T_threshold        = __float2half_rn(0.0001f);
        const int  matrix_base_offset = (thread_id & 31) << 2;

        for (int j = 0; !warp_done && j < gs_num; j += REDUCE_SIZE)
        {
            uint gsmat_reg[2];
            uint* matrix_addr = multiuse_matrix + (j << 2) + matrix_base_offset;
            load_matrix_x2(gsmat_reg[0], gsmat_reg[1], matrix_addr);

            uint expmat_reg_0[4] = {0u, 0u, 0u, 0u};
            mma_16x8x8_f16_f16(
                expmat_reg_0[0], expmat_reg_0[1],
                pixmat_reg[0],   pixmat_reg[1],
                gsmat_reg[0],
                expmat_reg_0[0], expmat_reg_0[1]
            );
            mma_16x8x8_f16_f16(
                expmat_reg_0[2], expmat_reg_0[3],
                pixmat_reg[2],   pixmat_reg[3],
                gsmat_reg[0],
                expmat_reg_0[2], expmat_reg_0[3]
            );
            store_exponent_mat(
                expmat_reg_0[0], expmat_reg_0[1],
                expmat_reg_0[2], expmat_reg_0[3],
                exponent_matrix_addr
            );

            uint expmat_reg_1[4] = {0u, 0u, 0u, 0u};
            mma_16x8x8_f16_f16(
                expmat_reg_1[0], expmat_reg_1[1],
                pixmat_reg[0],   pixmat_reg[1],
                gsmat_reg[1],
                expmat_reg_1[0], expmat_reg_1[1]
            );
            mma_16x8x8_f16_f16(
                expmat_reg_1[2], expmat_reg_1[3],
                pixmat_reg[2],   pixmat_reg[3],
                gsmat_reg[1],
                expmat_reg_1[2], expmat_reg_1[3]
            );
            __syncwarp();
            store_exponent_mat(
                expmat_reg_1[0], expmat_reg_1[1],
                expmat_reg_1[2], expmat_reg_1[3],
                exponent_matrix_addr + 1024
            );
            __syncwarp();

            RGBD = culling_and_blending(
                exponent_matrix, channels_smem, T, j, thread_id, RGBD
            );

            if (__hlt(T, T_threshold))
                done = true;
            if (__ballot_sync(~0, done) == (~0))
                warp_done = true;

            num_contrib += REDUCE_SIZE;
        }
    }

    if (inside)
    {
        float Tf = __half2float(T);
        half2 RGh = uint2half2(RGBD.x);
        half2 BDh = uint2half2(RGBD.y);
        final_T[pix_id] = Tf;

        float Tf_bg0 = Tf * bg_color[0];
        float Tf_bg1 = Tf * bg_color[1];
        float Tf_bg2 = Tf * bg_color[2];
        int   wh     = width * height;

        out_color[pix_id]        = __half2float(RGh.x) + Tf_bg0;
        out_color[pix_id + wh]   = __half2float(RGh.y) + Tf_bg1;
        out_color[pix_id + 2*wh] = __half2float(BDh.x) + Tf_bg2;
        if (invdepth)
            invdepth[pix_id] = __half2float(BDh.y);

        n_contrib[pix_id] = num_contrib;
    }
}

__global__ void transform_coefs(
    const int   P,
    const float* __restrict__ colors,
    const float* __restrict__ depths,
    float4* __restrict__ conic_opacity,
    uint2*  __restrict__ feature_encoded,
    float*  __restrict__ invdepth
)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    const float log2e_n_2 = LOG2E_N_2;
    const float log2e_n   = LOG2E_N;

    float4 conics = conic_opacity[idx];
    float  opacity_log = fast_lg2_f32(conics.w);
    conic_opacity[idx] = make_float4(
        log2e_n_2 * conics.x,
        log2e_n   * conics.y,
        log2e_n_2 * conics.z,
        opacity_log
    );

    float r = colors[idx * 3];
    float g = colors[idx * 3 + 1];
    float b = colors[idx * 3 + 2];
    float d = 0.0f;

    if (invdepth)
        d = __frcp_rn(depths[idx]);

    uint RG = float22reg(r, g);
    uint BD = float22reg(b, d);
    feature_encoded[idx] = make_uint2(RG, BD);
}

// Host API: non-graph path with static feature buffer reuse
void TCGS::renderCUDA_Forward(
    const dim3 grid,
    const dim3 block,
    const uint2* ranges,
    const uint*  point_list,
    int width,
    int height,
    int P,
    const float2* means2D,
    const float*  colors,
    float4*       conic_opacity,
    float*        final_T,
    uint*         n_contrib,
    const float*  bg_color,
    float*        out_color,
    float*        depths,
    float*        depth
)
{
    // Reuse encoded-feature buffer across calls
    static uint2* feature_encoded = nullptr;
    static size_t allocated_size  = 0;

    size_t required_size = static_cast<size_t>(P) * sizeof(uint2);
    if (feature_encoded == nullptr || allocated_size < required_size) {
        if (feature_encoded != nullptr) {
            cudaFree(feature_encoded);
        }
        cudaMalloc(&feature_encoded, required_size);
        allocated_size = required_size;
    }

    transform_coefs<<<(P + 255) / 256, 256>>>(
        P, colors, depths, conic_opacity, feature_encoded, depth
    );

    renderCUDA_TCGS<<<grid, block>>>(
        ranges, point_list,
        width, height,
        means2D, feature_encoded, conic_opacity,
        final_T, n_contrib,
        bg_color, out_color, depth
    );
}

void TCGS::renderCUDA_Forward_Graph(
    TCGSGraph* graphHandle,
    const dim3 grid,
    const dim3 block,
    const uint2* ranges,
    const uint*  point_list,
    int width,
    int height,
    int P,
    const float2* means2D,
    const float*  features,
    float4*       conic_opacity,
    float*        final_T,
    uint*         n_contrib,
    const float*  bg_color,
    float*        out_color,
    float*        depths,
    float*        depth,
    bool captureGraph
)
{
    static uint2* feature_encoded = nullptr;
    static size_t allocated_size  = 0;

    size_t required_size = static_cast<size_t>(P) * sizeof(uint2);
    if (feature_encoded == nullptr || allocated_size < required_size) {
        if (feature_encoded != nullptr) {
            cudaFree(feature_encoded);
        }
        cudaMalloc(&feature_encoded, required_size);
        allocated_size = required_size;
    }

    if (captureGraph || !graphHandle->isInstantiated) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        if (graphHandle->graph != nullptr) {
            cudaGraphDestroy(graphHandle->graph);
        }
        if (graphHandle->graphExec != nullptr) {
            cudaGraphExecDestroy(graphHandle->graphExec);
        }

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        transform_coefs<<<(P + 255) / 256, 256, 0, stream>>>(
            P, (const float*)features, depths, conic_opacity, feature_encoded, depth
        );

        renderCUDA_TCGS<<<grid, block, 0, stream>>>(
            ranges, point_list,
            width, height,
            means2D, feature_encoded, conic_opacity,
            final_T, n_contrib,
            bg_color, out_color, depth
        );

        cudaStreamEndCapture(stream, &graphHandle->graph);

        cudaGraphExecUpdateResult updateResult;
        cudaGraphNode_t errorNode;

        if (graphHandle->graphExec != nullptr) {
            cudaGraphExecUpdate(
                graphHandle->graphExec,
                graphHandle->graph,
                &errorNode, &updateResult
            );
            if (updateResult != cudaGraphExecUpdateSuccess) {
                cudaGraphExecDestroy(graphHandle->graphExec);
                cudaGraphInstantiate(
                    &graphHandle->graphExec,
                    graphHandle->graph,
                    nullptr, nullptr, 0
                );
            }
        } else {
            cudaGraphInstantiate(
                &graphHandle->graphExec,
                graphHandle->graph,
                nullptr, nullptr, 0
            );
        }

        graphHandle->isInstantiated = true;
        cudaStreamDestroy(stream);
    }

    if (graphHandle->isInstantiated && graphHandle->graphExec != nullptr) {
        cudaGraphLaunch(graphHandle->graphExec, cudaStreamLegacy);
    } else {
        transform_coefs<<<(P + 255) / 256, 256>>>(
            P, (const float*)features, depths, conic_opacity, feature_encoded, depth
        );
        renderCUDA_TCGS<<<grid, block>>>(
            ranges, point_list,
            width, height,
            means2D, feature_encoded, conic_opacity,
            final_T, n_contrib,
            bg_color, out_color, depth
        );
    }
}

void TCGS::destroyGraph(TCGSGraph* graphHandle)
{
    if (graphHandle == nullptr) return;

    if (graphHandle->graphExec != nullptr) {
        cudaGraphExecDestroy(graphHandle->graphExec);
        graphHandle->graphExec = nullptr;
    }
    if (graphHandle->graph != nullptr) {
        cudaGraphDestroy(graphHandle->graph);
        graphHandle->graph = nullptr;
    }
    graphHandle->isInstantiated = false;
}

void TCGS::updateGraph(TCGSGraph* graphHandle)
{
    if (graphHandle == nullptr || graphHandle->graph == nullptr) return;

    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;

    if (graphHandle->graphExec != nullptr) {
        cudaGraphExecUpdate(
            graphHandle->graphExec,
            graphHandle->graph,
            &errorNode, &updateResult
        );
        if (updateResult != cudaGraphExecUpdateSuccess) {
            // Recreate if update fails
            cudaGraphExecDestroy(graphHandle->graphExec);
            cudaGraphInstantiate(
                &graphHandle->graphExec,
                graphHandle->graph,
                nullptr, nullptr, 0
            );
        }
    }
}