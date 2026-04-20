#include "cat_pack.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#define CPU_HAS_SIMD 1
#include <arm_neon.h>
#endif

int cat_pack_forward(const uint8_t *src, uint8_t *dest, int32_t length, uint8_t meta, blosc2_cparams *cparams, uint8_t id) {
    BLOSC_UNUSED_PARAM(cparams);
    BLOSC_UNUSED_PARAM(id);

    uint8_t bits_per_val = meta;
    if (bits_per_val != 1 && bits_per_val != 2 && bits_per_val != 4) {
        // No-op for 8 or invalid meta
        memcpy(dest, src, length);
        return BLOSC2_ERROR_SUCCESS;
    }

    if (length == 0) return BLOSC2_ERROR_SUCCESS;

    // 1. Find unique values dynamically for this chunk
    uint8_t counts[256] = {0};
    for (int32_t i = 0; i < length; ++i) {
        counts[src[i]] = 1;
    }
    
    uint8_t unique_vals[256];
    int num_classes = 0;
    for (int i = 0; i < 256; ++i) {
        if (counts[i]) {
            unique_vals[num_classes++] = i;
        }
    }
    
    // Check if we exceed expected capacity for the chosen bit-width
    if ((bits_per_val == 1 && num_classes > 2) ||
        (bits_per_val == 2 && num_classes > 4) ||
        (bits_per_val == 4 && num_classes > 16)) {
        // Fallback: we cannot pack this chunk because there are too many unique values!
        // To be safe, we must copy it. But how do we flag this? 
        // We set num_classes = 0 as a flag! 
        // Wait, if num_classes=0, the first byte of dest is 0. 
        // This is a 1-byte header, but we still need to store all original length data.
        // But `dest` is exactly `length` bytes! We can't prepend anything if we don't compress.
        // Actually, if we just copy, we would exceed dest by 1 byte.
        // Therefore, the user MUST ensure the entire dataset respects the max classes!
        // For robustness, if it fails, we just return an error so the pipeline fails cleanly.
        BLOSC_TRACE_ERROR("Too many unique values (%d) for bits_per_val (%d)\n", num_classes, bits_per_val);
        return BLOSC2_ERROR_FAILURE;
    }

    uint8_t val_to_idx[256] = {0};
    for (int i = 0; i < num_classes; ++i) {
        val_to_idx[unique_vals[i]] = i;
    }
    
    dest[0] = num_classes;
    memcpy(dest + 1, unique_vals, num_classes);
    
    uint8_t *out = dest + 1 + num_classes;
    int32_t out_len = 0;
    int32_t i = 0;

#if defined(CPU_HAS_SIMD)
    for (; i <= length - 16; i += 16) {
        uint8x16_t v_src = vld1q_u8(src + i);
        uint8x16_t v_idx = vdupq_n_u8(0);
        
        for (uint8_t c = 1; c < num_classes; ++c) {
            uint8x16_t v_class = vdupq_n_u8(unique_vals[c]);
            uint8x16_t v_mask = vceqq_u8(v_src, v_class);
            uint8x16_t v_c = vdupq_n_u8(c);
            v_idx = vorrq_u8(v_idx, vandq_u8(v_mask, v_c));
        }
        
        if (bits_per_val == 4) {
            uint8x8_t even = vget_low_u8(vuzp1q_u8(v_idx, v_idx));
            uint8x8_t odd  = vget_low_u8(vuzp2q_u8(v_idx, v_idx));
            uint8x8_t packed = vorr_u8(vshl_n_u8(even, 4), odd);
            vst1_u8(out + out_len, packed);
            out_len += 8;
        } else if (bits_per_val == 2) {
            uint8x8_t even = vget_low_u8(vuzp1q_u8(v_idx, v_idx));
            uint8x8_t odd  = vget_low_u8(vuzp2q_u8(v_idx, v_idx));
            uint8x8_t combined2 = vorr_u8(vshl_n_u8(even, 2), odd);
            
            uint8x8_t ev2 = vuzp1_u8(combined2, combined2);
            uint8x8_t od2 = vuzp2_u8(combined2, combined2);
            uint8x8_t packed = vorr_u8(vshl_n_u8(ev2, 4), od2);
            
            uint32_t packed32 = vget_lane_u32(vreinterpret_u32_u8(packed), 0);
            memcpy(out + out_len, &packed32, 4);
            out_len += 4;
        } else if (bits_per_val == 1) {
            uint8x8_t even = vget_low_u8(vuzp1q_u8(v_idx, v_idx));
            uint8x8_t odd  = vget_low_u8(vuzp2q_u8(v_idx, v_idx));
            uint8x8_t comb1 = vorr_u8(vshl_n_u8(even, 1), odd);
            
            uint8x8_t ev2 = vuzp1_u8(comb1, comb1);
            uint8x8_t od2 = vuzp2_u8(comb1, comb1);
            uint8x8_t comb2 = vorr_u8(vshl_n_u8(ev2, 2), od2);
            
            uint8x8_t ev3 = vuzp1_u8(comb2, comb2);
            uint8x8_t od3 = vuzp2_u8(comb2, comb2);
            uint8x8_t comb3 = vorr_u8(vshl_n_u8(ev3, 4), od3);
            
            uint16_t packed16 = vget_lane_u16(vreinterpret_u16_u8(comb3), 0);
            memcpy(out + out_len, &packed16, 2);
            out_len += 2;
        }
    }
#endif

    // Scalar fallback/tail processing
    uint8_t current_byte = 0;
    uint8_t bits_filled = 0;
    
    for (; i < length; ++i) {
        uint8_t idx = val_to_idx[src[i]];
        current_byte |= (idx << (8 - bits_filled - bits_per_val));
        bits_filled += bits_per_val;
        if (bits_filled >= 8) {
            out[out_len++] = current_byte;
            current_byte = 0;
            bits_filled = 0;
        }
    }
    
    if (bits_filled > 0) {
        out[out_len++] = current_byte;
    }
    
    int32_t total_written = 1 + num_classes + out_len;
    // Zero-fill the remainder to optimize compression
    if (total_written < length) {
        memset(dest + total_written, 0, length - total_written);
    }
    
    return BLOSC2_ERROR_SUCCESS;
}

int cat_pack_backward(const uint8_t *src, uint8_t *dest, int32_t length, uint8_t meta, blosc2_dparams *dparams, uint8_t id) {
    BLOSC_UNUSED_PARAM(dparams);
    BLOSC_UNUSED_PARAM(id);

    uint8_t bits_per_val = meta;
    if (bits_per_val != 1 && bits_per_val != 2 && bits_per_val != 4) {
        memcpy(dest, src, length);
        return BLOSC2_ERROR_SUCCESS;
    }

    if (length == 0) return BLOSC2_ERROR_SUCCESS;

    uint8_t num_classes = src[0];
    if (num_classes == 0) {
        // Unexpected state
        return BLOSC2_ERROR_FAILURE;
    }
    
    const uint8_t *unique_vals = src + 1;
    const uint8_t *packed = src + 1 + num_classes;
    
    int32_t dest_len = 0;
    int32_t i = 0;

#if defined(CPU_HAS_SIMD)
    uint8_t table_data[16] = {0};
    memcpy(table_data, unique_vals, num_classes);
    uint8x16_t v_lut = vld1q_u8(table_data);
    
    if (bits_per_val == 4) {
        int8_t shifts_data4[16] = {-4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0, -4, 0};
        int8x16_t shifts4 = vld1q_s8(shifts_data4);
        uint8x16_t mask4 = vdupq_n_u8(0x0F);
        
        for (; dest_len <= length - 16; i += 8) {
            uint8x8_t p = vld1_u8(packed + i);
            uint8x8x2_t z = vzip_u8(p, p);
            uint8x16_t p_dup = vcombine_u8(z.val[0], z.val[1]);
            uint8x16_t shifted = vshlq_u8(p_dup, shifts4);
            uint8x16_t idx = vandq_u8(shifted, mask4);
            uint8x16_t mapped = vqtbl1q_u8(v_lut, idx);
            vst1q_u8(dest + dest_len, mapped);
            dest_len += 16;
        }
    } else if (bits_per_val == 2) {
        int8_t shifts_data2[16] = {-6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0};
        int8x16_t shifts2 = vld1q_s8(shifts_data2);
        uint8x16_t mask2 = vdupq_n_u8(0x03);
        
        for (; dest_len <= length - 16; i += 4) {
            uint32_t p_val;
            memcpy(&p_val, packed + i, 4);
            uint8x8_t p = vreinterpret_u8_u32(vdup_n_u32(p_val));
            uint8x8_t z1 = vzip1_u8(p, p);
            uint8x8_t d_low  = vzip1_u8(z1, z1);
            uint8x8_t d_high = vzip2_u8(z1, z1);
            uint8x16_t p_dup = vcombine_u8(d_low, d_high);
            uint8x16_t shifted = vshlq_u8(p_dup, shifts2);
            uint8x16_t idx = vandq_u8(shifted, mask2);
            uint8x16_t mapped = vqtbl1q_u8(v_lut, idx);
            vst1q_u8(dest + dest_len, mapped);
            dest_len += 16;
        }
    } else if (bits_per_val == 1) {
        int8_t shifts_data1[16] = {-7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0};
        int8x16_t shifts1 = vld1q_s8(shifts_data1);
        uint8x16_t mask1 = vdupq_n_u8(0x01);
        
        for (; dest_len <= length - 16; i += 2) {
            uint16_t p_val;
            memcpy(&p_val, packed + i, 2);
            uint8x8_t p = vreinterpret_u8_u16(vdup_n_u16(p_val));
            uint8x16_t p_dup = vcombine_u8(vdup_lane_u8(p, 0), vdup_lane_u8(p, 1));
            uint8x16_t shifted = vshlq_u8(p_dup, shifts1);
            uint8x16_t idx = vandq_u8(shifted, mask1);
            uint8x16_t mapped = vqtbl1q_u8(v_lut, idx);
            vst1q_u8(dest + dest_len, mapped);
            dest_len += 16;
        }
    }
#endif

    // Fallback unpack scalar for remainder
    uint8_t current_byte = packed[i];
    uint8_t bits_read = 0;
    
    for (; dest_len < length; ++dest_len) {
        uint8_t idx = (current_byte >> (8 - bits_read - bits_per_val)) & ((1 << bits_per_val) - 1);
        dest[dest_len] = unique_vals[idx];
        bits_read += bits_per_val;
        if (bits_read >= 8) {
            i++;
            current_byte = packed[i];
            bits_read = 0;
        }
    }
    
    return BLOSC2_ERROR_SUCCESS;
}
