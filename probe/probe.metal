// Self-contained public synthetic program. No captured data or product code.
#include <metal_stdlib>
using namespace metal;

// All three operands come from runtime device-buffer loads. The source contains
// no input triplets, function constants, or specialized bounds. Echoes independently
// expose the loaded operands. Output rows are operations; columns are input cases.
kernel void probe_f32(device const float4 *input [[buffer(0)]],
                      device uint4 *echo [[buffer(1)]],
                      device uint4 *bits [[buffer(2)]],
                      texture2d<float, access::write> rgba [[texture(0)]],
                      uint i [[thread_position_in_grid]]) {
    const float4 v = input[i];
    const float a = fast::clamp(v.x, v.y, v.z);
    const float b = precise::clamp(v.x, v.y, v.z);
    const float c = fast::fmin(fast::fmax(v.x, v.y), v.z);
    const float d = precise::fmin(precise::fmax(v.x, v.y), v.z);
    echo[i] = uint4(as_type<uint>(v.x), as_type<uint>(v.y), as_type<uint>(v.z), i);
    bits[i] = uint4(as_type<uint>(a), as_type<uint>(b), as_type<uint>(c), as_type<uint>(d));
    rgba.write(float4(a, a, a, 1.0f), uint2(i, 0));
    rgba.write(float4(b, b, b, 1.0f), uint2(i, 1));
    rgba.write(float4(c, c, c, 1.0f), uint2(i, 2));
    rgba.write(float4(d, d, d, 1.0f), uint2(i, 3));
}

// Half is a separate domain. MSL documents explicit fast/precise namespace
// variants for single precision. Do not silently promote half to those float
// overloads: use the half-typed builtins, compiled under both math modes.
kernel void probe_f16(device const half4 *input [[buffer(0)]],
                      device uint4 *echo [[buffer(1)]],
                      device uint4 *bits [[buffer(2)]],
                      texture2d<half, access::write> rgba [[texture(0)]],
                      uint i [[thread_position_in_grid]]) {
    const half4 v = input[i];
    const half a = clamp(v.x, v.y, v.z);
    const half b = fmin(fmax(v.x, v.y), v.z);
    echo[i] = uint4(uint(as_type<ushort>(v.x)), uint(as_type<ushort>(v.y)),
                   uint(as_type<ushort>(v.z)), i);
    // Raw half encodings AND exact f32 widening, not a host-approximated display.
    bits[i] = uint4(uint(as_type<ushort>(a)), uint(as_type<ushort>(b)),
                   as_type<uint>(float(a)), as_type<uint>(float(b)));
    rgba.write(half4(a, a, a, half(1.0f)), uint2(i, 0));
    rgba.write(half4(b, b, b, half(1.0f)), uint2(i, 1));
}
