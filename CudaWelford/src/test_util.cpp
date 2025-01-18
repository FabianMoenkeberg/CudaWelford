// #include "../include/test_util.h"

// template<>
// __noinline__ bool IsNaN<float>(float val)
// {
//   return std::isnan(val);
// }

// template<>
// __noinline__ bool IsNaN<float1>(float1 val)
// {
//     return (IsNaN(val.x));
// }

// template<>
// __noinline__ bool IsNaN<float2>(float2 val)
// {
//     return (IsNaN(val.y) || IsNaN(val.x));
// }

// template<>
// __noinline__ bool IsNaN<float3>(float3 val)
// {
//     return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
// }

// template<>
// __noinline__ bool IsNaN<float4>(float4 val)
// {
//     return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
// }

// template<>
// __noinline__ bool IsNaN<double>(double val)
// {
//   return std::isnan(val);
// }

// template<>
// __noinline__ bool IsNaN<double1>(double1 val)
// {
//     return (IsNaN(val.x));
// }

// template<>
// __noinline__ bool IsNaN<double2>(double2 val)
// {
//     return (IsNaN(val.y) || IsNaN(val.x));
// }

// template<>
// __noinline__ bool IsNaN<double3>(double3 val)
// {
//     return (IsNaN(val.z) || IsNaN(val.y) || IsNaN(val.x));
// }

// template<>
// __noinline__ bool IsNaN<double4>(double4 val)
// {
//     return (IsNaN(val.y) || IsNaN(val.x) || IsNaN(val.w) || IsNaN(val.z));
// }


// template<>
// __noinline__ bool IsNaN<half_t>(half_t val)
// {
//     const auto bits = SafeBitCast<unsigned short>(val);

//     // commented bit is always true, leaving for documentation:
//     return (((bits >= 0x7C01) && (bits <= 0x7FFF)) ||
//         ((bits >= 0xFC01) /*&& (bits <= 0xFFFFFFFF)*/));
// }