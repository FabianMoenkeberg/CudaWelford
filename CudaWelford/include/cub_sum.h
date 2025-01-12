 #pragma once
 
struct point {
    float M;
    float T;
    float N;
};

struct CustomSum
{
    __device__ __forceinline__
    point operator()(const point &a, const point &b) const {
        point res{0, a.T + b.T, a.N + b.N};
        return res;
    }
};

int cubCustomSum();