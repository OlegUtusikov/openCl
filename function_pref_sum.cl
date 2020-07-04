kernel void prefix_sum(global const float* elements, global float* res, global const size_t* size) {
    size_t N = *size;
    local float loc[MAX_ARRAY_SIZE];
    size_t id = get_group_id(0) * LOCAL_GROUP_SIZE + get_local_id(0);
    loc[id] = elements[id];
    size_t offset = 1;
    for (size_t d = N >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id < d) {
            size_t i = offset * (2 * id + 1) - 1;
            size_t j = offset * (2 * id + 2) - 1;
            loc[j] += loc[i];
        }
        offset <<= 1;
    }
    if (id == 0) {
        loc[offset - 1] = 0;
    }
    for (size_t d = 1; d < N; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id < d) {
            size_t i = offset * (2 * id + 1) - 1;
            size_t j = offset * (2 * id + 2) - 1;
            float tmp = loc[i];
            loc[i] = loc[j];
            loc[j] += tmp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res[id] = loc[id + 1];
}