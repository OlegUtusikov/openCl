kernel void prefix_sum(global const int* elements, global int* res, global const int* size) {
    size_t N = *size;
    local int loc[MAX_ARRAY_SIZE];
    size_t id = get_group_id(0) * LOCAL_GROUP_SIZE + get_local_id(0);
    size_t offset = 1;
    loc[2 * id] = elements[2 * id];
    loc[2 * id + 1] = elements[2 * id + 1];
    for (size_t d = N / 2; d > 0; d /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id < d) {
            size_t i = offset * (2 * id + 1) - 1;
            size_t j = offset * (2 * id + 2) - 1;
            loc[j] += loc[i];
        }
        offset *= 2;
    }
    if (id == 0) {
        loc[N - 1] = elements[id];
    }
    for (size_t d = 1; d < N; d *= 2) {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id < d) {
            size_t i = offset * (2 * id + 1) - 1;
            size_t j = offset * (2 * id + 2) - 1;
            int tmp = loc[i];
            loc[i] = loc[j];
            loc[j] += tmp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    res[2 * id] = loc[2 * id];
    res[2 * id + 1] = loc[2 * id + 1];
}