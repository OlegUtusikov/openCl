kernel void tiles_pref_sums(global const float* elements, global float* res, global const size_t* size) {
    size_t loc_id = get_local_id(0);
    size_t id = get_global_id(0);
    local float loc[LOCAL_GROUP_SIZE];
    loc[loc_id] = elements[id];
    for (size_t shift = 1; shift < LOCAL_GROUP_SIZE; shift *= 2) {
        float add = 0.0;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (shift <= loc_id) {
            add = loc[loc_id - shift];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        loc[loc_id] += add;
    }
    res[id] = loc[loc_id];
}

kernel void add_sums(global const float* sums, global float* res, global const size_t* size) {
    size_t id = get_global_id(0);
    float add = 0.0;
    for (size_t i = LOCAL_GROUP_SIZE - 1; i < id; i += LOCAL_GROUP_SIZE) {
        add += sums[i];
    }
    res[id] = sums[id] + add;
}