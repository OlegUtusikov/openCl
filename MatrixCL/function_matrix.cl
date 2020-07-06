// TILE_W == TILE_H == LOCAL_GROUP_SIZE
// firstSizeP % LOCAL_GROUP_SIZE == 0
// firstSizeP % LOCAL_SIZE_GROUP == 0
// thirdSizeP % LOCAL_GROUP_SIZE == 0
kernel void matrix_mul(global const float* first,
                       global const float* second,
                       global float* result,
                       global const int* firstSizeP,
                       global const int* secondSizeP,
                       global const int* thirdSizeP) {
    int firstSize  = *firstSizeP;
    int secondSize = *secondSizeP;
    int thirdSize  = *thirdSizeP;

    size_t locI = get_local_id(0);
    size_t locJ = get_local_id(1);
    int i = get_group_id(0) * LOCAL_GROUP_SIZE + locI;
    int j = get_group_id(1) * LOCAL_GROUP_SIZE + locJ * ELEMENTS;

    local float firstLoc[LOCAL_GROUP_SIZE][LOCAL_GROUP_SIZE];
    local float secondLoc[LOCAL_GROUP_SIZE][LOCAL_GROUP_SIZE];

    float acc[ELEMENTS];
    for (size_t elem = 0; elem < ELEMENTS; ++elem) {
        acc[elem] = 0.0;
    }

    for (size_t shift = 0; shift < secondSize; shift += LOCAL_GROUP_SIZE) {
        for (size_t elem = 0; elem < ELEMENTS; ++elem) {
        size_t indJ = locJ * ELEMENTS + elem;
            firstLoc[locI][indJ]  = first[i * secondSize + shift + locJ * ELEMENTS + elem];
            secondLoc[indJ][locI] = second[(j + elem) * secondSize + shift + locI];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < LOCAL_GROUP_SIZE; ++k) {
            for (size_t elem = 0; elem < ELEMENTS; ++elem) {
                acc[elem] += firstLoc[locI][k] * secondLoc[locJ * ELEMENTS + elem][k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t elem = 0; elem < ELEMENTS; ++elem) {
        result[i * thirdSize + j + elem] = acc[elem];
    }
}