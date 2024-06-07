
def construct_state(state, encoding_function):
    matrix_obs = state[0]
    fixed_grid = state[1]
    vector_obs = state[2]
    current_index = (vector_obs[0], vector_obs[2])
    facing = (vector_obs[3], vector_obs[5])
    one_hot = encoding_function(matrix_obs, 7)
    flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
    return flattened_matrix_obs, fixed_grid, current_index, facing
