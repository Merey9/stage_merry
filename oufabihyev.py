def separate_parameters(parameters, key_list):
    present_keys = {}
    absent_keys = {}

    for key, value in parameters.items():
        if key in key_list:
            present_keys[key] = value
        else:
            absent_keys[key] = value

    return present_keys, absent_keys


# Example usage:
parameters = {"a": 1, "b": 2, "c": 3, "d": 4}
key_list = ["a", "c"]

present_params, absent_params = separate_parameters(parameters, key_list)

print("Present keys:", present_params)
print("Absent keys:", absent_params)
