import ray

if __name__ == '__main__':
    ray_initialisation = False
    if not ray.is_initialized():
        ray.init()
        ray_initialisation = True

    # Get the current available resources
    available_resources = ray.available_resources()
    print("Available resources for Ray:")
    print(str(available_resources))
    print()

    # # Get the current cluster resources usage
    # cluster_resources = ray.cluster_resources()
    # print("Current cluster resources usage:")
    # print(str(cluster_resources))

    if ray_initialisation:
        ray.shutdown()
