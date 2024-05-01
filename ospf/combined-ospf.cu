#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

#define MAX_ROUTERS 1300
#define BLOCK_SIZE 256

// Structure to represent a router
typedef struct {
    int id;
    bool visited;
    int distance;
    int parent;
} Router;

// Function to generate a random network topology with N routers
void generateNetwork(int N, int graph[MAX_ROUTERS][MAX_ROUTERS]) {
    // Randomly assign connections between routers
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            // Simulating random connections
            if (rand() % 2 == 0) {
                int weight = rand() % 10 + 1; // Random weight for the connection
                graph[i][j] = weight;
                graph[j][i] = weight; // Assuming undirected connections
            } else {
                graph[i][j] = 0; // No connection
                graph[j][i] = 0;
            }
        }
    }
}

// Kernel function to find the shortest path using Dijkstra algorithm
__global__ void dijkstra_kernel(int N, int* graph, Router* routers, int src) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        // Find the router with the minimum distance
        int minDistance = INT_MAX;
        int minIndex = -1;
        for (int j = 0; j < N; j++) {
            if (!routers[j].visited && routers[j].distance < minDistance) {
                minDistance = routers[j].distance;
                minIndex = j;
            }
        }

        // Mark the router as visited
        routers[minIndex].visited = true;

        // Update distances of adjacent routers
        for (int j = 0; j < N; j++) {
            int weight = graph[minIndex * N + j];
            if (weight && !routers[j].visited && routers[minIndex].distance + weight < routers[j].distance) {
                routers[j].distance = routers[minIndex].distance + weight;
                routers[j].parent = minIndex;
            }
        }
    }
}

void dijkstra(int N, int graph[MAX_ROUTERS][MAX_ROUTERS], Router routers[MAX_ROUTERS], int source) {
    // Initialize routers
    for (int i = 0; i < N; i++) {
        routers[i].id = i;
        routers[i].visited = false;
        routers[i].distance = INT_MAX;
        routers[i].parent = -1;
    }

    // Set distance to source router as 0
    routers[source].distance = 0;

    // Iterate through all routers
    for (int i = 0; i < N; i++) {
        // Find the router with the minimum distance
        int minDistance = INT_MAX;
        int minIndex = -1;
        for (int j = 0; j < N; j++) {
            if (!routers[j].visited && routers[j].distance < minDistance) {
                minDistance = routers[j].distance;
                minIndex = j;
            }
        }

        // Mark the router as visited
        routers[minIndex].visited = true;

        // Update distances of adjacent routers
        for (int j = 0; j < N; j++) {
            if (graph[minIndex][j] && !routers[j].visited && routers[minIndex].distance + graph[minIndex][j] < routers[j].distance) {
                routers[j].distance = routers[minIndex].distance + graph[minIndex][j];
                routers[j].parent = minIndex;
            }
        }
    }
}

int main() {
    int N;
    printf("Enter the number of routers in the network (up to %d): ", MAX_ROUTERS);
    scanf("%d", &N);

    if (N <= 0 || N > MAX_ROUTERS) {
        printf("Invalid number of routers.\n");
        return 1;
    }

    int* d_graph;
    cudaMalloc(&d_graph, MAX_ROUTERS * MAX_ROUTERS * sizeof(int));

    Router* d_routers;
    cudaMalloc(&d_routers, N * sizeof(Router));

    int graph[MAX_ROUTERS][MAX_ROUTERS] = {0};
    Router routers[MAX_ROUTERS];

    // Seed random number generator
    srand(time(NULL));

    // Generate network topology
    generateNetwork(N, graph);

    // Copy graph to device
    cudaMemcpy(d_graph, graph, MAX_ROUTERS * MAX_ROUTERS * sizeof(int), cudaMemcpyHostToDevice);

    // Print network topology (optional)
    printf("Network Topology:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // Run OSPF (Dijkstra) kernel
    printf("Running Parallel OSPF...\n");
    for (int i = 0; i < N; i++) {
        // Initialize routers on device
        for (int j = 0; j < N; j++) {
            routers[j].id = j;
            routers[j].visited = false;
            routers[j].distance = INT_MAX;
            routers[j].parent = -1;
        }
        routers[i].distance = 0; // Set distance to source router as 0

        cudaMemcpy(d_routers, routers, N * sizeof(Router), cudaMemcpyHostToDevice);

        // Launch kernel
        dijkstra_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(N, d_graph, d_routers, i);

        // Copy results back to host
        cudaMemcpy(routers, d_routers, N * sizeof(Router), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken: %.8f seconds\n", cpu_time_used);

    cudaFree(d_graph);
    cudaFree(d_routers);

    start = clock();
    // Run OSPF (Dijkstra)
    printf("Running Serial OSPF...\n");
    for (int i = 0; i < N; i++) {
        dijkstra(N, graph, routers, i);
        // Print shortest paths (optional)
        // printf("Shortest paths from Router %d:\n", i);
        // for (int j = 0; j < N; j++) {
        //     printf("Router %d -> Router %d: Distance = %d\n", i, j, routers[j].distance);
        // }
        // printf("\n");
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken: %.8f seconds\n", cpu_time_used);

    return 0;
}
