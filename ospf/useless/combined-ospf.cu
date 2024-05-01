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
__global__ void dijkstraKernel(float *graph, int *visited, float *dist, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    while (true) {
        // Find the vertex with the minimum distance
        int min_index = -1;
        float min_distance = INT_MAX;

        for (int i = 0; i < n; i++) {
            if (!visited[i] && dist[i] < min_distance) {
                min_distance = dist[i];
                min_index = i;
            }
        }

        if (min_index == -1)
            break;

        // Mark the vertex as visited
        visited[min_index] = 1;

        // Update the distances for all other vertices
        for (int i = 0; i < n; i++) {
            float weight = graph[min_index * n + i];
            if (weight > 0 && dist[min_index] + weight < dist[i]) {
                dist[i] = dist[min_index] + weight;
            }
        }
    }
}

void runDijkstraOnGPU(int graph[MAX_ROUTERS][MAX_ROUTERS], int n) {
    float *d_graph, *d_dist;
    int *d_visited;

    cudaMalloc(&d_graph, n * n * sizeof(float));
    cudaMalloc(&d_dist, n * sizeof(float));
    cudaMalloc(&d_visited, n * sizeof(int));

    cudaMemcpy(d_graph, graph, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    dijkstraKernel<<<grid, block>>>(d_graph, d_visited, d_dist, n);

    cudaDeviceSynchronize();  // Ensure the kernel execution completes

    float *h_dist = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h_dist, d_dist, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Output distances for demonstration
    for (int i = 0; i < n; i++) {
        printf("Distance from source to %d is %f\n", i, h_dist[i]);
    }

    free(h_dist);
    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_visited);
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

    int graph[MAX_ROUTERS][MAX_ROUTERS] = {0};

    // Seed random number generator
    srand(time(NULL));

    // Generate network topology
    generateNetwork(N, graph);

    clock_t start, end;
    double cpu_time_used;

    // Run Parallel OSPF
    printf("Running Parallel OSPF...\n");
    start = clock();
    runDijkstraOnGPU(graph, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for Parallel OSPF: %.8f seconds\n", cpu_time_used);

    // Run Serial OSPF
    printf("Running Serial OSPF...\n");
    start = clock();
    Router routers[MAX_ROUTERS];
    for (int i = 0; i < N; i++) {
        dijkstra(N, graph, routers, i);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for Serial OSPF: %.8f seconds\n", cpu_time_used);

    return 0;
}
