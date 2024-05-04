#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define MAX_NODES 1280
#define INF FLT_MAX

__global__ void dijkstraKernel(float *graph, int *visited, float *dist, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    // Find the vertex with the minimum distance value, from the set of vertices not yet included in the shortest path tree
    int min = -1;
    float minDistance = INF;
    for (int v = 0; v < n; v++) {
        if (!visited[v] && dist[v] <= minDistance) {
            minDistance = dist[v];
            min = v;
        }
    }

    // Mark the picked vertex as processed
    if (min == idx) {
        visited[min] = 1;
        __syncthreads(); // Synchronize to ensure all threads see the updated visited array

        // Update dist value of the adjacent vertices of the picked vertex.
        for (int v = 0; v < n; v++) {
            if (!visited[v] && graph[min * n + v] && dist[min] != INF && dist[min] + graph[min * n + v] < dist[v]) {
                dist[v] = dist[min] + graph[min * n + v];
            }
        }
    }
}

void dijkstraCUDA(float *graph, int n) {
    float *dev_graph;
    int *dev_visited;
    float *dev_dist;

    cudaMalloc((void **)&dev_graph, n * n * sizeof(float));
    cudaMalloc((void **)&dev_visited, n * sizeof(int));
    cudaMalloc((void **)&dev_dist, n * sizeof(float));

    int *visited = (int*)malloc(n * sizeof(int));
    float *dist = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[0] = 0;

    cudaMemcpy(dev_graph, graph, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dist, dist, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_visited, visited, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < n; i++) {
        dijkstraKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_graph, dev_visited, dev_dist, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(dist, dev_dist, n * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Vertex   Distance from Source\n");
    // for (int i = 0; i < n; i++) {
    //     printf("%d \t %f\n", i, dist[i]);
    // }

    free(visited);
    free(dist);
    cudaFree(dev_graph);
    cudaFree(dev_visited);
    cudaFree(dev_dist);
}

int main() {
    const int n = MAX_NODES;
    float graph[MAX_NODES * MAX_NODES];

    // Initialize graph
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            graph[i * n + j] = rand() % 10 + 1;
            if (i == j) graph[i * n + j] = 0;
        }
    }

    clock_t start, end;
    double cpu_time_used;
    
    start = clock();
    dijkstraCUDA(graph, n);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for Parallel OSPF: %.8f seconds\n", cpu_time_used);
    return 0;
}
