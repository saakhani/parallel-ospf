#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <pthread.h>
#include <sys/time.h>

#define MAX_NODES 128
#define INF FLT_MAX

typedef struct {
    float *graph;
    int *visited;
    float *dist;
    int n;
    int start;
    int end;
} ThreadData;

void *dijkstraThread(void *arg) {
    ThreadData *data = (ThreadData *)arg;

    for (int count = 0; count < data->n - 1; count++) {
        int u = -1;
        float minDistance = INF;
        for (int v = data->start; v < data->end; v++) {
            if (!data->visited[v] && data->dist[v] <= minDistance) {
                minDistance = data->dist[v];
                u = v;
            }
        }

        if (u == -1) break;

        data->visited[u] = 1;

        for (int v = 0; v < data->n; v++) {
            if (!data->visited[v] && data->graph[u * data->n + v] && data->dist[u] != INF && data->dist[u] + data->graph[u * data->n + v] < data->dist[v]) {
                data->dist[v] = data->dist[u] + data->graph[u * data->n + v];
            }
        }
    }

    pthread_exit(NULL);
}

void dijkstra(float *graph, int n) {
    float dist[MAX_NODES];
    int visited[MAX_NODES];

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[0] = 0;

    pthread_t threads[MAX_NODES];
    ThreadData threadData[MAX_NODES];
    int threadsCount = 4; // Number of threads

    for (int i = 0; i < threadsCount; i++) {
        threadData[i].graph = graph;
        threadData[i].visited = visited;
        threadData[i].dist = dist;
        threadData[i].n = n;
        threadData[i].start = (n / threadsCount) * i;
        threadData[i].end = (n / threadsCount) * (i + 1);
        if (i == threadsCount - 1) threadData[i].end = n;

        pthread_create(&threads[i], NULL, dijkstraThread, (void *)&threadData[i]);
    }

    for (int i = 0; i < threadsCount; i++) {
        pthread_join(threads[i], NULL);
    }

    // printf("Vertex   Distance from Source\n");
    // for (int i = 0; i < n; i++) {
    //     printf("%d \t %f\n", i, dist[i]);
    // }
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

    struct timeval start, end;
    double cpu_time_used;

    gettimeofday(&start, NULL);

    dijkstra(graph, n);

    gettimeofday(&end, NULL);

    cpu_time_used = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Time taken for Parallel OSPF: %.8f seconds\n", cpu_time_used);

    return 0;
}