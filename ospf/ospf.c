#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_COST 10

void generate_network(int n, int **adj_matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                adj_matrix[i][j] = 0; // No self-loops, cost 0 to self.
            } else {
                int has_link = rand() % 5; // 20% chance of no link
                if (has_link == 0) {
                    adj_matrix[i][j] = 0; // No link
                } else {
                    adj_matrix[i][j] = rand() % MAX_COST + 1; // Random cost between 1 and MAX_COST
                }
            }
        }
    }
}

void print_network(int n, int **adj_matrix) {
    printf("Network Topology:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%3d ", adj_matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int n;
    printf("Enter the number of routers: ");
    scanf("%d", &n);

    // Seed the random number generator
    srand(time(NULL));

    // Allocate memory for the adjacency matrix
    int **adj_matrix = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        adj_matrix[i] = (int *)malloc(n * sizeof(int));
    }

    // Generate the network and print it
    generate_network(n, adj_matrix);
    print_network(n, adj_matrix);

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(adj_matrix[i]);
    }
    free(adj_matrix);

    return 0;
}
