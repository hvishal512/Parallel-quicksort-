// Include libraries and header files

#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#define ROOT 0

using namespace std;

int partition(int *arr, int length, int pivot)
{
    if (length == 0)
    {
        return -1;
    }

    int i = 0, j = length - 1;

    while (i <= j)
    {
        while (arr[i] <= pivot)
            i++;
        while (arr[j] > pivot)
            j--;
        if (i <= j)
        {
            swap(arr[i], arr[j]);
        }
        
    }
    return j;
}


int main(int argc, char *argv[])

{
    int rank, size;
    int n = atoi(argv[1]);

    //int seed = atoi(argv[2]);

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3)
    {
        if (rank == ROOT)
            cout << "Usage: " << argv[0] << " <np> <n> <input>" << endl;
        MPI_Finalize();
        return 1;
    }

    // Define local array

    
    int *sendcounts = new int[size];
    int *displs = new int[size];
    int *arr = new int[n];
    int *loc_arr = new int[n/size + 1];

    // Calculate sendcounts and displs to use MPI_Scatterv

    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = n / size;
        if (i < n%size)
        {
            sendcounts[i]++;
        }
        displs[i] = i * (n / size);
        if (i < n%size)
        {
            displs[i] += i;
        }
        else
        {
            displs[i] += n%size;
        }
    }

    // Use processor with rank 0 to read in the array from file
    // where array numbers are separated by space

    if (rank == ROOT)
    {
        ifstream fin(argv[2]);
        for (int i = 0; i < n; i++)
        {
            fin >> arr[i];
        }
        fin.close();
    }

    MPI_Scatterv(arr, sendcounts, displs, MPI_INT, loc_arr, sendcounts[rank], MPI_INT, ROOT, MPI_COMM_WORLD);

    // Pick a random pivot from 0 to n-1

    srand(time(NULL));

    //int pivot = rand() % n; //Make sure all processors get the same pivot
    int pivot = 0;

    // Find processor that has the pivot value

    int pivot_proc = 0;

    for (int i = 0; i < size; i++)
    {
        if (pivot >= displs[i] && pivot < displs[i] + sendcounts[i])
        {
            pivot_proc = i;
            break;
        }
    }

    // Broadcast pivot_proc to all processors

    MPI_Bcast(&pivot_proc, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Find pivot value on pivot_proc

    int pivot_value = 0;

    if (rank == pivot_proc)
    {
        pivot_value = loc_arr[pivot - displs[pivot_proc]];
    }

    // Broadcast pivot_value to all processors

    MPI_Bcast(&pivot_value, 1, MPI_INT, pivot_proc, MPI_COMM_WORLD);

    // Print the pivot value on processor with rank 0


    int *loc_arr1 = new int[n/size + 1];

    int *loc_arr2 = new int[n/size + 1];

    int loc_arr1_size = 0;

    int loc_arr2_size = 0;

    int partition_idx = partition(loc_arr, sendcounts[rank], pivot_value);

    // print partition index on each processor

    //cout << "Processor " << rank << " partition index: " << partition_idx << endl;

    for (int i = 0; i < sendcounts[rank]; i++)
    {
        if (loc_arr[i] <= pivot_value)
        {
            loc_arr1[loc_arr1_size] = loc_arr[i];
            loc_arr1_size++;
        }
        else
        {
            loc_arr2[loc_arr2_size] = loc_arr[i];
            loc_arr2_size++;
        }
    }

    // All Gather loc_arr1_size and loc_arr2_size to all processors

    int loc_arr1_sizes[size] = {0};

    int loc_arr2_sizes[size] = {0};

    MPI_Allgather(&loc_arr1_size, 1, MPI_INT, loc_arr1_sizes, 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Allgather(&loc_arr2_size, 1, MPI_INT, loc_arr2_sizes, 1, MPI_INT, MPI_COMM_WORLD);

    int m_prime = 0;
    int m_double_prime = 0;

    // Compute sum of local_arr1_sizes and loc_arr2_sizes

    // Optimise it later using MPI_Allreduce

    for (int i = 0; i < size; i++)
    {
        m_prime += loc_arr1_sizes[i];
        m_double_prime += loc_arr2_sizes[i];
    }


    //print the test array

    int comm_size_1 = 0;
    int comm_size_2 = 0;

    // Compute comm_size_1 and comm_size_2

    comm_size_1 = lround(1.0*m_prime*size/(m_prime + m_double_prime));

    if (comm_size_1 == 0)
    {
        comm_size_1 = 1;
    }
    else if(comm_size_1 == size)
    {
        comm_size_1 = size - 1;
    }

    comm_size_2 = size - comm_size_1;

    // MPI exclusive scan on loc_arr1_sizes to get displs_1 - optimise it later using MPI_Exscan

    int displs_1[size + 1] = {0};
   
    int displs_2[size + 1] = {0};

    for (int i = 1; i < size + 1; i++)
    {
        displs_1[i] = displs_1[i - 1] + loc_arr1_sizes[i - 1];
    }


    for (int i = 1; i < size + 1; i++)
    {
        displs_2[i] = displs_2[i - 1] + loc_arr2_sizes[i - 1];
    }

    // compute newsize

    int *newsize = new int[size];

    for (int i = 0; i < comm_size_1; i++)
    {
        newsize[i] = m_prime/comm_size_1;
        if (i < m_prime%comm_size_1)
        {
            newsize[i]++;
        }
    }

    for(int i=comm_size_1; i<size; i++)
    {
        newsize[i] = m_double_prime/comm_size_2;
        if (i < m_double_prime%comm_size_2 + comm_size_1)
        {
            newsize[i]++;
        }
    }

    // calculate global indices for the local array

    int *global_index = new int[sendcounts[rank]];
    int count_left = 0;
    int count_right = 0;
    for (int i = 0; i < sendcounts[rank]; i++) {
        
        if (loc_arr[i] <= pivot_value) {
            global_index[i] = displs_1[rank] + count_left;
            count_left++;
        }
        else {
            global_index[i] =  displs_1[size] +  displs_2[rank] + count_right;
            count_right++;
        }
    }

    // calculate target processor using global indices

    
    int newprefixsum[size + 1] = {0};
    //newprefixsum[0] = newsize[0];

    for (int i = 1; i < size + 1; i++)
    {
        newprefixsum[i] = newprefixsum[i - 1] + newsize[i-1];
    }

    int *target_proc = new int[sendcounts[rank]];

    for (int i = 0; i < sendcounts[rank]; i++)
    {

        if (global_index[i] < m_prime)
        {
            int temp_target = m_prime/comm_size_1;
            target_proc[i] = global_index[i]/temp_target;
            if (global_index[i] < newprefixsum[target_proc[i]])
            {
                target_proc[i]--;
            }
        }
        else
        {
            int temp_target = m_double_prime/comm_size_2;
            target_proc[i] = comm_size_1 + (global_index[i] - m_prime)/temp_target;
            if (global_index[i] < newprefixsum[target_proc[i]])
            {
                target_proc[i]--;
            }
        }
    }

    int *sendcounts_new = new int[size];

    // initialize sencounts_new to 0

    for (int i = 0; i < size; i++)
    {
        sendcounts_new[i] = 0;
    }

    for (int i = 0; i < sendcounts[rank]; i++)
    {
        sendcounts_new[target_proc[i]]++;
    }

    int *recvcounts_new = new int[size];

    // initialize recvcounts_new to 0

    for (int i = 0; i < size; i++)
    {
        recvcounts_new[i] = 0;
    }

    // compute recvcounts_new based on sendcounts_new (this is basically matrix transpose)

    MPI_Alltoall(sendcounts_new, 1, MPI_INT, recvcounts_new, 1, MPI_INT, MPI_COMM_WORLD);

    int *recvbuf = new int[newsize[rank]];

    // compute senddispls_new and recvdispls_new based on sendcounts_new and recvcounts_new

    int *senddispls_new = new int[size];

    // initialize senddispls_new to 0

    for (int i = 0; i < size; i++)
    {
        senddispls_new[i] = 0;
    }

    for (int i = 1; i < size; i++)
    {
        senddispls_new[i] = senddispls_new[i - 1] + sendcounts_new[i - 1];
    }

    int *recvdispls_new = new int[size];

    // initialize recvdispls_new to 0

    for (int i = 0; i < size; i++)
    {
        recvdispls_new[i] = 0;
    }

    for (int i = 1; i < size; i++)
    {
        recvdispls_new[i] = recvdispls_new[i - 1] + recvcounts_new[i - 1];
    }

    // MPI_Alltoallv to exchange data

    MPI_Alltoallv(loc_arr, sendcounts_new, senddispls_new, MPI_INT, recvbuf, recvcounts_new, recvdispls_new, MPI_INT, MPI_COMM_WORLD);

    // print recvbuf on every processor

    cout << "Processor " << rank << " has the following recvbuf: ";

    for (int i = 0; i < newsize[rank]; i++)
    {
        cout << recvbuf[i] << " ";
    }

    cout << endl;

    //parallelqsort(loc_arr, sendcounts[rank], MPI_COMM_WORLD);

    // print loc_arr on every processor

    MPI_Finalize();

    return 0;

}


