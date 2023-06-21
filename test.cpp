// Include libraries and header files

#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>

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
        while (arr[i] <= pivot && i < length)
            i++;
        while (arr[j] > pivot && j >= 0) 
            j--;
        if (i <= j)
        {
            swap(arr[i], arr[j]);
        }
        
    }
    return j;
}

int compute_displs(int* sdispls, int* rdispls, int* sendcnts, int* recvcnts, MPI_Comm comm) 
{
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    for (int i = 1; i < p; i++) {
        sdispls[i] = sdispls[i - 1] + sendcnts[i - 1];
        rdispls[i] = rdispls[i - 1] + recvcnts[i - 1];
    }
    return 0;
}

void parallelqsort(int *arr, int length, MPI_Comm comm)
{


    int rank, size;

    MPI_Comm_rank(comm, &rank);

    MPI_Comm_size(comm, &size);


     int p = size;

    if (size == 1)
    {
        std::sort(arr, arr + length);
        return;
    }

    // Use MPI All reduce to find total number of elements in a communicator 

    int total_size = 0;

    MPI_Allreduce(&size, &total_size, 1, MPI_INT, MPI_SUM, comm);

    int floor_val = total_size / size;
    int remainder = total_size % size;

    // Pick a random pivot from 0 to total_size-1

    int pivot_idx = rand() % total_size;

    // Find processor that has the pivot value

    int pivot_proc = 0;

    if (remainder == 0)
    {
        pivot_proc = pivot_idx / floor_val;
    }
    else
    {
        if (pivot_idx < remainder * (floor_val + 1))
        {
            pivot_proc = pivot_idx / (floor_val + 1);
        }
        else
        {
            pivot_proc = (pivot_idx - remainder * (floor_val + 1)) / floor_val + remainder;
        }
    }

    // Find pivot value on pivot_proc

    int pivot_value = 0;

    if (rank == pivot_proc)
    {
        if (remainder == 0)
        {
            pivot_value = arr[pivot_idx - rank * floor_val];
        }
        else
        {
            if (rank < remainder)
            {
                pivot_value = arr[pivot_idx - rank * (floor_val + 1)];
            }
            else
            {
                pivot_value = arr[pivot_idx - remainder * (floor_val + 1) - (rank - remainder) * floor_val];
            }
        }
    }

    // Broadcast pivot_value to all processors

    MPI_Bcast(&pivot_value, 1, MPI_INT, pivot_proc, comm);

    // Partition the array in place

    int partition_idx = partition(arr, length, pivot_value);

    int loc_arr1_size = partition_idx + 1;

    int loc_arr2_size = length - loc_arr1_size;

    int loc_arr1_sizes[size] = {0};

    int loc_arr2_sizes[size] = {0};

    // left and subarray sizes gathered on all the processors

    MPI_Allgather(&loc_arr1_size, 1, MPI_INT, loc_arr1_sizes, 1, MPI_INT, comm);

    MPI_Allgather(&loc_arr2_size, 1, MPI_INT, loc_arr2_sizes, 1, MPI_INT, comm);

    std::vector<int> cur_size(p);
    MPI_Allgather(&length, 1, MPI_INT, &cur_size[0], 1, MPI_INT, comm);

    int prefix_1[size + 1] = {0};

    int prefix_2[size + 1] = {0};

    // Optimize it later to compute prefixes in O(log p) instead of p

    for (int i = 1; i <= size; i++)
    {
        prefix_1[i] = prefix_1[i - 1] + loc_arr1_sizes[i - 1];
        prefix_2[i] = prefix_2[i - 1] + loc_arr2_sizes[i - 1];
    }

    int m_prime = prefix_1[size];

    int m_double_prime = prefix_2[size];

    int comm_size_1 = 0;
    int comm_size_2 = 0;

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

    // compute the newsizes of each processor in comm

    int newsize[size] = {0};

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

    int global_index[length] = {0};
    int count_left = 0;
    int count_right = 0;
    for (int i = 0; i < length; i++) {
        
        if (arr[i] <= pivot_value) {
            global_index[i] = prefix_1[rank] + count_left;
            count_left++;
        }
        else {
            global_index[i] =  prefix_1[size] +  prefix_2[rank] + count_right;
            count_right++;
        }
    }

    // calculate target processor using global indices

    int newprefixsum[size + 1] = {0};

    for (int i = 1; i < size + 1; i++)
    {
        newprefixsum[i] = newprefixsum[i - 1] + newsize[i-1];
    }

    int *target_proc = new int[length];

    for (int i = 0; i < length; i++)
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

    int sendcounts[size] = {0};

    for (int i = 0; i < length; i++)
    {
        sendcounts[target_proc[i]]++;
    }

    int recvcounts[size] = {0};

    // compute recvcounts_new based on sendcounts_new (this is basically matrix transpose)

    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);

    int *recvbuf = new int[newsize[rank]];

    // compute senddispls_new and recvdispls_new based on sendcounts_new and recvcounts_new

    int senddispls[size] = {0};

    for (int i = 1; i < size; i++)
    {
        senddispls[i] = senddispls[i - 1] + sendcounts[i - 1];
    }

    int recvdispls[size] = {0};

    for (int i = 1; i < size; i++)
    {
        recvdispls[i] = recvdispls[i - 1] + recvcounts[i - 1];
    }

    // MPI_Alltoallv to exchange data

    MPI_Alltoallv(arr, sendcounts, senddispls, MPI_INT, recvbuf, recvcounts, recvdispls, MPI_INT, comm);

    MPI_Comm comm_1;

    int new_local_size = newsize[rank];

    MPI_Comm_split(comm, rank < comm_size_1, rank, &comm_1);

    // print comm_size_1 and comm_size_2

    parallelqsort(recvbuf, new_local_size, comm_1);

    MPI_Comm_free(&comm_1);  

    //Code for second all to all v lifted from Github.com/ywhust/Parallel-Programming/blob/master/programming-assignment2/parallel_sort.cpp
    //Change it later to compute secondcounts using newsize and arr values like for first all to all v

    std::vector<int> tmp_sendcnts(p, 0), tmp_recvcnts(p, 0), tmp_sdispls(p, 0), tmp_rdispls(p, 0);
    std::vector<int> tmp_reference(cur_size);
    for (int i = 0, j = 0; i < p; i++) 
        {
        int send = newsize[i];
        while (send > 0) {
            int s = (send <= tmp_reference[j])? send : tmp_reference[j];
            if (i == rank) tmp_sendcnts[j] = s;
            if (j == rank) tmp_recvcnts[i] = s;
            send -= s;
            tmp_reference[j] -= s;
            if (tmp_reference[j] == 0) j++;
        }
    }
    compute_displs(&tmp_sdispls[0], &tmp_rdispls[0], &tmp_sendcnts[0], &tmp_recvcnts[0], comm);

    // Second all to all v for load balancing - simple all gather would suffice if goal is just write to file

    MPI_Alltoallv(recvbuf, &tmp_sendcnts[0], &tmp_sdispls[0], MPI_INTEGER, arr, &tmp_recvcnts[0], &tmp_rdispls[0], MPI_INTEGER, comm);

}


int main(int argc, char *argv[])

{
    int rank, size;

    //int seed = atoi(argv[2]);

    srand(rand());

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int *arr;
    int n;

    if (rank == ROOT)
    {
        ifstream infile(argv[1]);
        infile >> n;
        arr = new int[n];
        for (int i = 0; i < n; i++)
        {
            infile >> arr[i];
        }
        infile.close();
    }

    // Broadcast n

    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Compute sendcounts and displs in O(p) for scatterv
    
    int *sendcounts = new int[size];
    int *displs = new int[size];

    int rem = n % size;
    int floor = n / size;

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

    // Initialize local array

    int *loc_arr = new int[sendcounts[rank]];

    // Scatter arr on ROOT

    MPI_Scatterv(arr, sendcounts, displs, MPI_INT, loc_arr, sendcounts[rank], MPI_INT, ROOT, MPI_COMM_WORLD);

    // sort the received data recusively

    double start = MPI_Wtime();

    parallelqsort(loc_arr, sendcounts[rank], MPI_COMM_WORLD);

    double time_elapsed = MPI_Wtime() - start;

    // Gather loc_arr on ROOT

    MPI_Gatherv(loc_arr, sendcounts[rank], MPI_INT, arr, sendcounts, displs, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Write the sorted array in first line and time elapsed in second line

    if (rank == ROOT)
    {
        ofstream outfile(argv[2]);
        for (int i = 0; i < n; i++)
        {
            outfile << arr[i] << " ";
        }
        outfile << endl;
        outfile << std::setprecision(6)<<1000*time_elapsed << endl;
        outfile.close();
    }

    MPI_Finalize();

    return 0;

}


