/***************************************************************************
* Filename: LabClass.cpp
* Author: Benjamin Carrington
* Purpose: Coursework #1 [CS-M98]
*
***************************************************************************
* Execution times (averages of five executions):
*
*   FindClosestDistance:		31,105,061 us, 31.1 s
*
* -------------------------------------------------------------------------
*   FindClosestDistance:		  5,967,435 us, 6.0 s
*       + Multi-threading
*
* -------------------------------------------------------------------------
*   FindClosestDistance:		  6,287,287 us, 6.3 s
*       + Multi-threading
*       + Improved locality
*
* -------------------------------------------------------------------------
*   FindClosestDistance:		  7,078,622 us, 7.1 s
*       + Multi-threading
*       + Improved locality
*       + Branchless
*
* -------------------------------------------------------------------------
*   FindClosestDistance:			818,097 us, 0.8 s
*       + Multi-threading
*       + Improved locality
*       + Branchless
*       + SIMD
*
* -------------------------------------------------------------------------
*   Test information:
*       N:              32,768
*       NB_THREADS:     8
*       Processor:      Intel i7-7700k (~4.2 GHz)
*           # Cores:    4
*           # Threads:  8
*
*	NOTE: Execution times were taken without the switch statement in
*		CreateThreads() or the one in _tmain().
*
**************************************************************************/

#include "Chrono.h"
#include <intrin.h>
#include <thread>
#include <mutex>
#include <random>

#define NB_THREADS 8
#define CHUNK_SIZE 32
#define N 32768

/******* SIMD definitions *******/
#define float8 __m256
#define mul8(x, y) _mm256_mul_ps(x, y)
#define add8(x, y) _mm256_add_ps(x, y)
#define sub8(x, y) _mm256_sub_ps(x, y)
#define sqrt8(x) _mm256_sqrt_ps(x)
#define set8(x) _mm256_set1_ps(x)

/******* Random point generation *******/
float *GenerateRandomCoordinates(int n)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());	
	static std::uniform_real_distribution<> dist(-1.0, 1.0);

	float *pt = new float[n];
	for (int i = 0; i<n; i++)
		pt[i] = dist(gen);
	return pt;
}

/******* Distance between two points *******/
float Distance2D(float *x, float *y, int i, int j)
{
	return sqrt((x[i] - x[j])*(x[i] - x[j]) + (y[i] - y[j])*(y[i] - y[j]));
}

/******* SIMD version of Distance2D() *******/
float8 Distance2D_SIMD(float8 x_i, float8 y_i, float8 x_j, float8 y_j)
{
	return sqrt8(
		add8(
			mul8(
				sub8(x_i, x_j),
				sub8(x_i, x_j)
			),
			mul8(
				sub8(y_i, y_j),
				sub8(y_i, y_j)
			)
		)
	);
}

/******* Basic algorithm *******/
void FindClosestDistance(int n, float *x, float *y, float *results)
{
	for (int i = 0; i<n; i++)
	{
		results[i] = 999.;
		for (int j = 0; j < n; j++)
		{
			if (i != j)
			{
				float d = Distance2D(x, y, i, j);
				if (d < results[i])
					results[i] = d;
			}
		}
	}
}

/******* [MT] *******/
void FindClosestDistance_MT(int n, float *x, float *y, float *results, int rankID)
{
	int begin = rankID * (n / NB_THREADS);
	int end = (rankID + 1) * (n / NB_THREADS);

	for (int i = begin; i < end; i++)
	{
		results[i] = 999.;
		for (int j = 0; j<n; j++)
		{
			if (i != j)
			{
				float d = Distance2D(x, y, i, j);
				if (d < results[i])
					results[i] = d;
			}
		}
	}
}

/******* [MT | Locality] *******/
void FindClosestDistance_MT_Block(int n, float *x, float *y, float *results, int rankID)
{
	int begin = rankID * (n / NB_THREADS);
	int end = (rankID + 1) * (n / NB_THREADS);

	for (int i = begin; i < end; i++)
		results[i] = 999.;

	for (int i = begin; i < end; i += CHUNK_SIZE)
	{
		for (int j = 0; j < n; j += CHUNK_SIZE)
		{
			for (int i_chunk = 0; i_chunk < CHUNK_SIZE; i_chunk++)
			{
				for (int j_chunk = 0; j_chunk < CHUNK_SIZE; j_chunk++)
				{
					if (i + i_chunk != j + j_chunk)
					{
						float d = Distance2D(x, y, i + i_chunk, j + j_chunk);

						if (d < results[i + i_chunk])
							results[i + i_chunk] = d;
					}
				}
			}
		}
	}
}

/******* [MT | Locality | Branchless] *******/
void FindClosestDistance_MT_Block_Branchless(int n, float *x, float *y, float *results, int rankID)
{
	int begin = rankID * (n / NB_THREADS);
	int end = (rankID + 1) * (n / NB_THREADS);

	for (int i = begin; i < end; i++)
		results[i] = 999.;

	for (int i = begin; i < end; i += CHUNK_SIZE)
	{
		for (int j = 0; j < n; j += CHUNK_SIZE)
		{
			for (int i_chunk = 0; i_chunk < CHUNK_SIZE; i_chunk++)
			{
				for (int j_chunk = 0; j_chunk < CHUNK_SIZE; j_chunk++)
				{
					float d = Distance2D(x, y, i + i_chunk, j + j_chunk);

					results[i + i_chunk] = ((i + i_chunk != j + j_chunk) * (((d < results[i + i_chunk]) * d) +
						((d > results[i + i_chunk]) * results[i + i_chunk]))) +
						((i + i_chunk == j + j_chunk) * results[i + i_chunk]);
				}
			}
		}
	}
}

/******* [MT | Locality | Branchless | SIMD] *******/
void FindClosestDistance_MT_Block_Branchless_SIMD(int n, float *x, float *y, float *results, int rankID)
{
	int begin = rankID * (n / NB_THREADS);
	int end = (rankID + 1) * (n / NB_THREADS);

	float8 *results_256 = (float8 *)results;
	float8 *x_256 = (float8 *)x;
	float8 *y_256 = (float8 *)y;
	float8 x_256_j, y_256_j; 

	for (int i = begin; i < end; i += 8)
		results_256[i / 8] = set8(999);

	for (int i = begin; i < end; i += CHUNK_SIZE)
	{
		for (int j = 0; j < n; j += CHUNK_SIZE)
		{
			for (int i_chunk = 0; i_chunk < (CHUNK_SIZE / 8); i_chunk++)
			{
				for (int j_chunk = 0; j_chunk < (CHUNK_SIZE / 8); j_chunk++)
				{
					x_256_j = x_256[(j / 8) + j_chunk];
					y_256_j = y_256[(j / 8) + j_chunk];

					for (int k = 0; k < 7 + (i + i_chunk != j + j_chunk); k++)
					{
						float tmp_x = x_256_j.m256_f32[0];
						float tmp_y = y_256_j.m256_f32[0];

						x_256_j = _mm256_loadu_ps((float *)&x_256_j + 1);
						y_256_j = _mm256_loadu_ps((float *)&y_256_j + 1);

						x_256_j.m256_f32[7] = tmp_x;
						y_256_j.m256_f32[7] = tmp_y;

						float8 d = Distance2D_SIMD(x_256[(i / 8) + i_chunk], y_256[(i / 8) + i_chunk], x_256_j, y_256_j);
						
						results_256[(i / 8) + i_chunk] = _mm256_blendv_ps(results_256[(i / 8) + i_chunk], d,
							_mm256_cmp_ps(d, results_256[(i / 8) + i_chunk], _CMP_LT_OQ) // MASK
						);
					}
				}
			}
		}
	}
}

/******* Thread creation *******/
void CreateThreads(int opt, int n, float *x, float *y, float *results)
{
	std::thread t[NB_THREADS];

	switch (opt)
	{
	case 0:
		for (int i = 0; i < NB_THREADS; i++)
			t[i] = std::thread(FindClosestDistance_MT, n, x, y, results, i);
		break;
	case 1:
		for (int i = 0; i < NB_THREADS; i++)
			t[i] = std::thread(FindClosestDistance_MT_Block, n, x, y, results, i);
		break;
	case 2:
		for (int i = 0; i < NB_THREADS; i++)
			t[i] = std::thread(FindClosestDistance_MT_Block_Branchless, n, x, y, results, i);
		break;
	case 3:
		for (int i = 0; i < NB_THREADS; i++)
			t[i] = std::thread(FindClosestDistance_MT_Block_Branchless_SIMD, n, x, y, results, i);
		break;
	default:
		printf("T in:\n\tCreateThreads()\n");
		return;
	}

	for (int i = 0; i < NB_THREADS; i++)
		t[i].join();
}

/******* Main function *******/
int _tmain(int argc, _TCHAR* argv[])
{
	int n = N;
	float *comp = new float[n];
	float *closest = new float[n];
	float *x = GenerateRandomCoordinates(n);
	float *y = GenerateRandomCoordinates(n);
	bool tester;

	Chrono timer;
	FindClosestDistance(n, x, y, comp);
	printf("Original:   Time: ");
	timer.PrintElapsedTime_us("");
	printf(" Original\n\n");

	/** Threaded */
	for (int i = 0; i < 4; i++)
	{
		tester = false;

		timer.InitChrono();

		CreateThreads(i, n, x, y, closest);

		switch (i)
		{
		case 0:
			printf("MT:         Time: ");
			break;
		case 1:
			printf("MT_Lo:      Time: ");
			break;
		case 2:
			printf("MT_Lo_Br:   Time: ");
			break;
		case 3:
			printf("MT_Lo_Br_S: Time: ");
			break;
		default:
			printf("Switch error\n");
			break;
		}

		timer.PrintElapsedTime_us("");

		for (int j = 0; j < n; j++)
		{
			if (closest[j] != comp[j])
				tester = true;
			closest[j] = 0.;
		}

		if (tester)
			printf(" Failure\n\n");
		else
			printf(" Success\n\n");
	}

	delete[] closest;
	delete[] x;
	delete[] y;

	return 0;
}
