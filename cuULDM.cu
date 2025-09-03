#include <stdio.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include "common/book.h"
#include "common/gpu_anim.h"

//per compilare basta: "nvcc -arch=sm_75 cuULDM.cu -o main -L".\lib" -lglut64 -lopengl32 -lglu32 -lgdi32 -lcufft"

#define SQR(x) ((x)*(x))
#define PI 3.1415926535
#define EPS (1e-12)

#define N 256
#define NNN (N*N*N)
#define L 0.016
#define DX (L/N)
#define DT (DX*DX/PI * 100)
#define T_MAX 0.26
//#define N_OF_SAVE ((int)(T_MAX/DT) + 1)
#define N_OF_SAVE 1
#define SAVE_ENERGY 0
#define SAVE_DENSITY_PLAN 0
#define SAVE_DENSITY_LINE 0
#define SAVE_POTENTIAL_PLAN 0

#define CMASS 57.

typedef struct Soliton {
    double x,y,z;
    double vx,vy,vz;
    double mass;
    double delta;
} Soliton;

#define N_OF_SOLITONS 1
__constant__ Soliton d_solitons[N_OF_SOLITONS];

struct DataBlock {
    cufftDoubleComplex *d_psi, *d_phi_k;
    double *d_phi, *d_k2;
    cufftHandle planZ2Z, planD2Z, planZ2D;

    double *d_X, *d_factor, *d_cos_k2dt2, *d_sin_k2dt2;
};

/*************************FUNCTIONs PROTOTYPEs******************************/

__global__ void SetSinCos(double *d_sin_k2dt2, double *d_cos_k2dt2, double *d_k2);
__global__ void HalfStep(cufftDoubleComplex *d_psi, double *d_phi);
__global__ void HalfStepFourier(cufftDoubleComplex *d_psi, double *d_k2, double *d_sin_k2dt2, double *d_cos_k2dt2);
__global__ void NormalizeZ(cufftDoubleComplex *d_psi);
__global__ void PsiToRho(cufftDoubleComplex *d_psi, double *d_rho);
__global__ void PoissonFourier(cufftDoubleComplex *d_phi_k, double *d_factor);
__global__ void NormalizeR(double *d_phi);
__global__ void CPotential(double *d_phi, double *d_X);
__global__ void Step(cufftDoubleComplex *d_psi, double *d_phi);
__global__ void Init(cufftDoubleComplex *d_psi, double *d_X, double delta_x, double *d_f, double *d_k2, double *d_factor);
__global__ void GetSlice(cufftDoubleComplex *d_psi, double *d_slice);
void Initialize(cufftDoubleComplex *d_psi, double *d_phi, cufftDoubleComplex *d_phi_k, double *d_k2, cufftHandle planD2Z, cufftHandle planZ2D, double *d_X, double *d_factor);
void PseudoSpectralSolver(uchar4* outputBitmap, int t, cufftDoubleComplex *d_psi, double *d_phi, cufftDoubleComplex *d_phi_k, double *d_k2, cufftHandle planZ2Z, cufftHandle planD2Z, cufftHandle planZ2D, double *d_X, double *d_factor, double *d_sin_k2dt2, double *d_cos_k2dt2);

__device__ void colormap_seismic(float value, unsigned char &r, unsigned char &g, unsigned char &b) {
    value = fminf(fmaxf(value, -1.0f), 1.0f);  // clamp tra -1 e 1

    if (value < 0.0f) {
        // Blu → Bianco
        r = (unsigned char)(255.0f * (1.0f + value));
        g = (unsigned char)(255.0f * (1.0f + value));
        b = 255;
    } else {
        // Bianco → Rosso
        r = 255;
        g = (unsigned char)(255.0f * (1.0f - value));
        b = (unsigned char)(255.0f * (1.0f - value));
    }
}
__global__ void field_to_color(uchar4 *ptr, cufftDoubleComplex *psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= N || y >= N) return;

    int offset = x + y * blockDim.x * gridDim.x;
    int z = N / 2;
    int offset2 = z + y * N + x * N * N;

    float phi = psi[offset2].x * psi[offset2].x + psi[offset2].y * psi[offset2].y;

    float alpha = SQR(1428. / 3.883);
    float norm = SQR(alpha);

    float value = phi/norm * 2 - 1;

    unsigned char r, g, b;
    colormap_seismic(value, r, g, b);

    ptr[offset].x = r;
    ptr[offset].y = g;
    ptr[offset].z = b;
    ptr[offset].w = 255;
}

void anim_exit( DataBlock *d ) {
    cudaFree(d->d_psi);
    cudaFree(d->d_phi);
    cudaFree(d->d_phi_k);
    cudaFree(d->d_k2);
    
    cufftDestroy(d->planZ2Z);
    cufftDestroy(d->planD2Z);
    cufftDestroy(d->planZ2D);

    cudaFree(d->d_X);
    cudaFree(d->d_factor);
    cudaFree(d->d_cos_k2dt2);
    cudaFree(d->d_sin_k2dt2);
}
void anim_gpu( uchar4* outputBitmap, DataBlock *d, int ticks ) {
    
    PseudoSpectralSolver(outputBitmap, ticks, d->d_psi, d->d_phi, d->d_phi_k, d->d_k2, d->planZ2Z, d->planD2Z, d->planZ2D, d->d_X, d->d_factor, d->d_sin_k2dt2, d->d_cos_k2dt2);
    printf("TIME = %lf Gyr\r", ticks * DT * 75.5);
    fflush(stdout);
}
/*************************MAIN******************************/

int main() {
    //
    dim3    blocks(N/8,N/8,N/8); 
    dim3    threads(8,8,8);
    //

    DataBlock   data;
    GPUAnimBitmap bitmap( N, N, &data );
    int imageSize = bitmap.image_size();

    //

    cudaMalloc((void**)&data.d_psi, sizeof(cufftDoubleComplex) * NNN);
    cudaMalloc((void**)&data.d_phi_k, sizeof(cufftDoubleComplex) * N * N * (N/2 + 1));
    cudaMalloc((void**)&data.d_phi, sizeof(double) * NNN);
    cudaMalloc((void**)&data.d_k2, sizeof(double) * NNN);

    cufftHandle planZ2Z, planD2Z, planZ2D;
    cufftPlan3d(&planZ2Z, N, N, N, CUFFT_Z2Z);
    cufftPlan3d(&planD2Z, N, N, N, CUFFT_D2Z);
    cufftPlan3d(&planZ2D, N, N, N, CUFFT_Z2D);

    data.planZ2Z = planZ2Z;
    data.planZ2D = planZ2D;
    data.planD2Z = planD2Z;
    //

    //solitons
    Soliton *h_solitons = (Soliton *)malloc(sizeof(Soliton) * N_OF_SOLITONS);
                       //       x,  y,  z,  vx, vy, vz,  m,  ph
    h_solitons[0] = Soliton{  0., 0., 0., 0., 0., 0., 1428., 0.};

    cudaMemcpyToSymbol(d_solitons, h_solitons, sizeof(Soliton) * N_OF_SOLITONS);
    free(h_solitons);
    //

    //
    double* X = (double *)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) X[i] = -L/2 + i * DX + DX * 0.5;

    cudaMalloc((void **)&data.d_X, sizeof(double) * N);

    cudaMemcpy(data.d_X, X, sizeof(double) * N, cudaMemcpyHostToDevice);
    free(X);
    
    cudaMalloc((void **)&data.d_factor, sizeof(double) * NNN);
    
    int save_size = (int)(T_MAX/DT) + 1;
    int* save = (int*)malloc(sizeof(int)*save_size);
    for(int t = 0; t < save_size; t++) save[t] = 0;
    for(int t = 1; t < N_OF_SAVE; t++) {
        //save[(int)(t * (double)((int)(T_MAX/DT) / (N_OF_SAVE - 1.)))] = 1;
    }

    FILE *out1, *out2, *out3, *out4;
    double energy0;
    //

    //
    Initialize(data.d_psi, data.d_phi, data.d_phi_k, data.d_k2, planD2Z, planZ2D,data.d_X, data.d_factor);
    //

    cudaMalloc((void **)&data.d_sin_k2dt2, sizeof(double) * NNN);
    cudaMalloc((void **)&data.d_cos_k2dt2, sizeof(double) * NNN);

    SetSinCos<<<blocks,threads>>>(data.d_sin_k2dt2, data.d_cos_k2dt2, data.d_k2);

    double *h_slice = (double *)malloc(sizeof(double) * N * N);
    double *d_slice;
    cudaMalloc((void **)&d_slice, sizeof(double) * N * N);

    //
    if(SAVE_DENSITY_PLAN) {
        out1 = fopen("plan.dat", "w");
    }
    if(SAVE_ENERGY) {
        out2 = fopen("energy.dat", "w");
    }
    if(SAVE_DENSITY_LINE) {
        out3 = fopen("line.dat", "w");
    }
    if(SAVE_POTENTIAL_PLAN) {
        out4 = fopen("pot.dat", "w");
    }
    //

    fprintf(stdout, "SIMULAZIONE PRONTA A PARTIRE!\n");

    HalfStep<<<blocks,threads>>>(data.d_psi, data.d_phi);

    bitmap.anim_and_exit( (void (*)(uchar4*,void*,int))anim_gpu, (void (*)(void*))anim_exit );

    /*
    for(int t = 1; t <= T_MAX / DT; t++) {
        
        printf("\rTIME = %.6f / %.6f T.U.", t * DT, T_MAX);
        fflush(stdout);

        PseudoSpectralSolver(t, d_psi, d_phi, d_phi_k, d_k2, planZ2Z, planD2Z, planZ2D, d_X, d_factor, d_sin_k2dt2, d_cos_k2dt2, save, out1, out2, out3, out4, energy0, h_slice, d_slice);   
    }

    cudaFree(d_psi);
    cudaFree(d_phi);
    cudaFree(d_phi_k);
    cudaFree(d_k2);
    
    cufftDestroy(planZ2Z);
    cufftDestroy(planD2Z);
    cufftDestroy(planZ2D);

    cudaFree(d_X);
    cudaFree(d_factor);
    cudaFree(d_cos_k2dt2);
    cudaFree(d_sin_k2dt2);
    */

    if(SAVE_DENSITY_PLAN) fclose(out1);
    if(SAVE_ENERGY) fclose(out2);
    if(SAVE_DENSITY_LINE) fclose(out3);
    if(SAVE_POTENTIAL_PLAN) fclose(out4);
    cudaFree(d_slice);
    free(h_slice);

    fprintf(stdout, "\nSIMULAZIONE FINITA!\n");

    return 0;
}

/*************************FUNCTIONs DEFINITIONs******************************/

__global__ void SetSinCos(double *d_sin_k2dt2, double *d_cos_k2dt2, double *d_k2) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;
    d_sin_k2dt2[offset] = sin(-d_k2[offset] * DT * 0.5);
    d_cos_k2dt2[offset] = cos(-d_k2[offset] * DT * 0.5);
    
}
__global__ void HalfStep(cufftDoubleComplex *d_psi, double *d_phi)  {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;
    double a,b,c,d;

    a = d_psi[offset].x;
    b = d_psi[offset].y;

    c = cos(- d_phi[offset] * DT * 0.5);
    d = sin(- d_phi[offset] * DT * 0.5);

    d_psi[offset].x = a * c - b * d;
    d_psi[offset].y = a * d + b * c;
}
__global__ void HalfStepFourier(cufftDoubleComplex *d_psi, double *d_k2, double *d_sin_k2dt2, double *d_cos_k2dt2) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;
    double a,b,c,d;

    a = d_psi[offset].x;
    b = d_psi[offset].y;

    c = d_cos_k2dt2[offset];
    d = d_sin_k2dt2[offset];

    d_psi[offset].x = a * c - b * d;
    d_psi[offset].y = a * d + b * c;
}
__global__ void NormalizeZ(cufftDoubleComplex *d_psi) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;

    d_psi[offset].x /= NNN;
    d_psi[offset].y /= NNN;
}
__global__ void PsiToRho(cufftDoubleComplex *d_psi, double *d_rho) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;

    d_rho[offset] = 4 * PI * (d_psi[offset].x * d_psi[offset].x + d_psi[offset].y * d_psi[offset].y);
}
__global__ void PoissonFourier(cufftDoubleComplex *d_phi_k, double *d_factor) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * (N/2 + 1) + i * N * (N/2 + 1);
    int offset_k = k + j * N + i * N * N;

    d_phi_k[offset].x *= d_factor[offset_k];
    d_phi_k[offset].y *= d_factor[offset_k];
}
__global__ void NormalizeR(double *d_phi) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;

    d_phi[offset] /= NNN;
}
__global__ void CPotential(double *d_phi, double *d_X) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;

    double r = sqrt(SQR(d_X[i]) + SQR(d_X[j]) + SQR(d_X[k]));

    if(r > EPS) d_phi[offset] -= (CMASS / r);
}
__global__ void Step(cufftDoubleComplex *d_psi, double *d_phi)  {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;
    double a,b,c,d;

    a = d_psi[offset].x;
    b = d_psi[offset].y;

    c = cos(- d_phi[offset] * DT);
    d = sin(- d_phi[offset] * DT);

    d_psi[offset].x = a * c - b * d;
    d_psi[offset].y = a * d + b * c;

}
__global__ void Init(cufftDoubleComplex *d_psi, double *d_X, double delta_x, double *d_f, double *d_k2, double *d_factor) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int offset = k + j * N + i * N * N;

    d_psi[offset].x = 0.0;
    d_psi[offset].y = 0.0;

    for(int l = 0; l < N_OF_SOLITONS; l++) {

        double alpha = SQR(d_solitons[l].mass / 3.883);
        double r = sqrt(SQR(d_X[i]-d_solitons[l].x) + SQR(d_X[j]-d_solitons[l].y) + SQR(d_X[k] - d_solitons[l].z));

        int idx = (int)(sqrt(alpha) * (r / delta_x));
        
        if (sqrt(alpha) * r < 5.6) {                                                                                             
            d_psi[offset].x += alpha * d_f[idx] * cos(d_solitons[l].delta + d_solitons[l].vx*(d_X[i]-d_solitons[l].x) + d_solitons[l].vy*(d_X[j]-d_solitons[l].y) + d_solitons[l].vz*(d_X[k]-d_solitons[l].z)); 
            d_psi[offset].y += alpha * d_f[idx] * sin(d_solitons[l].delta + d_solitons[l].vx*(d_X[i]-d_solitons[l].x) + d_solitons[l].vy*(d_X[j]-d_solitons[l].y) + d_solitons[l].vz*(d_X[k]-d_solitons[l].z));
        }
    }

    // k² per il Laplaciano in Fourier
    int N2 = N/2;

    int kx = (i <= N2) ? i : i - N;
    int ky = (j <= N2) ? j : j - N;
    int kz = (k <= N2) ? k : k - N;

    double kx2 = 2 * PI * kx / L;
    double ky2 = 2 * PI * ky / L;
    double kz2 = 2 * PI * kz / L;

    double ksq = kx2*kx2 + ky2*ky2 + kz2*kz2;
    d_k2[offset] = ksq;

    if(d_k2[offset] > EPS) {
        d_factor[offset] = (-1./d_k2[offset]);
    } else {
        d_factor[offset] = 0.0;
    }
}
__global__ void GetSlice(cufftDoubleComplex *d_psi, double *d_slice) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = N/2 + j * N + i * N * N;
    int offset_slice = j + i * N;

    d_slice[offset_slice] = d_psi[offset].x * d_psi[offset].x + d_psi[offset].y * d_psi[offset].y;
}
void Initialize(cufftDoubleComplex *d_psi, double *d_phi, cufftDoubleComplex *d_phi_k, double *d_k2, cufftHandle planD2Z, cufftHandle planZ2D, double *d_X, double *d_factor) {
    
    //DO NOT CHANGE, THEY MUST MATCH "soliton_solution.cpp"
    double r_max = 9.0;
    double delta_x = 0.00001; 
    int size = r_max / delta_x;
    //

    FILE* file = fopen("soliton_profile.bin", "rb");
    double* f = (double*)malloc(size * sizeof(double));
    fread(f, sizeof(double), size , file);
    fclose(file);

    double *d_f;
    cudaMalloc(&d_f, size * sizeof(double));
    cudaMemcpy(d_f, f, size * sizeof(double), cudaMemcpyHostToDevice);
    free(f);

    dim3    blocks(N/8,N/8,N/8);
    dim3    threads(8,8,8);
    dim3    blocks2(N/8,N/8,(N/2 + 1));
    dim3    threads2(8,8,1);

    Init<<<blocks,threads>>>(d_psi, d_X, delta_x, d_f, d_k2, d_factor);

    cudaFree(d_f);

    PsiToRho<<<blocks,threads>>>(d_psi, d_phi);

    cufftExecD2Z(planD2Z, d_phi, d_phi_k);

    PoissonFourier<<<blocks2,threads2>>>(d_phi_k, d_factor);

    cufftExecZ2D(planZ2D, d_phi_k, d_phi);

    NormalizeR<<<blocks,threads>>>(d_phi);

    CPotential<<<blocks,threads>>>(d_phi, d_X);
}
void PseudoSpectralSolver(uchar4* outputBitmap, int t, cufftDoubleComplex *d_psi, double *d_phi, cufftDoubleComplex *d_phi_k, double *d_k2, cufftHandle planZ2Z, cufftHandle planD2Z, cufftHandle planZ2D, double *d_X, double *d_factor, double *d_sin_k2dt2, double *d_cos_k2dt2) {
    dim3    blocks(N/8,N/8,N/8); 
    dim3    threads(8,8,8);

    //TO DO: ottimizzare, insieme a PoissonFourier
    dim3    blocks2(N/8,N/8,(N/2 + 1));
    dim3    threads2(8,8,1);

    cufftExecZ2Z(planZ2Z, d_psi, d_psi, CUFFT_FORWARD);

    HalfStepFourier<<<blocks,threads>>>(d_psi, d_k2, d_sin_k2dt2, d_cos_k2dt2);

    cufftExecZ2Z(planZ2Z, d_psi, d_psi, CUFFT_INVERSE);

    NormalizeZ<<<blocks,threads>>>(d_psi);

    PsiToRho<<<blocks,threads>>>(d_psi, d_phi);

    cufftExecD2Z(planD2Z, d_phi, d_phi_k);

    PoissonFourier<<<blocks2,threads2>>>(d_phi_k, d_factor);

    cufftExecZ2D(planZ2D, d_phi_k, d_phi);

    NormalizeR<<<blocks,threads>>>(d_phi);

    CPotential<<<blocks,threads>>>(d_phi, d_X);

    if(1) {

        HalfStep<<<blocks,threads>>>(d_psi, d_phi);

        dim3    blocks3(N/8,N/8);
        dim3    threads3(8,8);
    
        field_to_color<<<blocks3,threads3>>>( outputBitmap, d_psi );

        HalfStep<<<blocks,threads>>>(d_psi, d_phi);

    } else {
        Step<<<blocks,threads>>>(d_psi, d_phi);
    }
    

}
