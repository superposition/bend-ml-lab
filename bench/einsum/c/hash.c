#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static inline uint32_t prng(uint32_t x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static void gen(float*a,size_t n,uint32_t s){for(size_t i=0;i<n;i++){s=prng(s);a[i]=(float)(s>>8)/16777216.0f;}}
static void gen2(float*a,size_t n,uint32_t s){for(size_t i=0;i<n;i++){s=prng(s);a[i]=2.0f*((float)(s>>8)/16777216.0f)-1.0f;}}
static uint32_t hash_bits(const float*a,size_t n){uint32_t h=2166136261u;for(size_t i=0;i<n;i++){uint32_t b;memcpy(&b,&a[i],4);h=(h^b)*16777619u;}return h;}
int main(){size_t n=64,cells=n*n;float*a=malloc(cells*4),*b=malloc(cells*4),*c=malloc(cells*4);gen(a,cells,7);gen(b,cells,11);
 for(size_t i=0;i<n;i++)for(size_t j=0;j<n;j++){float s=0;for(size_t k=0;k<n;k++)s+=a[i*n+k]*b[k*n+j];c[i*n+j]=s;}
 printf("seq,%u\n",hash_bits(c,cells));
 gen2(a,cells,13);gen2(b,cells,17);
 for(size_t i=0;i<n;i++)for(size_t j=0;j<n;j++){float s=0;for(size_t k=0;k<n;k++)s+=a[i*n+k]*b[k*n+j];c[i*n+j]=s;}
 printf("neg,%u\n",hash_bits(c,cells));
 free(a);free(b);free(c);return 0;}
